# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

import pandas as pd

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils.model import compute_position_id_with_mask
from verl.utils import hf_tokenizer
from verl.utils.agent_dataset.agent_action_schema import (
    validate_agent_action_record,
)


SFT_DATA_MODES = ("conversations", "agent_action_v1")


class SFTDataset(Dataset):
    """
    This is an in-memory SFTDataset
    """

    def __init__(self,
                 json_file: str,
                 tokenizer,
                 prompt_key='conversations',
                 max_length=4096,
                 truncation='right',
                 data_mode='conversations'):
        assert truncation in ['error', 'left', 'right']
        if data_mode not in SFT_DATA_MODES:
            raise ValueError(
                f"data_mode must be one of {SFT_DATA_MODES!r}, got {data_mode!r}"
            )
        self.truncation = truncation
        self.data_mode = data_mode

        self.json_file = json_file
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.prompt_key = prompt_key

        self.max_length = max_length

        self._read_files_and_tokenize()

    def _read_files_and_tokenize(self):
        self.dataframe = pd.read_json(self.json_file)
        if self.data_mode == 'conversations':
            self.prompts = self.dataframe[self.prompt_key].tolist()
        else:
            self.prompts = self.dataframe.to_dict(orient='records')

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        tokenizer = self.tokenizer

        prompt = self.prompts[item]

        if self.data_mode == 'agent_action_v1':
            input_ids, attention_mask, loss_mask = self._tokenize_agent_action(
                prompt
            )
        else:
            input_ids, attention_mask, loss_mask = self._tokenize_conversations(
                prompt
            )

        input_ids, attention_mask, loss_mask = self._pad_or_truncate(
            input_ids, attention_mask, loss_mask
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'loss_mask': loss_mask
        }

    def _tokenize_agent_action(self, record):
        fields = validate_agent_action_record(record)
        messages = [
            {'role': 'system', 'content': fields['system_prompt']},
            {'role': 'user', 'content': fields['observation']},
        ]
        encoded = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=False,
        )
        prompt_ids = _normalize_token_ids(encoded, field='generation prompt')
        action_ids = _normalize_token_ids(
            self.tokenizer.encode(
                fields['assistant_action'], add_special_tokens=False
            ),
            field='assistant action',
        )
        terminator_ids = _normalize_token_ids(
            self.tokenizer.encode('<|im_end|>', add_special_tokens=False),
            field='assistant terminator',
        )
        if not prompt_ids or not action_ids or not terminator_ids:
            raise ValueError(
                'agent_action_v1 prompt, action, and terminator must tokenize '
                'to non-empty sequences'
            )
        input_ids = torch.tensor(
            prompt_ids + action_ids + terminator_ids, dtype=torch.long
        )
        attention_mask = torch.ones_like(input_ids)
        loss_mask = torch.tensor(
            [0] * len(prompt_ids)
            + [1] * (len(action_ids) + len(terminator_ids)),
            dtype=torch.long,
        )
        return input_ids, attention_mask, loss_mask

    def _tokenize_conversations(self, prompt):
        tokenizer = self.tokenizer

        # string
        system_chat_dict = {'role': 'system', 'content': ''}
        system_chat_str = tokenizer.apply_chat_template([system_chat_dict], tokenize=False)
        prompt_ids_output = tokenizer(tokenizer.apply_chat_template([{'role': 'user', 'content': prompt[0]['value']}], tokenize=False), return_tensors='pt', add_special_tokens=False)
        input_ids = prompt_ids_output['input_ids'][0]
        attention_mask = prompt_ids_output['attention_mask'][0]
        loss_mask = torch.zeros_like(input_ids)
        for c in prompt[1:]:
            if c['from'] == 'system':
                prompt_ids_output = tokenizer(tokenizer.apply_chat_template([system_chat_dict, {'role': 'system', 'content': c['value']}], tokenize=False).replace(system_chat_str, ""), return_tensors='pt', add_special_tokens=False)
                input_ids = torch.concat([input_ids, prompt_ids_output['input_ids'][0]])
                attention_mask = torch.concat([attention_mask, prompt_ids_output['attention_mask'][0]])
                loss_mask = torch.cat([loss_mask, torch.zeros_like(prompt_ids_output['input_ids'][0])])
            elif c['from'] == 'human':
                prompt_ids_output = tokenizer(tokenizer.apply_chat_template([system_chat_dict, {'role': 'user', 'content': c['value']}], tokenize=False).replace(system_chat_str, ""), return_tensors='pt', add_special_tokens=False)
                input_ids = torch.concat([input_ids, prompt_ids_output['input_ids'][0]])
                attention_mask = torch.concat([attention_mask, prompt_ids_output['attention_mask'][0]])
                loss_mask = torch.cat([loss_mask, torch.zeros_like(prompt_ids_output['input_ids'][0])])
            elif c['from'] == 'gpt':
                prompt_ids_output = tokenizer(tokenizer.apply_chat_template([system_chat_dict, {'role': 'assistant', 'content': c['value']}], tokenize=False).replace(system_chat_str, ""), return_tensors='pt', add_special_tokens=False)
                input_ids = torch.concat([input_ids, prompt_ids_output['input_ids'][0]])
                attention_mask = torch.concat([attention_mask, prompt_ids_output['attention_mask'][0]])
                loss_mask = torch.cat([loss_mask, torch.ones_like(prompt_ids_output['input_ids'][0])])
            else:
                raise NotImplementedError

        return input_ids, attention_mask, loss_mask

    def _pad_or_truncate(self, input_ids, attention_mask, loss_mask):
        tokenizer = self.tokenizer
        supervised_tokens = int(torch.sum(loss_mask).item())

        # padding to max length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = torch.ones(size=(self.max_length - sequence_length,),
                                          dtype=input_ids.dtype) * tokenizer.pad_token_id
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)
            padded_loss_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=loss_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
            loss_mask = torch.cat((loss_mask, padded_loss_mask))
        elif sequence_length > self.max_length:
            if self.truncation == 'left':
                # actually, left truncation may not be reasonable
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
                loss_mask = loss_mask[-self.max_length:]
            elif self.truncation == 'right':
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
                loss_mask = loss_mask[:self.max_length]
            elif self.truncation == 'error':
                raise NotImplementedError(f'{sequence_length=} is larger than {self.max_length=}')
            else:
                raise NotImplementedError(f'Unknown truncation method {self.truncation}')
        if (
            self.data_mode == 'agent_action_v1'
            and int(torch.sum(loss_mask).item()) != supervised_tokens
        ):
            raise ValueError(
                'agent_action_v1 truncation removed supervised action tokens; '
                'increase max_length or use left truncation with enough target space'
            )
        if not torch.any(loss_mask):
            raise ValueError(
                'SFT sample has no supervised target tokens after truncation; '
                'increase max_length or use left truncation'
            )
        return input_ids, attention_mask, loss_mask


def _normalize_token_ids(encoded, *, field):
    if isinstance(encoded, dict) or (
        hasattr(encoded, '__contains__')
        and hasattr(encoded, '__getitem__')
        and 'input_ids' in encoded
    ):
        encoded = encoded['input_ids']
    if hasattr(encoded, 'tolist'):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], (list, tuple)):
        if len(encoded) != 1:
            raise ValueError(f'{field} produced a batch instead of one sequence')
        encoded = encoded[0]
    token_ids = list(encoded)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in token_ids):
        raise TypeError(f'{field} produced non-integer token ids')
    return token_ids
