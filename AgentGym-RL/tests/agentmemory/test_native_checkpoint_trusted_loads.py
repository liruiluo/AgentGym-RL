import ast
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import dill
import numpy as np
import torch

from verl.utils.agent_dataset.rl_dataset import RLHFDataset


_REPO_ROOT = Path(__file__).resolve().parents[2]
_FSDP_CHECKPOINT_MANAGER = (
    _REPO_ROOT / "verl/utils/checkpoint/fsdp_checkpoint_manager.py"
)
_RAY_TRAINER = _REPO_ROOT / "verl/agent_trainer/ppo/ray_trainer.py"


class _StatefulDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.values = torch.arange(6)
        self.cursor = 3
        self.resume_calls = 0

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def resume_dataset_state(self):
        self.resume_calls += 1


def _torch_load_assignments(path):
    assignments = {}
    module = ast.parse(path.read_text())
    for node in ast.walk(module):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        function = node.value.func
        if not (
            isinstance(function, ast.Attribute)
            and function.attr == "load"
            and isinstance(function.value, ast.Name)
            and function.value.id == "torch"
        ):
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            name = target.id
        elif isinstance(target, ast.Attribute):
            name = target.attr
        else:
            continue
        assignments[name] = node.value
    return assignments


def _keyword_literal(call, name):
    keyword = next((item for item in call.keywords if item.arg == name), None)
    return None if keyword is None else ast.literal_eval(keyword.value)


class NativeCheckpointTrustedLoadTest(unittest.TestCase):
    def test_legacy_rlhf_dataset_without_procedural_attribute_can_resume(self):
        dataset = RLHFDataset.__new__(RLHFDataset)
        dataset.original_data_file = "legacy.json"
        dataset.serialize_dataset = True
        self.assertFalse(hasattr(dataset, "procedural_index_source"))

        with mock.patch.object(
            RLHFDataset,
            "_read_files_and_tokenize",
        ) as read_files:
            dataset.resume_dataset_state()

        self.assertFalse(dataset.serialize_dataset)
        read_files.assert_called_once_with()

    def test_dill_dataloader_checkpoint_round_trips_resume_state(self):
        generator = torch.Generator().manual_seed(17)
        loader = torch.utils.data.DataLoader(
            _StatefulDataset(),
            batch_size=2,
            shuffle=True,
            generator=generator,
        )
        generator_state = generator.get_state().clone()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.pt"
            torch.save(loader, path, pickle_module=dill)
            with self.assertRaises((pickle.UnpicklingError, RuntimeError)):
                torch.load(path, weights_only=True)
            loaded = torch.load(path, weights_only=False)

        self.assertIsInstance(loaded, torch.utils.data.DataLoader)
        self.assertEqual(loaded.dataset.cursor, 3)
        self.assertEqual(loaded.dataset.resume_calls, 0)
        self.assertIsInstance(loaded.sampler, torch.utils.data.RandomSampler)
        torch.testing.assert_close(loaded.generator.get_state(), generator_state)
        torch.testing.assert_close(loaded.sampler.generator.get_state(), generator_state)
        loaded.dataset.resume_dataset_state()
        self.assertEqual(loaded.dataset.resume_calls, 1)

    def test_numpy_rng_extra_state_round_trips(self):
        rng_state = np.random.RandomState(7).get_state()
        payload = {
            "lr_scheduler": {"last_epoch": 1},
            "rng": {"numpy": rng_state},
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "extra_state.pt"
            torch.save(payload, path)
            loaded = torch.load(path, weights_only=False)

        self.assertEqual(loaded["lr_scheduler"], payload["lr_scheduler"])
        loaded_rng = loaded["rng"]["numpy"]
        self.assertEqual(loaded_rng[0], rng_state[0])
        np.testing.assert_array_equal(loaded_rng[1], rng_state[1])
        self.assertEqual(loaded_rng[2:], rng_state[2:])

    def test_only_trusted_pickle_loads_disable_weights_only(self):
        fsdp_loads = _torch_load_assignments(_FSDP_CHECKPOINT_MANAGER)
        trainer_loads = _torch_load_assignments(_RAY_TRAINER)

        explicit_trusted_loads = {
            f"fsdp:{name}"
            for name, call in fsdp_loads.items()
            if _keyword_literal(call, "weights_only") is False
        } | {
            f"trainer:{name}"
            for name, call in trainer_loads.items()
            if _keyword_literal(call, "weights_only") is False
        }
        self.assertEqual(
            explicit_trusted_loads,
            {"fsdp:extra_state_dict", "trainer:train_dataloader"},
        )

    def test_model_and_optimizer_loads_keep_weights_only_default(self):
        fsdp_loads = _torch_load_assignments(_FSDP_CHECKPOINT_MANAGER)

        self.assertIsNone(
            _keyword_literal(fsdp_loads["model_state_dict"], "weights_only")
        )
        self.assertIsNone(
            _keyword_literal(fsdp_loads["optimizer_state_dict"], "weights_only")
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
