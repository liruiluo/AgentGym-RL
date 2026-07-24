#!/usr/bin/env python3
"""Evidence-first AgentMemoryGym evaluation against HTTP services.

This driver intentionally has no dependency on torch, transformers, or the
training stack.  It mirrors the HTTP behavior of
``agentenv.envs.agentmemory.AgentMemoryEnvClient`` and calls an
OpenAI-compatible vLLM server one policy turn at a time.

The evaluator is deliberately strict about prompt evidence: every turn must
be tokenized by the server's ``/tokenize`` endpoint.  A local tokenizer or a
tokenized copy of the chat response is never substituted for server evidence.
Response token ids are marked exact only when the completion response carries
an explicit integer id sequence (the usual OpenAI response does not).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


FORMAL_SCHEMA_V3 = "agentmemory_formal_step_v3"
WEBSHOP_V2_SURFACE = "memoryarena_webshop_native_v1"
TRAVEL_FAILFAST_SURFACE = "memoryarena_travel_planner_failfast_one_action_v3"
TRAVEL_PAPER_EVAL_SURFACE = (
    "memoryarena_travel_planner_paper_eval_one_action_v3"
)
SEARCH_FAILFAST_SURFACE = (
    "memoryarena_progressive_search_failfast_public221_one_action_v3"
)
SEARCH_PAPER_EVAL_SURFACE = (
    "memoryarena_progressive_search_paper_eval_public221_one_action_v3"
)
MATH_FAILFAST_SURFACE = "memoryarena_formal_reasoning_math_failfast_v3"
PHYS_FAILFAST_SURFACE = "memoryarena_formal_reasoning_phys_failfast_v3"
MATH_PAPER_EVAL_SURFACE = (
    "memoryarena_formal_reasoning_math_paper_eval_one_action_v3"
)
PHYS_PAPER_EVAL_SURFACE = (
    "memoryarena_formal_reasoning_phys_paper_eval_one_action_v3"
)
# Compatibility aliases for existing fail-fast callers and launchers.
MATH_SURFACE = MATH_FAILFAST_SURFACE
PHYS_SURFACE = PHYS_FAILFAST_SURFACE
EVAL_SCHEMA = "agentmemory_eval_evidence_v1"
PAPER_SUCCESS_COLUMNS = ("Shopping", "Travel", "Search", "Math", "Physics")
MEMORYARENA_HF_REPO = "ZexueHe/memoryarena"
MEMORYARENA_HF_REVISION = "da1a37c8b19280e18627ca01cf368195a5e1d92e"
MEMORYARENA_FROZEN_DATASETS = {
    "formal_reasoning_math": {
        "repo_path": "formal_reasoning_math/data.jsonl",
        "sha256": "ff5b0ad575847c7476a02d1e35661592a833bd0cff384cb54bc6f35b46de7803",
        "record_count": 40,
        "phase_count": 354,
        "phase_field": "questions",
    },
    "formal_reasoning_phys": {
        "repo_path": "formal_reasoning_phys/data.jsonl",
        "sha256": "580862006af2ff2bfc8c5d2d2b9a60bf33a46cbb64f27d60a2bfe039aec61cf6",
        "record_count": 20,
        "phase_count": 86,
        "phase_field": "questions",
    },
    "group_travel_planner": {
        "repo_path": "group_travel_planner/data.jsonl",
        "sha256": "2f955d444f6f3ad3c5da2064359ab19f8fc1f90621ff9d00723a450a009c3732",
        "record_count": 270,
        "phase_count": 1869,
        "phase_field": "questions",
    },
    "progressive_search": {
        "repo_path": "progressive_search/data.jsonl",
        "sha256": "b445ee36fa3ccb9ad08eae9e7adda86bbc64f14f1e2a0682a8b2085cdb8e4c0e",
        "record_count": 221,
        "phase_count": 1641,
        "phase_field": "questions",
    },
}
TRAVEL_PAPER_METRIC_CONTRACT = "memoryarena_travel_eval_py_ps_sps_sr_v1"
TRAVEL_PAPER_DATASET_SCOPE = "memoryarena_group_travel_planner_frozen270"
SEARCH_PAPER_METRIC_CONTRACT = (
    "memoryarena_progressive_search_ps_sr_at_k_final_sr_v1"
)
SEARCH_PAPER_DATASET_SCOPE = "public221_of_paper256"
FORMAL_PAPER_METRIC_CONTRACT = "memoryarena_formal_reasoning_ps_final_sr_v1"
FORMAL_PAPER_DATASET_SCOPES = {
    MATH_PAPER_EVAL_SURFACE: "memoryarena_formal_reasoning_math_frozen40",
    PHYS_PAPER_EVAL_SURFACE: "memoryarena_formal_reasoning_phys_frozen20",
}
PAPER_MACRO5_METRIC_CONTRACT = "memoryarena_paper_macro5_v1"
TRAVEL_RECORD_COUNT = 270
TRAVEL_PHASE_COUNT = 1869
SEARCH_RECORD_COUNT = 221
SEARCH_PHASE_COUNT = 1641
SEARCH_PAPER_TASK_COUNT = 256
SEARCH_TASK_PHASE_COUNTS = (
    9, 12, 6, 11, 5, 9, 5, 6, 6, 8, 4, 7, 10, 14, 9, 5, 9, 6, 5, 6,
    10, 8, 8, 9, 5, 8, 5, 9, 8, 9, 7, 7, 6, 8, 5, 11, 9, 6, 4, 8, 8,
    8, 5, 6, 6, 7, 7, 6, 6, 8, 7, 8, 8, 12, 8, 4, 11, 5, 11, 8, 8,
    5, 7, 13, 8, 6, 6, 6, 6, 7, 8, 10, 6, 7, 10, 9, 6, 7, 9, 10, 11,
    7, 7, 10, 6, 7, 9, 6, 9, 6, 12, 7, 7, 5, 7, 5, 5, 8, 6, 8, 8,
    6, 5, 8, 6, 7, 7, 7, 9, 8, 7, 7, 8, 7, 7, 11, 5, 8, 6, 5, 11,
    5, 9, 6, 8, 6, 11, 7, 8, 6, 5, 7, 7, 9, 8, 6, 8, 10, 7, 6, 7,
    5, 5, 6, 8, 7, 5, 7, 7, 5, 8, 7, 8, 5, 8, 10, 5, 5, 7, 9, 6,
    16, 5, 8, 7, 7, 7, 7, 10, 7, 8, 4, 8, 6, 6, 7, 9, 11, 12, 9, 7,
    8, 7, 6, 7, 4, 8, 5, 6, 7, 8, 9, 9, 10, 4, 8, 6, 6, 6, 11, 10,
    6, 7, 5, 8, 8, 8, 6, 5, 10, 8, 11, 4, 7, 14, 6, 8, 8, 5, 10, 6,
)
if (
    len(SEARCH_TASK_PHASE_COUNTS) != SEARCH_RECORD_COUNT
    or sum(SEARCH_TASK_PHASE_COUNTS) != SEARCH_PHASE_COUNT
):  # pragma: no cover - immutable evaluator constants
    raise RuntimeError("Search frozen row map disagrees with public221 totals")
SEARCH_MEMORYARENA_COMMIT = "6cd9de14b71915e39ac742a20dc33785e14b6aab"
SEARCH_PAPER_CONTRACT_ID = (
    "memoryarena_progressive_search_paper_eval_public221_"
    "one_action_v3_20260723"
)
SEARCH_PAPER_SYSTEM_PROMPT_SHA256 = (
    "4f2f7c36be24d95427c2a9d0667f2a859ba75e3236d9392c8d3a145b66108275"
)
SEARCH_PAPER_CONTRACT_SHA256 = (
    "50cee77b16d7b23cc5a054b8992e068a0c3400e596edd426febccc0fcbad343f"
)
SEARCH_FROZEN_EMBEDDING_MODEL = "text-embedding-3-small"
SEARCH_FROZEN_DOCUMENT_COUNT = 100195
SEARCH_FROZEN_INDEX_DIMENSION = 1536
SEARCH_FROZEN_INDEX_REPOSITORY = "Joanna690/websearch-embeddings"
SEARCH_FROZEN_INDEX_REVISION = "7a784780b46d16ddc926aed9b63c34def2014c47"
SEARCH_FROZEN_CORPUS_REPOSITORY = "Tevatron/browsecomp-plus-corpus"
SEARCH_FROZEN_CORPUS_REVISION = "b27b02bc3e45511b8b82a13e6f90ce761df726f6"
SEARCH_FROZEN_CORPUS_SHA256 = (
    "6b306573f6194367d5e2a7daaae12d9cb4242409413f261ea6d81a19d7cf4b26"
)
SEARCH_FROZEN_CORPUS_MANIFEST_SHA256 = (
    "ba722b74477fd88e2d7745b391deecabc1f823a997c4908c3378a29e90d085aa"
)
SEARCH_FROZEN_INDEX_SHARDS = (
    {
        "name": "shard0.index",
        "index_sha256": "57c7b94af14d6d84445da99d207f52dd044f275c6d6142353c88b19e0d938956",
        "id_map_sha256": "1d65ae1fa019cc8b61ffc29d0a72adf1567ebef1b01a735697eb61019719d1e4",
        "vector_count": 25049,
    },
    {
        "name": "shard1.index",
        "index_sha256": "d253728c07e641e0d9ec1dbf992a519db2c1cf4808cbf8167bf3bdc40df5c41f",
        "id_map_sha256": "4d1c2272ebb9d359cfab6cb07f7affd8e78a3823335b42aa2bf3cd39e64e4150",
        "vector_count": 25049,
    },
    {
        "name": "shard2.index",
        "index_sha256": "22726188a49f92bbcddf14bee9d00b3bb69d11eb01e5995b52c9c79a8d111ca9",
        "id_map_sha256": "dd4e3014335e41c979984bdbbb79bc20ee554ad5c6e70a59209a395080deffbd",
        "vector_count": 25049,
    },
    {
        "name": "shard3.index",
        "index_sha256": "652eb57c1c0ab498d7b53253ebdd21591be394e5d546650ca6027774ddc4845c",
        "id_map_sha256": "334aa558fbe525b9be2db23a9338446fdb07c8713ab033d8fe7871ed19233d9c",
        "vector_count": 25048,
    },
)
SEARCH_FROZEN_CORPUS_SOURCE_SHARDS = (
    ("train-00000-of-00007.parquet", "7c07f9e23b1ca548110fd831714cadc67d44db5223bace6e45fcaa795d3153d0", 218510750, 14314),
    ("train-00001-of-00007.parquet", "e92d8202e0f656a85b262153dbcd22ecf80ea2d0c96d9884f9c8e25480b869ab", 271758203, 14314),
    ("train-00002-of-00007.parquet", "0e4113a4503342527258d8f2c49877747435f3e65bfe1f7306b4f488c8d225fe", 257605847, 14314),
    ("train-00003-of-00007.parquet", "0ceea5e703332a2e3ce700f641273400d84583fad84b659d3248ed06d3a9fef3", 239125125, 14314),
    ("train-00004-of-00007.parquet", "15b62914ddc3de6946893c770f07d5d84d29646e833ca1447955668f2b57940c", 259072824, 14313),
    ("train-00005-of-00007.parquet", "a9a75708ad37c522e93a774e5a968a3129e12b0559971c8f950a5628e0201df0", 257706279, 14313),
    ("train-00006-of-00007.parquet", "290062b60c1a6ebba7d5469a37a431f0a2596e68788295284b1b2d35db07b62c", 257801458, 14313),
)
SEARCH_UPSTREAM_SOURCE_FILES_SHA256 = {
    "agent/__init__.py": "85d235b457008048be08e1a04ed61a26767f56334ff4d0a8030515fc11083a69",
    "agent/base.py": "3a7edf6b0f96e1793aef3d658e586c078ebc6a67b0ab0bc4239ba64370571b26",
    "agent/base_agent.py": "61eac895f6e8d4bc347898898bb7c431d8a6f7ee2f37377ede747a9620a6e28c",
    "agent/math.py": "33faff8e13f7c12d69d375f319fa151102a8268b8a52d1795304a325e2e85141",
    "agent/search.py": "de74f03af2dfc51ae000b861326a54df07f989e65d017f4491c831ed648843b6",
    "agent/travel_planner.py": "e96c4ef040156a2d9e6ac7c6598b685e4779adb44fb468bcbe48baa9f982b93f",
    "agent/webshop.py": "b346bf2d7d4786770848c95c1159a87a00bea5cdb2724fd062c223311a7e1918",
    "env/__init__.py": "d8c76f5f3b7e9f6a213ae19d9d0ec0a7afce7ad632ab43dc69079049e3fa0703",
    "env/env_systems/__init__.py": "9a27be753adf9f8b55aabb1f7d3eb2207ed0511469bf6d563d4140d33fc909d5",
    "env/env_systems/base_env.py": "3c7a2dcd6595f3ec39c06e1fefcaaf8903d0848fb17e8bdf56daaf35c46a6278",
    "env/env_systems/browsecomp_plus_env.py": "d74c1eabf7d5555915c2d07606c4e51784527c56d460a3def4c73270b80a82f6",
    "env/env_systems/web_search_env/download_embeddings_from_hf.py": "ae5a82c3c275017e38d6ed8a1235b9641aa27c6870288464a75dc5e6a59ef18c",
    "env/env_systems/web_search_env/evaluate_with_openai.py": "eebc3d0bd8ab8d39e7240a9e60714e26789b391814d5f9c6309102088cf3a760",
    "env/env_systems/web_search_env/scripts_build_index/build_openai_embedding_index.py": "c4dd386cf078cbd4159b23ae443dff0dcff0e42ebaa8c978c5781c3177728a64",
    "env/env_systems/web_search_env/scripts_build_index/decrypt_dataset.py": "d25c7565ee7c1d9c8ec37f610b1f67756db8f8924ac6b8175ffff6d261a0e32d",
    "env/env_systems/web_search_env/search_agent/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "env/env_systems/web_search_env/search_agent/anthropic_client.py": "256c0b15361233a9980982795e5055b216a1e1c0b0127578746e01f74561dd42",
    "env/env_systems/web_search_env/search_agent/gemini_client.py": "69be544788a382e77bd4b8c5d900f96a4eec445fe0624c0c206c60a675487944",
    "env/env_systems/web_search_env/search_agent/openai_client.py": "c6c0b280b01c3ee6294eda2f2854bd752d5a106f6aef88380c39d9db956692c5",
    "env/env_systems/web_search_env/search_agent/openai_client_with_mcp.py": "c7b2a09ab97eaf30f7528c4d142dd7a8ed2e4988d771f2251514beb33e63b09f",
    "env/env_systems/web_search_env/search_agent/oss_client.py": "65ffe77b7c2b3a42ce9a87aa2d728a4a8badb22ba8bc2a2f9c6eb7477070e7e0",
    "env/env_systems/web_search_env/search_agent/prompts.py": "cf974be9bd61d52919411e89a37776d6e8365ab22f42262b19fe0d6c4125f8a4",
    "env/env_systems/web_search_env/search_agent/qwen_client.py": "b4419daac509d8ea5e7223628b3ab4ba215e36a2684c7e293f88e54ec5093d88",
    "env/env_systems/web_search_env/search_agent/tongyi_utils/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "env/env_systems/web_search_env/search_agent/tongyi_utils/react_agent.py": "0e2c0832a41f0fd77c9536bb49ae6ca2232dc8e28bcd71606bd3522debc9a066",
    "env/env_systems/web_search_env/search_agent/tongyi_utils/tool_search.py": "3397ae48a920ee349b1eb1878a649a839cedc3ddaa93a62323b64e6f8273873d",
    "env/env_systems/web_search_env/search_agent/utils.py": "3a6d9bc4956deeafce4a4515c9d025e3bf1d3e1dbc97945413b84e87d2bc3e8f",
    "env/env_systems/web_search_env/searcher/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "env/env_systems/web_search_env/searcher/mcp_server.py": "c6c7082f93c8483d77d07e920827d74c6e29f9099fd65f2cfcdfe63714d78cc3",
    "env/env_systems/web_search_env/searcher/search_r1_server.py": "a02cf1c81bf33c33f06ae3f64f89ea8153937d5af8cece3812a35f3549c8560a",
    "env/env_systems/web_search_env/searcher/searchers/__init__.py": "2641bf9f069e01f13f07be6668649c4de33bebdbff384a29cc7ec0125abcc49a",
    "env/env_systems/web_search_env/searcher/searchers/base.py": "870e646f9dba8f2f99b1bbf596e6e8aa3ed9ba52127a60ffbc64b781bae815e0",
    "env/env_systems/web_search_env/searcher/searchers/bm25_searcher.py": "b2bf9174def9d6df582c8e370c1c5e0fad2915fd64242b7afed74e74bcc3322f",
    "env/env_systems/web_search_env/searcher/searchers/custom_searcher.py": "78dcbbafba1d5839bf0970e61e6ef3cc0aabd63474c023e6f8e3724817f58c66",
    "env/env_systems/web_search_env/searcher/searchers/faiss_searcher.py": "78234e7764fe08d6893ccf81037597662ce61ca4478790fc69c076f729a3977d",
    "env/env_systems/web_search_env/searcher/searchers/openai_searcher.py": "0de0263e9b60c5a29b5c6b81c37b0787df8e983c4f8b1d702749bcf9d39f0a12",
    "env/env_systems/web_search_env/searcher/tools.py": "75faed3bb31046fa45a8924bce229af879d9b5c1a496509aa17ef175103caa35",
    "env/env_systems/web_search_env/upload_embeddings_to_hf.py": "74b130e48e4342442922efb0179e57789d40524f0cdb6606fc05dea953e95612",
    "run_search.py": "f58ee0eb1e7ad209f251f11ee57949a9e6077e4296b04f1177479e89041ac431",
}
SEARCH_UPSTREAM_SOURCE_BUNDLE_SHA256 = (
    "5f74c1848fc401ee604db8e68f6a221a2f90d8f9f80a7236e12eb1c79b58a363"
)
SEARCH_UPSTREAM_SOURCE_FILE_COUNT = 39
SEARCH_JUDGE_MODEL = "gpt-4.1"
SEARCH_JUDGE_MAX_TOKENS = 8000
SEARCH_JUDGE_PROMPT_TEMPLATE_SHA256 = (
    "0f0023ee579b8c134f1834ed8952778b9e01460e31d47c242ee3629da9d44835"
)
SEARCH_SNIPPET_TOKENIZER = {
    "repository": "Qwen/Qwen3-0.6B",
    "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
    "local_files_only": True,
    "files_sha256": {
        "config.json": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
        "generation_config.json": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
        "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
        "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
        "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    },
    "bundle_sha256": "62851e5e39395f893633e2283ace53d5b223896d0058c751fa086f81c7a4f187",
}
SHOPPING_BUNDLE_COUNT = 150
SHOPPING_SESSIONS_PER_BUNDLE = 6
SHOPPING_SESSION_COUNT = 900
SHOPPING_RAW_DATASET_SHA256 = (
    "4411a2da528a33dc6aca519b49cc225895363f18b2d19b191fddb501200134ef"
)
SHOPPING_MEMORYARENA_COMMIT = "6cd9de14b71915e39ac742a20dc33785e14b6aab"
SHOPPING_DOMAIN_DATA_SHA256 = (
    "2576aa9637ab6691c14f26e5f0b022b3a16c325a312ebc856c271f8e641f2afc"
)
SHOPPING_SPLIT_STRATEGY = "source_position_mod10_8_1_1_v1"
SHOPPING_SPLIT_MANIFEST_SHA256 = (
    "fc694c3d9f62845556cfa18a16357ee91d4587c9a9dbfd78f51f9c6e461216f7"
)
SHOPPING_TASK_IDS_SHA256 = (
    "40038828983707b2cee5369d981d8b5f37cfe4b965a416d19576bcce619f5b0a"
)
SHOPPING_TASK_CATEGORIES = ("baking", "beauty", "electronics", "grocery", "home")
SHOPPING_DATASET_PROVENANCE_FIELDS = frozenset(
    {
        "schema",
        "raw_dataset_path",
        "raw_dataset_sha256",
        "memoryarena_commit",
        "domain_data_sha256",
        "action_surface_version",
        "split_strategy",
        "split_manifest_sha256",
        "split_counts",
        "bundle_count",
        "sessions_per_bundle",
        "session_count",
        "target_asin_membership_verified",
    }
)
# Row order and question counts are part of each frozen JSONL byte contract.
# Binding both prevents a complete-looking panel assembled from the wrong rows.
MATH_TASK_PHASE_COUNTS = (
    5, 5, 8, 10, 11, 6, 10, 5, 13, 7,
    6, 8, 9, 6, 7, 9, 16, 8, 7, 8,
    13, 14, 7, 9, 12, 10, 13, 12, 9, 6,
    9, 8, 15, 11, 7, 11, 7, 4, 11, 2,
)
PHYS_TASK_PHASE_COUNTS = (
    3, 12, 8, 3, 4, 3, 3, 2, 5, 4,
    4, 3, 4, 9, 3, 3, 2, 4, 4, 3,
)
FORMAL_TASK_PHASE_COUNTS = {
    MATH_FAILFAST_SURFACE: MATH_TASK_PHASE_COUNTS,
    PHYS_FAILFAST_SURFACE: PHYS_TASK_PHASE_COUNTS,
    MATH_PAPER_EVAL_SURFACE: MATH_TASK_PHASE_COUNTS,
    PHYS_PAPER_EVAL_SURFACE: PHYS_TASK_PHASE_COUNTS,
}
FORMAL_SURFACE_DATASETS = {
    MATH_FAILFAST_SURFACE: "formal_reasoning_math",
    PHYS_FAILFAST_SURFACE: "formal_reasoning_phys",
    MATH_PAPER_EVAL_SURFACE: "formal_reasoning_math",
    PHYS_PAPER_EVAL_SURFACE: "formal_reasoning_phys",
}
FORMAL_MEMORYARENA_COMMIT = "6cd9de14b71915e39ac742a20dc33785e14b6aab"
FORMAL_ENV_GIT_TREE_OID = "5576f4aaa4bf17a2a807650635ce335b8c620d32"
FORMAL_SELECTED_SOURCE_BUNDLE_SHA256 = (
    "05d83f61539f4b2b48b5bc67734deda36d7220f68233ed29592e00250001ff5d"
)
FORMAL_RUNTIME_SOURCE_FILES_SHA256 = {
    "env/__init__.py": "d8c76f5f3b7e9f6a213ae19d9d0ec0a7afce7ad632ab43dc69079049e3fa0703",
    "env/env_client.py": "6d9a25b0b8c75a3e2d76abedc6e2c140d09c1202688996840fe756ba10f537a1",
    "env/env_systems/__init__.py": "9a27be753adf9f8b55aabb1f7d3eb2207ed0511469bf6d563d4140d33fc909d5",
    "env/env_systems/base_env.py": "3c7a2dcd6595f3ec39c06e1fefcaaf8903d0848fb17e8bdf56daaf35c46a6278",
    "env/env_systems/webshop_env.py": "8dc933f5eb703679d16cc2d7a60c5616dc4d855538e883fa41adddde59d0c033",
    "env/env_systems/browsecomp_plus_env.py": "d74c1eabf7d5555915c2d07606c4e51784527c56d460a3def4c73270b80a82f6",
    "env/env_systems/math_env.py": "c5d45b3354433ce188ada6d1e638d9cbb794f8c6e42de62c4d8cd1967f7e2d74",
    "env/env_systems/travel_env.py": "b189f76c4cd0db5984cce29a1ad19cba98ebdd033128679287106a054590797f",
    "env/env_systems/formal_reasoning_env/llm_backend.py": (
        "27c9898bd56becb732c2eed53036c3de289fe3cb0540ec73e53354abc8c64e67"
    ),
}
FORMAL_REFERENCE_SOURCE_FILES_SHA256 = {
    "agent/math.py": "33faff8e13f7c12d69d375f319fa151102a8268b8a52d1795304a325e2e85141",
    "env/env_systems/formal_reasoning_env/eval.py": (
        "fa8ac267b397561eccd552a0fe05db5334d604ba3da3fb56eebc6bcaa700cf30"
    ),
    "run_math.py": "6d652ac1218e2d296a80e3fdb3281106ed1a16187ed3d648636b45f6ce895b6a",
}
FORMAL_JUDGE_MODEL = "gpt-5-mini"
FORMAL_JUDGE_TEMPERATURE = 1.0
FORMAL_JUDGE_MAX_TOKENS = 4096
FORMAL_JUDGE_PROMPT_TEMPLATE_SHA256 = (
    "3bae603838f61deb76cec86feb7a1cd2c83df6f62921563d79dbf38f81b09cd9"
)


def _formal_system_prompt(label: str, *, contract_mode: str) -> str:
    if contract_mode == "failfast":
        outcome = (
            "A correct answer advances to the next question and earns +1; an "
            "incorrect answer ends the episode immediately."
        )
    elif contract_mode == "paper_eval":
        outcome = (
            "Every submitted answer is privately judged and then advances to the "
            "next question, whether correct or incorrect. A correct answer earns "
            "+1 and an incorrect answer earns 0. The final question's correctness "
            "determines task success."
        )
    else:  # pragma: no cover - immutable evaluator construction owns this.
        raise ValueError(f"unsupported Formal contract mode: {contract_mode!r}")
    return (
        f"You are operating the MemoryArena formal-reasoning {label} domain. "
        "An episode contains sequential questions from one paper. The current "
        "question and its published background are visible; answer text is "
        "privately evaluated by the original MemoryArena judge. "
        f"{outcome} Submit one final answer for the current question.\n\n"
        "Native domain action forms:\n- <final answer text>\n\n"
        "Policy memory action forms:\n"
        '- ADD {"key": "...", "value": "..."}\n'
        '- UPDATE {"memory_id": "mem_0000", "value": "..."}\n'
        '- DELETE {"memory_id": "mem_0000"}\n'
        '- RETRIEVE {"query": "...", "top_k": 3}\n'
        '- SUMMARY {"text": "...", "source_ids": ["S0", "C0"]}\n'
        '- FILTER {"keep_ids": ["C0"], "scope": "active"}\n\n'
        "Cross-phase memory lifecycle:\n"
        "- ADD writes policy-authored text to long-term memory for the current "
        "episode.\n"
        "- A native phase advance clears the current phase's short-term/page "
        "trace and active retrieved or summarized S*/C* context. Long-term "
        "memory is retained, but it is not automatically visible in the next "
        "phase.\n"
        "- RETRIEVE queries text previously written with ADD and exposes matching "
        "long-term memories in active context for the current phase.\n\n"
        "Reply in exactly this format:\n\nThought:\nbrief reasoning\n\nAction:\n"
        "<exactly one native domain action or uppercase memory action>"
    )


FORMAL_RUNTIME_CONTRACTS = {
    MATH_FAILFAST_SURFACE: {
        "contract_id": "memoryarena_formal_reasoning_math_failfast_v3_20260721",
        "contract_sha256": (
            "87b7ac62bd2595ad32c10f5586713b78ec91564f7f6e950fd65c567e7df22f6a"
        ),
        "system_prompt": _formal_system_prompt(
            "mathematics", contract_mode="failfast"
        ),
        "system_prompt_sha256": (
            "07575bd2d47ca1997871fbf32866a92485c054c3c63b5450536fe17cb873a812"
        ),
        "contract_mode": "failfast",
        "semantic_variant": "ordered_subtask_failfast_v1",
        "phase_transition": "advance_on_correct; terminal_on_incorrect",
        "episode_success": "all_questions_correct",
    },
    PHYS_FAILFAST_SURFACE: {
        "contract_id": "memoryarena_formal_reasoning_phys_failfast_v3_20260721",
        "contract_sha256": (
            "fee34fe351102ae1a3ee43dc26d92689a1246175086d5d515679331a6b4bef4a"
        ),
        "system_prompt": _formal_system_prompt("physics", contract_mode="failfast"),
        "system_prompt_sha256": (
            "a2cec3dcdca74764585b7c6f8c3343dd11cb1ce172d4368170abed7373c65eb0"
        ),
        "contract_mode": "failfast",
        "semantic_variant": "ordered_subtask_failfast_v1",
        "phase_transition": "advance_on_correct; terminal_on_incorrect",
        "episode_success": "all_questions_correct",
    },
    MATH_PAPER_EVAL_SURFACE: {
        "contract_id": (
            "memoryarena_formal_reasoning_math_"
            "paper_eval_one_action_v3_20260723"
        ),
        "contract_sha256": (
            "cbccf42639727276346ee524429e3826c59e8d9d62c63f207e2b15e127f7c305"
        ),
        "system_prompt": _formal_system_prompt(
            "mathematics", contract_mode="paper_eval"
        ),
        "system_prompt_sha256": (
            "6b6498b25081ae3c11f7ea8332508f9474c75432c6b1901028dbc44a2d776a85"
        ),
        "contract_mode": "paper_eval",
        "semantic_variant": "paper_metric_continue_on_incorrect_one_action_v1",
        "phase_transition": "advance_after_every_judged_answer",
        "episode_success": "final_question_correct",
    },
    PHYS_PAPER_EVAL_SURFACE: {
        "contract_id": (
            "memoryarena_formal_reasoning_phys_"
            "paper_eval_one_action_v3_20260723"
        ),
        "contract_sha256": (
            "339bf7f07ed94b1089682f80486396753953fc047df9b9e807be5bf6c2888120"
        ),
        "system_prompt": _formal_system_prompt(
            "physics", contract_mode="paper_eval"
        ),
        "system_prompt_sha256": (
            "f8caa9d251656b345097fe09c4ac51648db1abbc4d84be10ec00355ef9ddb620"
        ),
        "contract_mode": "paper_eval",
        "semantic_variant": "paper_metric_continue_on_incorrect_one_action_v1",
        "phase_transition": "advance_after_every_judged_answer",
        "episode_success": "final_question_correct",
    },
}
for _formal_runtime in FORMAL_RUNTIME_CONTRACTS.values():
    if hashlib.sha256(
        _formal_runtime["system_prompt"].encode("utf-8")
    ).hexdigest() != _formal_runtime["system_prompt_sha256"]:
        raise RuntimeError("Formal canonical system prompt hash drifted")
TRAVEL_PAPER_LEDGER_FIELDS = frozenset(
    {
        "metric_contract",
        "dataset_scope",
        "source_id",
        "complete",
        "full_pass_people",
        "total_people",
        "group_success",
        "group_constraint_rate",
        "constraint_people",
        "online_reward_is_separate",
    }
)
SEARCH_PAPER_METADATA_FIELDS = frozenset(
    {
        "id",
        "dataset_scope",
        "available",
        "metrics",
        "metric_scale",
        "paper_panel_complete",
        "public_task_count",
        "paper_task_count",
        "separate_from_online_reward",
    }
)
SEARCH_PAPER_LEDGER_FIELDS = frozenset(
    {
        "metric_contract",
        "dataset_scope",
        "query_id",
        "complete",
        "metric_scale",
        "phase_verdicts",
        "completed_phase_count",
        "process_score_numerator",
        "process_score_denominator",
        "process_score",
        "sr_at_k",
        "final_sr_numerator",
        "final_sr_denominator",
        "final_success",
        "online_reward_is_separate",
    }
)
SEARCH_PHASE_VERDICT_FIELDS = frozenset(
    {
        "phase_index",
        "phase_kind",
        "correct",
        "verdict_source",
        "answer_sha256",
        "judge_response_sha256",
        "judge_confidence",
        "judge_parse_error",
        "retrieved_docids",
    }
)
SEARCH_SR_AT_K_FIELDS = frozenset({"correct", "numerator", "denominator"})
FORMAL_PAPER_LEDGER_FIELDS = frozenset(
    {
        "metric_contract",
        "dataset_scope",
        "task_id",
        "paper_name",
        "complete",
        "phase_results",
        "completed_phase_count",
        "process_score_numerator",
        "process_score_denominator",
        "process_score",
        "final_sr_numerator",
        "final_sr_denominator",
        "final_success",
        "online_reward_is_separate",
    }
)

PAPER_SURFACE_REGISTRY = {
    WEBSHOP_V2_SURFACE: {
        "paper_column": "Shopping",
        "domain_id": None,
        "variant": "native_v1",
        "metric_mode": "episode_success",
        "canonical_macro_candidate": True,
    },
    TRAVEL_FAILFAST_SURFACE: {
        "paper_column": "Travel",
        "domain_id": "travel_planner",
        "variant": "failfast_one_action_v3",
        "metric_mode": "travel_failfast_diagnostic",
        "canonical_macro_candidate": False,
    },
    TRAVEL_PAPER_EVAL_SURFACE: {
        "paper_column": "Travel",
        "domain_id": "travel_planner",
        "variant": "paper_eval_one_action_v3",
        "metric_mode": "travel_paper_ledger",
        "canonical_macro_candidate": True,
    },
    SEARCH_FAILFAST_SURFACE: {
        "paper_column": "Search",
        "domain_id": "progressive_search",
        "variant": "public221_failfast_one_action_v3",
        "metric_mode": "search_failfast_diagnostic",
        "canonical_macro_candidate": False,
    },
    SEARCH_PAPER_EVAL_SURFACE: {
        "paper_column": "Search",
        "domain_id": "progressive_search",
        "variant": "public221_paper_eval_one_action_v3",
        "metric_mode": "search_paper_ledger",
        # The public release contains 221 of the paper's 256 Search tasks.
        # It is a valid public-panel metric, never a complete paper macro column.
        "canonical_macro_candidate": False,
    },
    MATH_FAILFAST_SURFACE: {
        "paper_column": "Math",
        "domain_id": "formal_reasoning_math",
        "variant": "failfast_v3",
        "metric_mode": "episode_success",
        "canonical_macro_candidate": False,
    },
    PHYS_FAILFAST_SURFACE: {
        "paper_column": "Physics",
        "domain_id": "formal_reasoning_phys",
        "variant": "failfast_v3",
        "metric_mode": "episode_success",
        "canonical_macro_candidate": False,
    },
    MATH_PAPER_EVAL_SURFACE: {
        "paper_column": "Math",
        "domain_id": "formal_reasoning_math",
        "variant": "paper_eval_one_action_v3",
        "metric_mode": "formal_paper_ledger",
        "canonical_macro_candidate": True,
    },
    PHYS_PAPER_EVAL_SURFACE: {
        "paper_column": "Physics",
        "domain_id": "formal_reasoning_phys",
        "variant": "paper_eval_one_action_v3",
        "metric_mode": "formal_paper_ledger",
        "canonical_macro_candidate": True,
    },
}

# Native WebShop v2 predates the v3 metadata contract and therefore does not
# expose ``system_prompt`` from its server. Load the canonical no-thinking
# prompt from the rollout schema so evaluation cannot drift from training when
# that contract changes. The lightweight evaluator still avoids importing the
# training stack: loading ``schemas.py`` only needs type-only dependencies,
# which are stubbed when they are unavailable in an eval-only environment.
def _load_legacy_webshop_system_prompt() -> str:
    schemas_path = Path(__file__).resolve().parents[2] / "verl/workers/rollout/schemas.py"
    if not schemas_path.is_file():
        raise RuntimeError(f"missing canonical AgentMemory prompt schema: {schemas_path}")

    sentinel = object()
    original_modules = {
        name: sys.modules.get(name, sentinel) for name in ("torch", "transformers")
    }
    try:
        if original_modules["torch"] is sentinel:
            import types

            torch_stub = types.ModuleType("torch")
            torch_stub.Tensor = object
            sys.modules["torch"] = torch_stub
        if original_modules["transformers"] is sentinel:
            import types

            transformers_stub = types.ModuleType("transformers")
            transformers_stub.PreTrainedTokenizer = object
            sys.modules["transformers"] = transformers_stub

        spec = importlib.util.spec_from_file_location(
            "agentmemory_eval_prompt_schema", schemas_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load canonical prompt schema: {schemas_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        prompt = module.AGENTMEMORY_ACTION_SYSTEM_PROMPT
    finally:
        for name, original in original_modules.items():
            if original is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

    if not isinstance(prompt, str) or not prompt:
        raise RuntimeError("canonical AgentMemory WebShop prompt is empty")
    return prompt


LEGACY_WEBSHOP_SYSTEM_PROMPT = _load_legacy_webshop_system_prompt()
LEGACY_WEBSHOP_MAX_POLICY_TURNS = 56


class EvalError(RuntimeError):
    """Base class for fail-closed evaluation errors."""


class HttpError(EvalError):
    def __init__(self, method: str, url: str, status: int, body: str):
        self.method = method
        self.url = url
        self.status = status
        self.body = body
        super().__init__(f"{method} {url} returned HTTP {status}: {body[:500]}")


class TokenizationError(EvalError):
    """The model server did not provide authoritative prompt token ids."""


def resolve_paper_surface(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve one exact runtime surface to its paper column and variant."""

    if not isinstance(metadata, Mapping):
        raise EvalError("environment metadata must be a mapping")
    surface = metadata.get("surface")
    if not isinstance(surface, str) or not surface:
        raise EvalError("environment metadata has no non-empty surface")
    try:
        registered = PAPER_SURFACE_REGISTRY[surface]
    except KeyError as exc:
        raise EvalError(f"unregistered AgentMemory paper surface: {surface}") from exc
    expected_domain = registered["domain_id"]
    if expected_domain is not None and metadata.get("domain_id") != expected_domain:
        raise EvalError(
            f"surface {surface} requires domain_id={expected_domain!r}, "
            f"observed {metadata.get('domain_id')!r}"
        )
    result = dict(registered)
    result["surface"] = surface
    return result


def compute_paper_macro5(success_rates: Mapping[str, Any]) -> float:
    """Compute the canonical five-surface paper macro, failing on drift."""

    if not isinstance(success_rates, Mapping):
        raise ValueError("paper success rates must be a mapping")
    required = set(PAPER_SUCCESS_COLUMNS)
    observed = set(success_rates)
    if observed != required:
        missing = sorted(required - observed)
        extra = sorted(observed - required)
        raise ValueError(
            "paper macro5 requires exactly Shopping/Travel/Search/Math/Physics: "
            f"missing={missing} extra={extra}"
        )
    values = []
    for column in PAPER_SUCCESS_COLUMNS:
        value = success_rates[column]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(
                f"paper success rate {column} must be finite in [0, 1]"
            )
        values.append(float(value))
    return math.fsum(values) / len(PAPER_SUCCESS_COLUMNS)


def compute_paper_macro5_from_manifests(
    manifests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate five complete, canonical surface manifests without guessing."""

    rates: dict[str, float] = {}
    surfaces: dict[str, str] = {}
    for index, manifest in enumerate(manifests):
        if not isinstance(manifest, Mapping):
            raise EvalError(f"paper manifest {index} must be a mapping")
        environment = manifest.get("environment")
        metadata = environment.get("metadata") if isinstance(environment, Mapping) else None
        registration = resolve_paper_surface(metadata)
        summary = manifest.get("summary")
        if not isinstance(summary, Mapping):
            raise EvalError(f"paper manifest {index} has no summary mapping")
        episodes = manifest.get("episodes")
        if not isinstance(episodes, list):
            raise EvalError(f"paper manifest {index} has no episode evidence list")
        recomputed = summarize_paper_surface(episodes, metadata)
        for key in (
            "paper_column",
            "paper_surface",
            "paper_variant",
            "paper_metric_mode",
            "paper_metric_contract",
            "paper_success_rate",
            "paper_metrics",
            "paper_macro_eligible",
            "paper_panel_complete",
            "expected_episode_count",
            "evaluated_episode_count",
            "panel_complete",
            "missing_episode_count",
            "missing_data_idx_preview",
        ):
            if summary.get(key) != recomputed.get(key):
                raise EvalError(
                    f"paper manifest {index} summary {key} disagrees with episodes"
                )
        column = registration["paper_column"]
        if summary.get("paper_column") != column:
            raise EvalError(
                f"paper manifest {index} summary column disagrees with surface"
            )
        if summary.get("panel_complete") is not True:
            raise EvalError(f"paper surface {column} does not cover its complete panel")
        if summary.get("paper_panel_complete") is not True:
            raise EvalError(
                f"paper surface {registration['surface']} lacks a complete paper panel"
            )
        if summary.get("paper_macro_eligible") is not True:
            raise EvalError(
                f"paper surface {registration['surface']} is not macro5 eligible"
            )
        if column in rates:
            raise EvalError(f"duplicate paper macro5 column: {column}")
        value = summary.get("paper_success_rate")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise EvalError(f"paper surface {column} has no valid success rate")
        rates[column] = float(value)
        surfaces[column] = registration["surface"]
    macro = compute_paper_macro5(rates)
    return {
        "metric_contract": PAPER_MACRO5_METRIC_CONTRACT,
        "columns": {column: rates[column] for column in PAPER_SUCCESS_COLUMNS},
        "surfaces": {
            column: surfaces[column] for column in PAPER_SUCCESS_COLUMNS
        },
        "macro5": macro,
    }


def _require_public_dataset(
    metadata: Mapping[str, Any],
    *,
    config: str,
    record_count: int,
    phase_count: int,
) -> Mapping[str, Any]:
    provenance = metadata.get("dataset_provenance")
    if not isinstance(provenance, Mapping):
        raise EvalError(f"{config} metadata lacks frozen dataset_provenance")
    try:
        frozen_spec = MEMORYARENA_FROZEN_DATASETS[config]
    except KeyError as exc:  # pragma: no cover - caller owns the supported set
        raise EvalError(f"no frozen dataset specification for {config}") from exc
    if (
        record_count != frozen_spec["record_count"]
        or phase_count != frozen_spec["phase_count"]
    ):
        raise EvalError(f"{config} evaluator constants disagree with frozen spec")
    expected = {
        "mode": "frozen_public_hf_dataset",
        "dataset_config": config,
        "split": "test",
        "repo_id": MEMORYARENA_HF_REPO,
        "revision": MEMORYARENA_HF_REVISION,
        "repo_path": frozen_spec["repo_path"],
        "sha256": frozen_spec["sha256"],
        "record_count": record_count,
        "phase_count": phase_count,
        "phase_field": frozen_spec["phase_field"],
    }
    attestation_payload = json.dumps(
        expected,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    expected["attestation_sha256"] = hashlib.sha256(attestation_payload).hexdigest()
    fields = set(provenance)
    expected_fields = set(expected)
    if fields != expected_fields:
        missing = sorted(expected_fields - fields)
        extra = sorted(str(key) for key in fields - expected_fields)
        raise EvalError(
            f"{config} dataset provenance fields mismatch: "
            f"missing={missing} extra={extra}"
        )
    mismatches = {
        key: (value, provenance.get(key))
        for key, value in expected.items()
        if provenance.get(key) != value
    }
    if mismatches:
        raise EvalError(
            f"{config} frozen dataset provenance mismatch: {mismatches}"
        )
    for key in ("record_count", "phase_count"):
        if isinstance(provenance.get(key), bool) or not isinstance(
            provenance.get(key), int
        ):
            raise EvalError(f"{config} dataset provenance {key} must be an integer")
    if metadata.get("dataset_sha256") != frozen_spec["sha256"]:
        raise EvalError(
            f"{config} top-level dataset_sha256 disagrees with frozen dataset"
        )
    metadata_task_count = metadata.get("task_count")
    if (
        isinstance(metadata_task_count, bool)
        or not isinstance(metadata_task_count, int)
        or metadata_task_count != record_count
    ):
        raise EvalError(
            f"{config} metadata task_count must be {record_count}, "
            f"observed {metadata.get('task_count')!r}"
        )
    return provenance


def _paper_evaluation_metadata(
    metadata: Mapping[str, Any],
    *,
    expected_contract: str,
) -> Mapping[str, Any]:
    paper = metadata.get("paper_evaluation")
    if not isinstance(paper, Mapping):
        raise EvalError("environment metadata lacks paper_evaluation contract")
    if paper.get("id") != expected_contract:
        raise EvalError(
            "paper_evaluation id mismatch: "
            f"expected {expected_contract!r}, observed {paper.get('id')!r}"
        )
    if paper.get("available") is not True:
        raise EvalError("paper_evaluation must be available in this runtime")
    dataset_scope = paper.get("dataset_scope")
    if not isinstance(dataset_scope, str) or not dataset_scope:
        raise EvalError("paper_evaluation metadata lacks dataset_scope")
    return paper


def _final_domain_evidence(episode: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    if episode.get("done") is not True or episode.get("timed_out") is not False:
        raise EvalError(f"episode {index} has no complete terminal paper evidence")
    steps = episode.get("steps")
    if not isinstance(steps, list) or not steps:
        raise EvalError(f"episode {index} has no terminal step evidence")
    final_step = steps[-1]
    after = final_step.get("env_info_after") if isinstance(final_step, Mapping) else None
    domain = after.get("domain_evidence") if isinstance(after, Mapping) else None
    if not isinstance(domain, Mapping):
        raise EvalError(f"episode {index} lacks terminal domain_evidence")
    return domain


def _terminal_env_info(episode: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    if episode.get("done") is not True or episode.get("timed_out") is not False:
        raise EvalError(f"episode {index} has no complete terminal evidence")
    steps = episode.get("steps")
    if not isinstance(steps, list) or not steps:
        raise EvalError(f"episode {index} has no terminal step evidence")
    final_step = steps[-1]
    after = final_step.get("env_info_after") if isinstance(final_step, Mapping) else None
    if not isinstance(after, Mapping):
        raise EvalError(f"episode {index} lacks terminal env_info_after")
    if after.get("sample_excluded") is not False:
        raise EvalError(f"episode {index} is excluded or lacks exclusion evidence")
    return after


def _require_shopping_dataset(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    provenance = metadata.get("dataset_provenance")
    if not isinstance(provenance, Mapping):
        raise EvalError("Shopping metadata lacks frozen dataset_provenance")
    fields = set(provenance)
    if fields != SHOPPING_DATASET_PROVENANCE_FIELDS:
        missing = sorted(SHOPPING_DATASET_PROVENANCE_FIELDS - fields)
        extra = sorted(str(key) for key in fields - SHOPPING_DATASET_PROVENANCE_FIELDS)
        raise EvalError(
            "Shopping dataset provenance fields mismatch: "
            f"missing={missing} extra={extra}"
        )
    raw_path = provenance.get("raw_dataset_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise EvalError("Shopping dataset provenance raw_dataset_path is invalid")
    expected = {
        "schema": "memoryarena_raw_dataset_provenance_v1",
        "raw_dataset_sha256": SHOPPING_RAW_DATASET_SHA256,
        "memoryarena_commit": SHOPPING_MEMORYARENA_COMMIT,
        "domain_data_sha256": SHOPPING_DOMAIN_DATA_SHA256,
        "action_surface_version": WEBSHOP_V2_SURFACE,
        "split_strategy": SHOPPING_SPLIT_STRATEGY,
        "split_manifest_sha256": SHOPPING_SPLIT_MANIFEST_SHA256,
        "bundle_count": SHOPPING_BUNDLE_COUNT,
        "sessions_per_bundle": SHOPPING_SESSIONS_PER_BUNDLE,
        "session_count": SHOPPING_SESSION_COUNT,
        "target_asin_membership_verified": True,
    }
    mismatches = {
        key: (value, provenance.get(key))
        for key, value in expected.items()
        if provenance.get(key) != value
    }
    if mismatches:
        raise EvalError(f"Shopping frozen dataset provenance mismatch: {mismatches}")
    for key in ("bundle_count", "sessions_per_bundle", "session_count"):
        if type(provenance.get(key)) is not int:
            raise EvalError(f"Shopping dataset provenance {key} must be an integer")
    if type(provenance.get("target_asin_membership_verified")) is not bool:
        raise EvalError(
            "Shopping target_asin_membership_verified must be a boolean"
        )
    split_counts = provenance.get("split_counts")
    expected_split_counts = {"train": 120, "dev": 15, "test": 15}
    if not isinstance(split_counts, Mapping) or dict(split_counts) != expected_split_counts:
        raise EvalError("Shopping frozen split counts mismatch")
    if any(type(value) is not int for value in split_counts.values()):
        raise EvalError("Shopping split counts must be integers")

    top_level_expected = {
        "task_count": SHOPPING_BUNDLE_COUNT,
        "dataset_sha256": SHOPPING_RAW_DATASET_SHA256,
        "raw_dataset_sha256": SHOPPING_RAW_DATASET_SHA256,
        "split_manifest_sha256": SHOPPING_SPLIT_MANIFEST_SHA256,
        "memoryarena_commit": SHOPPING_MEMORYARENA_COMMIT,
        "domain_data_sha256": SHOPPING_DOMAIN_DATA_SHA256,
        "annotation_gate_allowed_task_count": SHOPPING_BUNDLE_COUNT,
        "annotation_gate_allowed_task_ids_sha256": SHOPPING_TASK_IDS_SHA256,
    }
    top_level_mismatches = {
        key: (value, metadata.get(key))
        for key, value in top_level_expected.items()
        if metadata.get(key) != value
    }
    if top_level_mismatches:
        raise EvalError(
            "Shopping top-level dataset provenance mismatch: "
            f"{top_level_mismatches}"
        )
    for key in ("task_count", "annotation_gate_allowed_task_count"):
        if type(metadata.get(key)) is not int:
            raise EvalError(f"Shopping metadata {key} must be an integer")
    if metadata.get("splits") != ["dev", "test", "train"]:
        raise EvalError("Shopping paper panel requires all frozen dataset splits")
    return provenance


def _shopping_task_id(data_idx: int) -> str:
    category_index, category_task_index = divmod(data_idx, 30)
    return f"{SHOPPING_TASK_CATEGORIES[category_index]}_item_{category_task_index}"


def aggregate_shopping_panel_evidence(
    episodes: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate native Shopping task identities and six-session ledgers."""

    _require_shopping_dataset(metadata)
    task_ids: set[str] = set()
    phase_total = 0
    data_indices: set[int] = set()
    for episode_index, episode in enumerate(episodes):
        data_idx = episode.get("data_idx")
        if (
            isinstance(data_idx, bool)
            or not isinstance(data_idx, int)
            or not 0 <= data_idx < SHOPPING_BUNDLE_COUNT
        ):
            raise EvalError(f"Shopping episode {episode_index} has invalid data_idx")
        if data_idx in data_indices:
            raise EvalError(f"Shopping paper evaluation repeats data_idx {data_idx}")
        data_indices.add(data_idx)
        expected_task_id = _shopping_task_id(data_idx)
        initial = episode.get("initial_env_info")
        if not isinstance(initial, Mapping):
            raise EvalError(f"Shopping episode {episode_index} lacks reset evidence")
        after = _terminal_env_info(episode, episode_index)
        for label, info in (("initial", initial), ("terminal", after)):
            if info.get("task_id") != expected_task_id:
                raise EvalError(
                    f"Shopping episode {episode_index} {label} task_id mismatch"
                )
            if (
                type(info.get("phase_count")) is not int
                or info.get("phase_count") != SHOPPING_SESSIONS_PER_BUNDLE
                or type(info.get("subtask_count")) is not int
                or info.get("subtask_count") != SHOPPING_SESSIONS_PER_BUNDLE
            ):
                raise EvalError(
                    f"Shopping episode {episode_index} {label} phase ledger mismatch"
                )
        if (
            type(initial.get("current_subtask_index")) is not int
            or initial.get("current_subtask_index") != 0
        ):
            raise EvalError(
                f"Shopping episode {episode_index} did not start at session zero"
            )
        final_index = after.get("current_subtask_index")
        if (
            isinstance(final_index, bool)
            or not isinstance(final_index, int)
            or not 0 <= final_index <= SHOPPING_SESSIONS_PER_BUNDLE
        ):
            raise EvalError(
                f"Shopping episode {episode_index} has invalid terminal session index"
            )
        progress = episode.get("final_phase_progress")
        if (
            not isinstance(progress, Mapping)
            or progress.get("phase_index_after") != final_index
            or type(progress.get("phase_index_after")) is not int
            or progress.get("phase_count") != SHOPPING_SESSIONS_PER_BUNDLE
            or type(progress.get("phase_count")) is not int
        ):
            raise EvalError(
                f"Shopping episode {episode_index} terminal progress conflicts"
            )
        success = episode.get("episode_success")
        if type(success) is not bool or after.get("episode_success") is not success:
            raise EvalError(
                f"Shopping episode {episode_index} terminal success conflicts"
            )
        purchases = after.get("purchase_history")
        if not isinstance(purchases, list) or not purchases:
            raise EvalError(
                f"Shopping episode {episode_index} lacks purchase phase ledger"
            )
        expected_purchase_count = (
            SHOPPING_SESSIONS_PER_BUNDLE if success else final_index + 1
        )
        if len(purchases) != expected_purchase_count:
            raise EvalError(
                f"Shopping episode {episode_index} purchase ledger length conflicts"
            )
        for session_index, event in enumerate(purchases):
            if not isinstance(event, Mapping):
                raise EvalError(
                    f"Shopping episode {episode_index} purchase event is invalid"
                )
            correct = event.get("purchase_correct")
            expected_correct = success or session_index < final_index
            if (
                event.get("op") != "BUY"
                or event.get("committed") is not True
                or type(correct) is not bool
                or correct is not expected_correct
                or event.get("session_advanced") is not correct
                or type(event.get("session_advanced")) is not bool
                or event.get("session_index") != session_index
                or type(event.get("session_index")) is not int
            ):
                raise EvalError(
                    f"Shopping episode {episode_index} purchase event {session_index} "
                    "conflicts with phase progress"
                )
        if success and final_index != SHOPPING_SESSIONS_PER_BUNDLE:
            raise EvalError(
                f"Shopping episode {episode_index} success lacks all six sessions"
            )
        if expected_task_id in task_ids:
            raise EvalError(f"Shopping paper evaluation repeats task_id {expected_task_id}")
        task_ids.add(expected_task_id)
        phase_total += SHOPPING_SESSIONS_PER_BUNDLE

    complete = (
        data_indices == set(range(SHOPPING_BUNDLE_COUNT))
        and len(task_ids) == SHOPPING_BUNDLE_COUNT
        and phase_total == SHOPPING_SESSION_COUNT
    )
    return {
        "metric_contract": "episode_success",
        "dataset_scope": "memoryarena_bundled_shopping_frozen150",
        "task_count": len(data_indices),
        "unique_task_id_count": len(task_ids),
        "phase_count": phase_total,
        "expected_task_count": SHOPPING_BUNDLE_COUNT,
        "expected_phase_count": SHOPPING_SESSION_COUNT,
        "server_provenance_verified": True,
        "paper_panel_complete": complete,
    }


def _require_formal_runtime_metadata(
    metadata: Mapping[str, Any],
    *,
    surface: str,
    dataset_config: str,
    record_count: int,
    phase_count: int,
) -> None:
    try:
        runtime = FORMAL_RUNTIME_CONTRACTS[surface]
    except KeyError as exc:  # pragma: no cover - caller validates the surface.
        raise EvalError(f"unsupported Formal runtime surface: {surface!r}") from exc
    _require_public_dataset(
        metadata,
        config=dataset_config,
        record_count=record_count,
        phase_count=phase_count,
    )
    exact_top_level = {
        "formal_schema_version": FORMAL_SCHEMA_V3,
        "source": "MemoryArena",
        "surface": surface,
        "domain_id": dataset_config,
        "dataset_config": dataset_config,
        "dataset_sha256": MEMORYARENA_FROZEN_DATASETS[dataset_config]["sha256"],
        "task_count": record_count,
        "phase_count": phase_count,
        "contract_id": runtime["contract_id"],
        "contract_sha256": runtime["contract_sha256"],
        "system_prompt": runtime["system_prompt"],
        "system_prompt_sha256": runtime["system_prompt_sha256"],
        "native_action_descriptions": ["<final answer text>"],
        "max_steps": 64,
        "judge": "memoryarena_llm_math_equivalence_v1",
        "contract_mode": runtime["contract_mode"],
        "semantic_variant": runtime["semantic_variant"],
        "phase_transition": runtime["phase_transition"],
        "episode_success": runtime["episode_success"],
        "reward_overlay": "none",
    }
    mismatches = {
        key: (expected, metadata.get(key))
        for key, expected in exact_top_level.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise EvalError(f"Formal runtime contract provenance mismatch: {mismatches}")
    _require_exact_mapping(
        "Formal memory_reward_policy",
        metadata.get("memory_reward_policy"),
        {
            "first_add": 0.0,
            "first_later_phase_retrieve": 0.0,
            "exact_repeat": 0.0,
            "invalid_action": 0.0,
        },
    )
    if runtime["contract_mode"] == "paper_eval":
        _require_exact_mapping(
            "Formal paper_evaluation",
            metadata.get("paper_evaluation"),
            {
                "id": FORMAL_PAPER_METRIC_CONTRACT,
                "dataset_scope": FORMAL_PAPER_DATASET_SCOPES[surface],
                "available": True,
                "metrics": ["PS", "SR"],
                "metric_scale": "unit_interval",
                "canonical_semantics": True,
                "paper_panel_complete": True,
                "paper_column_eligible": True,
                "continue_after_incorrect": True,
                "separate_from_online_reward": True,
            },
        )
    elif "paper_evaluation" in metadata:
        raise EvalError("Formal fail-fast surface must not claim paper evaluation")
    expected_upstream = {
        "mode": "pinned_pristine_upstream_scopes",
        "memoryarena_commit": FORMAL_MEMORYARENA_COMMIT,
        "pristine_git_scopes": ["env", "agent/math.py", "run_math.py"],
        "env_git_tree_oid": FORMAL_ENV_GIT_TREE_OID,
        "runtime_import_entry_files_sha256": dict(
            FORMAL_RUNTIME_SOURCE_FILES_SHA256
        ),
        "reference_entrypoint_files_sha256": dict(
            FORMAL_REFERENCE_SOURCE_FILES_SHA256
        ),
        "selected_files_bundle_sha256": FORMAL_SELECTED_SOURCE_BUNDLE_SHA256,
    }
    _require_exact_mapping(
        "Formal upstream_provenance",
        metadata.get("upstream_provenance"),
        expected_upstream,
    )
    judge = metadata.get("judge_provenance")
    if not isinstance(judge, Mapping) or set(judge) != {
        "mode",
        "backend",
        "model",
        "temperature",
        "max_tokens",
        "endpoint_sha256",
        "prompt_template_sha256",
        "config_sha256",
    }:
        raise EvalError("Formal judge_provenance fields mismatch")
    judge_config = {
        "mode": "upstream_memoryarena_judge",
        "backend": "openai",
        "model": FORMAL_JUDGE_MODEL,
        "temperature": FORMAL_JUDGE_TEMPERATURE,
        "max_tokens": FORMAL_JUDGE_MAX_TOKENS,
        "endpoint_sha256": judge.get("endpoint_sha256"),
        "prompt_template_sha256": FORMAL_JUDGE_PROMPT_TEMPLATE_SHA256,
    }
    if (
        not _valid_sha256_or_none(judge.get("endpoint_sha256"))
        or judge.get("endpoint_sha256") is None
        or any(judge.get(key) != value for key, value in judge_config.items())
        or judge.get("config_sha256") != _canonical_json_sha256(judge_config)
    ):
        raise EvalError("Formal judge provenance mismatch")


def _validate_formal_info_identity(
    info: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any],
    expected_phase_count: int,
    boundary: str,
) -> None:
    expected = {
        "formal_schema_version": FORMAL_SCHEMA_V3,
        "domain_id": metadata["domain_id"],
        "surface": metadata["surface"],
        "contract_id": metadata["contract_id"],
        "contract_sha256": metadata["contract_sha256"],
        "phase_count": expected_phase_count,
    }
    mismatches = {
        key: (value, info.get(key))
        for key, value in expected.items()
        if info.get(key) != value
    }
    if "phase_count" in mismatches:
        raise EvalError(f"{boundary} Formal phase_count mismatch")
    if mismatches:
        raise EvalError(f"{boundary} Formal runtime identity mismatch: {mismatches}")
    if type(info.get("phase_index")) is not int or not (
        0 <= info["phase_index"] <= expected_phase_count
    ):
        raise EvalError(f"{boundary} Formal phase_index is invalid")
    if type(info.get("episode_success")) is not bool:
        raise EvalError(f"{boundary} Formal episode_success is not boolean")
    if info.get("sample_excluded") is not False:
        raise EvalError(f"{boundary} Formal sample is excluded or unlabelled")


def _validate_formal_episode_steps(
    episode: Mapping[str, Any],
    *,
    episode_index: int,
    metadata: Mapping[str, Any],
    expected_task_id: str,
    expected_paper_name: str,
    expected_phase_count: int,
) -> tuple[list[bool], Mapping[str, Any]]:
    contract_mode = metadata.get("contract_mode")
    if contract_mode not in {"failfast", "paper_eval"}:
        raise EvalError(f"unsupported Formal contract mode: {contract_mode!r}")
    initial = episode["initial_env_info"]
    assert isinstance(initial, Mapping)
    _validate_formal_info_identity(
        initial,
        metadata=metadata,
        expected_phase_count=expected_phase_count,
        boundary=f"Formal episode {episode_index} reset",
    )
    steps = episode.get("steps")
    if not isinstance(steps, list) or not steps:
        raise EvalError(f"Formal episode {episode_index} has no step ledger")
    previous: Mapping[str, Any] = initial
    answer_results: list[bool] = []
    reward_total = 0.0
    for position, step in enumerate(steps, start=1):
        if not isinstance(step, Mapping):
            raise EvalError(f"Formal episode {episode_index} step {position} is invalid")
        before = step.get("env_info_before")
        after = step.get("env_info_after")
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            raise EvalError(
                f"Formal episode {episode_index} step {position} lacks env evidence"
            )
        if dict(before) != dict(previous):
            raise EvalError(
                f"Formal episode {episode_index} step {position} breaks state continuity"
            )
        for label, info in (("before", before), ("after", after)):
            _validate_formal_info_identity(
                info,
                metadata=metadata,
                expected_phase_count=expected_phase_count,
                boundary=f"Formal episode {episode_index} step {position} {label}",
            )
            evidence = info.get("domain_evidence")
            if not isinstance(evidence, Mapping):
                raise EvalError(
                    f"Formal episode {episode_index} step {position} {label} "
                    "lacks domain evidence"
                )
            if evidence.get("task_id") != expected_task_id:
                raise EvalError(
                    f"Formal episode {episode_index} step {position} {label} "
                    "task_id mismatch"
                )
            if evidence.get("paper_name") != expected_paper_name:
                raise EvalError(
                    f"Formal episode {episode_index} step {position} {label} "
                    "paper identity mismatch"
                )
        before_index = before["phase_index"]
        after_index = after["phase_index"]
        if after_index not in {before_index, before_index + 1}:
            raise EvalError(
                f"Formal episode {episode_index} step {position} phase jump is invalid"
            )
        if step.get("turn") != position:
            raise EvalError(
                f"Formal episode {episode_index} step {position} turn is non-contiguous"
            )
        if type(step.get("done")) is not bool:
            raise EvalError(
                f"Formal episode {episode_index} step {position} done is not boolean"
            )
        if step["done"] and position != len(steps):
            raise EvalError(
                f"Formal episode {episode_index} continues after a terminal step"
            )
        reward = step.get("reward")
        if (
            isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not math.isfinite(float(reward))
        ):
            raise EvalError(
                f"Formal episode {episode_index} step {position} reward is invalid"
            )
        reward = float(reward)
        components = after.get("reward_components")
        if not isinstance(components, list) or not components:
            raise EvalError(
                f"Formal episode {episode_index} step {position} lacks reward ledger"
            )
        component_total = 0.0
        for component in components:
            if not isinstance(component, Mapping) or component.get("step") != position:
                raise EvalError(
                    f"Formal episode {episode_index} step {position} has invalid "
                    "reward component"
                )
            value = component.get("value")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise EvalError(
                    f"Formal episode {episode_index} step {position} has non-finite "
                    "reward component"
                )
            component_total += float(value)
        if not math.isclose(component_total, reward, rel_tol=0.0, abs_tol=1e-8):
            raise EvalError(
                f"Formal episode {episode_index} step {position} reward ledger mismatch"
            )
        execution = after.get("action_execution")
        if not isinstance(execution, Mapping) or execution.get("step") != position:
            raise EvalError(
                f"Formal episode {episode_index} step {position} lacks action execution"
            )
        op = str(execution.get("op", "")).upper()
        status = str(execution.get("status", "")).lower()
        if op == "ANSWER":
            if status == "committed_correct":
                passed = True
                expected_reward = 1.0
                expected_component = "formal_reasoning_answer_correct"
                if after_index != before_index + 1:
                    raise EvalError(
                        f"Formal episode {episode_index} correct answer did not advance"
                    )
                expected_done = after_index == expected_phase_count
                if step["done"] is not expected_done:
                    raise EvalError(
                        f"Formal episode {episode_index} correct answer has invalid "
                        f"{contract_mode} termination"
                    )
            elif status == "committed_incorrect":
                passed = False
                expected_reward = 0.0
                expected_component = "formal_reasoning_answer_incorrect"
                if contract_mode == "failfast":
                    if after_index != before_index or step["done"] is not True:
                        raise EvalError(
                            f"Formal episode {episode_index} incorrect answer did not "
                            "fail fast"
                        )
                else:
                    expected_after = before_index + 1
                    expected_done = expected_after == expected_phase_count
                    if (
                        after_index != expected_after
                        or step["done"] is not expected_done
                    ):
                        raise EvalError(
                            f"Formal episode {episode_index} incorrect answer violates "
                            "paper-eval continuation"
                        )
            else:
                raise EvalError(
                    f"Formal episode {episode_index} has unsupported ANSWER status"
                )
            if (
                not math.isclose(reward, expected_reward, rel_tol=0.0, abs_tol=1e-8)
                or len(components) != 1
                or components[0].get("name") != expected_component
                or str(components[0].get("op", "")).upper() != "ANSWER"
            ):
                raise EvalError(
                    f"Formal episode {episode_index} ANSWER reward semantics mismatch"
                )
            answer_results.append(passed)
            evidence = after["domain_evidence"]
            for digest_key in (
                "answer_sha256",
                "ground_truth_sha256",
                "judge_output_sha256",
            ):
                if (
                    evidence.get(digest_key) is None
                    or not _valid_sha256_or_none(evidence.get(digest_key))
                ):
                    raise EvalError(
                        f"Formal episode {episode_index} ANSWER lacks {digest_key}"
                    )
        elif after_index != before_index or step["done"]:
            raise EvalError(
                f"Formal episode {episode_index} non-ANSWER action changed outcome"
            )
        evidence = after["domain_evidence"]
        if "phase_results" in evidence:
            if evidence.get("phase_results") != answer_results:
                raise EvalError(
                    f"Formal episode {episode_index} cumulative phase ledger mismatch"
                )
            if evidence.get("correct_count") != sum(answer_results):
                raise EvalError(
                    f"Formal episode {episode_index} cumulative correct_count mismatch"
                )
        reward_total += reward
        previous = after
    if episode.get("done") is not True or episode.get("timed_out") is not False:
        raise EvalError(f"Formal episode {episode_index} lacks a judged terminal outcome")
    if steps[-1].get("done") is not True:
        raise EvalError(f"Formal episode {episode_index} final step is not terminal")
    episode_return = episode.get("episode_return")
    if (
        isinstance(episode_return, bool)
        or not isinstance(episode_return, (int, float))
        or not math.isclose(
            float(episode_return), reward_total, rel_tol=0.0, abs_tol=1e-8
        )
    ):
        raise EvalError(f"Formal episode {episode_index} return ledger mismatch")
    return answer_results, previous


def _validate_formal_paper_ledger(
    ledger: Any,
    *,
    episode_index: int,
    dataset_scope: str,
    task_id: str,
    paper_name: str,
    phase_results: Sequence[bool],
) -> tuple[float, bool]:
    if not isinstance(ledger, Mapping):
        raise EvalError(f"Formal episode {episode_index} lacks paper_evaluation ledger")
    if set(ledger) != FORMAL_PAPER_LEDGER_FIELDS:
        raise EvalError(
            f"Formal episode {episode_index} paper_evaluation fields mismatch"
        )

    phase_count = len(phase_results)
    correct_count = sum(phase_results)
    final_success = phase_results[-1]
    exact_values = {
        "metric_contract": FORMAL_PAPER_METRIC_CONTRACT,
        "dataset_scope": dataset_scope,
        "task_id": task_id,
        "paper_name": paper_name,
        "complete": True,
        "phase_results": list(phase_results),
        "completed_phase_count": phase_count,
        "process_score_numerator": correct_count,
        "process_score_denominator": phase_count,
        "final_sr_numerator": int(final_success),
        "final_sr_denominator": 1,
        "final_success": final_success,
        "online_reward_is_separate": True,
    }
    mismatches = {
        key: (expected, ledger.get(key))
        for key, expected in exact_values.items()
        if ledger.get(key) != expected
    }
    integer_fields = (
        "completed_phase_count",
        "process_score_numerator",
        "process_score_denominator",
        "final_sr_numerator",
        "final_sr_denominator",
    )
    if any(type(ledger.get(key)) is not int for key in integer_fields):
        raise EvalError(
            f"Formal episode {episode_index} paper_evaluation integer fields mismatch"
        )
    boolean_fields = (
        "complete",
        "final_success",
        "online_reward_is_separate",
    )
    if any(type(ledger.get(key)) is not bool for key in boolean_fields):
        raise EvalError(
            f"Formal episode {episode_index} paper_evaluation boolean fields mismatch"
        )
    ledger_phase_results = ledger.get("phase_results")
    if not isinstance(ledger_phase_results, list) or any(
        type(value) is not bool for value in ledger_phase_results
    ):
        raise EvalError(
            f"Formal episode {episode_index} paper_evaluation phase results mismatch"
        )
    if mismatches:
        raise EvalError(
            f"Formal episode {episode_index} paper_evaluation ledger mismatch: "
            f"{mismatches}"
        )

    expected_process_score = correct_count / phase_count
    process_score = ledger.get("process_score")
    if (
        isinstance(process_score, bool)
        or not isinstance(process_score, (int, float))
        or not math.isfinite(float(process_score))
        or not math.isclose(
            float(process_score),
            expected_process_score,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise EvalError(
            f"Formal episode {episode_index} paper_evaluation process score mismatch"
        )
    return expected_process_score, final_success


def aggregate_formal_panel_evidence(
    episodes: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate frozen Formal task ids and their per-paper phase ledgers."""

    surface = metadata.get("surface")
    try:
        dataset_config = FORMAL_SURFACE_DATASETS[surface]
        phase_counts = FORMAL_TASK_PHASE_COUNTS[surface]
    except KeyError as exc:
        raise EvalError(f"unsupported Formal surface: {surface!r}") from exc
    expected_phase_total = sum(phase_counts)
    _require_formal_runtime_metadata(
        metadata,
        surface=surface,
        dataset_config=dataset_config,
        record_count=len(phase_counts),
        phase_count=expected_phase_total,
    )
    contract_mode = metadata["contract_mode"]
    if not episodes:
        raise EvalError("Formal panel requires at least one episode")

    data_indices: set[int] = set()
    task_ids: set[str] = set()
    observed_phase_total = 0
    process_scores: list[float] = []
    final_successes = 0
    for episode_index, episode in enumerate(episodes):
        data_idx = episode.get("data_idx")
        if (
            isinstance(data_idx, bool)
            or not isinstance(data_idx, int)
            or not 0 <= data_idx < len(phase_counts)
        ):
            raise EvalError(f"Formal episode {episode_index} has invalid data_idx")
        if data_idx in data_indices:
            raise EvalError(f"Formal paper evaluation repeats data_idx {data_idx}")
        data_indices.add(data_idx)
        expected_task_id = str(data_idx)
        expected_phase_count = phase_counts[data_idx]
        initial = episode.get("initial_env_info")
        if not isinstance(initial, Mapping):
            raise EvalError(f"Formal episode {episode_index} lacks reset evidence")
        initial_domain = initial.get("domain_evidence")
        if not isinstance(initial_domain, Mapping):
            raise EvalError(f"Formal episode {episode_index} lacks reset task identity")
        if initial_domain.get("task_id") != expected_task_id:
            raise EvalError(f"Formal episode {episode_index} initial task_id mismatch")
        if (
            type(initial.get("phase_count")) is not int
            or initial.get("phase_count") != expected_phase_count
        ):
            raise EvalError(f"Formal episode {episode_index} initial phase_count mismatch")
        if type(initial.get("phase_index")) is not int or initial.get("phase_index") != 0:
            raise EvalError(f"Formal episode {episode_index} did not start at phase zero")
        if (
            not isinstance(initial_domain.get("paper_name"), str)
            or not initial_domain.get("paper_name").strip()
        ):
            raise EvalError(f"Formal episode {episode_index} paper identity mismatch")
        phase_results, after = _validate_formal_episode_steps(
            episode,
            episode_index=episode_index,
            metadata=metadata,
            expected_task_id=expected_task_id,
            expected_paper_name=initial_domain["paper_name"],
            expected_phase_count=expected_phase_count,
        )
        domain = after["domain_evidence"]
        progress = episode.get("final_phase_progress")
        final_index = progress.get("phase_index_after") if isinstance(progress, Mapping) else None
        if (
            not isinstance(progress, Mapping)
            or isinstance(final_index, bool)
            or not isinstance(final_index, int)
            or progress.get("phase_count") != expected_phase_count
            or type(progress.get("phase_count")) is not int
            or after.get("phase_index") != final_index
            or type(after.get("phase_index")) is not int
        ):
            raise EvalError(
                f"Formal episode {episode_index} terminal phase progress conflicts"
            )
        correct_count = domain.get("correct_count")
        if type(correct_count) is not int or correct_count != sum(phase_results):
            raise EvalError(f"Formal episode {episode_index} correct_count conflicts")
        success = episode.get("episode_success")
        if type(success) is not bool or after.get("episode_success") is not success:
            raise EvalError(f"Formal episode {episode_index} terminal success conflicts")
        if contract_mode == "failfast":
            if success:
                valid_ledger = (
                    final_index == expected_phase_count
                    and len(phase_results) == expected_phase_count
                    and all(phase_results)
                )
            else:
                valid_ledger = (
                    0 <= final_index < expected_phase_count
                    and len(phase_results) == final_index + 1
                    and all(phase_results[:-1])
                    and phase_results[-1] is False
                )
        else:
            valid_ledger = (
                final_index == expected_phase_count
                and len(phase_results) == expected_phase_count
                and success is phase_results[-1]
            )
        if not valid_ledger:
            raise EvalError(
                f"Formal episode {episode_index} phase results conflict with progress"
            )
        if contract_mode == "paper_eval":
            process_score, final_success = _validate_formal_paper_ledger(
                domain.get("paper_evaluation"),
                episode_index=episode_index,
                dataset_scope=FORMAL_PAPER_DATASET_SCOPES[surface],
                task_id=expected_task_id,
                paper_name=initial_domain["paper_name"],
                phase_results=phase_results,
            )
            process_scores.append(process_score)
            final_successes += int(final_success)
        if expected_task_id in task_ids:
            raise EvalError(f"Formal paper evaluation repeats task_id {expected_task_id}")
        task_ids.add(expected_task_id)
        observed_phase_total += expected_phase_count

    complete = (
        data_indices == set(range(len(phase_counts)))
        and len(task_ids) == len(phase_counts)
        and observed_phase_total == expected_phase_total
    )
    dataset_scope = f"memoryarena_{dataset_config}_frozen{len(phase_counts)}"
    common = {
        "dataset_scope": dataset_scope,
        "task_count": len(data_indices),
        "unique_task_id_count": len(task_ids),
        "phase_count": observed_phase_total,
        "expected_task_count": len(phase_counts),
        "expected_phase_count": expected_phase_total,
        "server_provenance_verified": True,
        "paper_panel_complete": complete,
    }
    if contract_mode == "failfast":
        return {"metric_contract": "episode_success", **common}

    process_score_sum = math.fsum(process_scores)
    task_count = len(process_scores)
    return {
        "metric_contract": FORMAL_PAPER_METRIC_CONTRACT,
        **common,
        "process_score_numerator": process_score_sum,
        "process_score_denominator": task_count,
        "process_score": process_score_sum / task_count,
        "final_sr_numerator": final_successes,
        "final_sr_denominator": task_count,
        "final_success_rate": final_successes / task_count,
        "online_reward_is_separate": True,
    }


def _panel_coverage(
    episodes: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    *,
    expected_task_count: int | None = None,
) -> dict[str, Any]:
    task_count = (
        metadata.get("task_count")
        if expected_task_count is None
        else expected_task_count
    )
    if isinstance(task_count, bool) or not isinstance(task_count, int) or task_count < 1:
        raise EvalError("environment metadata task_count must be a positive integer")
    if expected_task_count is not None and metadata.get("task_count") != task_count:
        raise EvalError(
            f"environment metadata task_count must be {task_count}, "
            f"observed {metadata.get('task_count')!r}"
        )
    indices = []
    for index, episode in enumerate(episodes):
        value = episode.get("data_idx")
        if isinstance(value, bool) or not isinstance(value, int):
            raise EvalError(f"episode {index} has invalid data_idx")
        indices.append(value)
    if len(indices) != len(set(indices)):
        raise EvalError("paper evaluation contains duplicate data_idx episodes")
    expected = set(range(task_count))
    observed = set(indices)
    extra = sorted(observed - expected)
    if extra:
        raise EvalError(f"paper evaluation has out-of-panel data_idx values: {extra}")
    missing = sorted(expected - observed)
    return {
        "expected_episode_count": task_count,
        "evaluated_episode_count": len(indices),
        "panel_complete": not missing and len(indices) == task_count,
        "missing_episode_count": len(missing),
        "missing_data_idx_preview": missing[:20],
    }


def aggregate_travel_paper_metrics(
    episodes: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate official Travel PS/SPS/SR contribution ledgers."""

    if metadata.get("surface") != TRAVEL_PAPER_EVAL_SURFACE:
        raise EvalError("Travel paper metrics require the paper-eval runtime surface")
    if metadata.get("domain_id") != "travel_planner":
        raise EvalError("Travel paper surface requires domain_id='travel_planner'")
    _require_public_dataset(
        metadata,
        config="group_travel_planner",
        record_count=TRAVEL_RECORD_COUNT,
        phase_count=TRAVEL_PHASE_COUNT,
    )
    if metadata.get("contract_mode") != "paper_eval":
        raise EvalError("Travel paper surface requires contract_mode='paper_eval'")
    if (
        type(metadata.get("phase_count")) is not int
        or metadata.get("phase_count") != TRAVEL_PHASE_COUNT
    ):
        raise EvalError(
            f"Travel metadata phase_count must be {TRAVEL_PHASE_COUNT}"
        )
    contract = TRAVEL_PAPER_METRIC_CONTRACT
    paper_metadata = _paper_evaluation_metadata(
        metadata,
        expected_contract=contract,
    )
    expected_paper_flags = {
        "canonical_semantics": True,
        "paper_panel_complete": True,
        "paper_column_eligible": True,
        "separate_from_online_reward": True,
    }
    for key, expected in expected_paper_flags.items():
        if type(paper_metadata.get(key)) is not bool:
            raise EvalError(f"Travel paper_evaluation {key} must be boolean")
        if paper_metadata.get(key) is not expected:
            raise EvalError(
                f"Travel paper_evaluation {key} must be {expected!r}"
            )
    dataset_scope = paper_metadata["dataset_scope"]
    if dataset_scope != TRAVEL_PAPER_DATASET_SCOPE:
        raise EvalError(
            "Travel paper_evaluation dataset_scope mismatch: "
            f"expected {TRAVEL_PAPER_DATASET_SCOPE!r}, observed {dataset_scope!r}"
        )
    full_pass_people = 0
    total_people = 0
    total_constraint_people = 0
    successful_groups = 0
    constraint_group_rates: list[float] = []
    source_ids: set[Any] = set()
    for index, episode in enumerate(episodes):
        domain = _final_domain_evidence(episode, index)
        ledger = domain.get("paper_evaluation")
        if not isinstance(ledger, Mapping):
            raise EvalError(f"Travel episode {index} lacks paper_evaluation ledger")
        ledger_fields = set(ledger)
        if ledger_fields != TRAVEL_PAPER_LEDGER_FIELDS:
            missing = sorted(TRAVEL_PAPER_LEDGER_FIELDS - ledger_fields)
            extra = sorted(str(key) for key in ledger_fields - TRAVEL_PAPER_LEDGER_FIELDS)
            raise EvalError(
                f"Travel episode {index} paper ledger fields mismatch: "
                f"missing={missing} extra={extra}"
            )
        expected_pairs = {
            "metric_contract": contract,
            "dataset_scope": dataset_scope,
            "complete": True,
            "online_reward_is_separate": True,
        }
        for key, expected in expected_pairs.items():
            if ledger.get(key) != expected:
                raise EvalError(
                    f"Travel episode {index} paper ledger {key} mismatch: "
                    f"expected {expected!r}, observed {ledger.get(key)!r}"
                )
        if type(ledger.get("complete")) is not bool or type(
            ledger.get("online_reward_is_separate")
        ) is not bool:
            raise EvalError(
                f"Travel episode {index} paper completion flags must be boolean"
            )
        data_idx = episode.get("data_idx")
        if isinstance(data_idx, bool) or not isinstance(data_idx, int):
            raise EvalError(f"Travel episode {index} has invalid data_idx")
        dataset_position = domain.get("dataset_position")
        if (
            isinstance(dataset_position, bool)
            or not isinstance(dataset_position, int)
            or dataset_position != data_idx
        ):
            raise EvalError(f"Travel episode {index} dataset_position mismatch")
        source_id = ledger.get("source_id")
        if isinstance(source_id, bool) or not isinstance(source_id, int):
            raise EvalError(f"Travel episode {index} source_id must be an integer")
        if not 1 <= source_id <= TRAVEL_RECORD_COUNT:
            raise EvalError(f"Travel episode {index} source_id is out of range")
        if source_id != data_idx + 1:
            raise EvalError(
                f"Travel episode {index} source_id does not match frozen position"
            )
        domain_source_id = domain.get("source_id")
        if (
            isinstance(domain_source_id, bool)
            or not isinstance(domain_source_id, int)
            or source_id != domain_source_id
        ):
            raise EvalError(f"Travel episode {index} source_id ledger mismatch")
        if source_id in source_ids:
            raise EvalError(f"Travel paper ledger repeats source_id {source_id!r}")
        source_ids.add(source_id)
        passed = ledger.get("full_pass_people")
        people = ledger.get("total_people")
        constraint_people = ledger.get("constraint_people")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (passed, people, constraint_people)
        ):
            raise EvalError(f"Travel episode {index} has non-integer paper counts")
        if (
            not 5 <= people <= 8
            or not 0 <= passed <= people
            or not 0 <= constraint_people <= people
        ):
            raise EvalError(f"Travel episode {index} has invalid paper counts")
        group_success = ledger.get("group_success")
        if type(group_success) is not bool or group_success != (passed == people):
            raise EvalError(f"Travel episode {index} has contradictory group_success")
        rate = ledger.get("group_constraint_rate")
        if constraint_people == 0:
            if rate is not None:
                raise EvalError(
                    f"Travel episode {index} constraint rate must be null with no people"
                )
        else:
            if (
                isinstance(rate, bool)
                or not isinstance(rate, (int, float))
                or not math.isfinite(float(rate))
                or not 0.0 <= float(rate) <= 1.0
            ):
                raise EvalError(
                    f"Travel episode {index} has invalid group_constraint_rate"
                )
            constraint_group_rates.append(float(rate))
        full_pass_people += passed
        total_people += people
        total_constraint_people += constraint_people
        successful_groups += int(group_success)
    group_count = len(episodes)
    if group_count == 0 or total_people == 0:
        raise EvalError("Travel paper metrics require at least one complete group")
    if group_count == TRAVEL_RECORD_COUNT and total_people != TRAVEL_PHASE_COUNT:
        raise EvalError(
            "Travel complete panel total_people disagrees with frozen phase count: "
            f"expected {TRAVEL_PHASE_COUNT}, observed {total_people}"
        )
    ps = 100.0 * full_pass_people / total_people
    sps = (
        100.0 * math.fsum(constraint_group_rates) / len(constraint_group_rates)
        if constraint_group_rates
        else 0.0
    )
    sr = 100.0 * successful_groups / group_count
    return {
        "metric_contract": contract,
        "dataset_scope": dataset_scope,
        "scale": "percent_0_100",
        "ps": ps,
        "sps": sps,
        "sr": sr,
        "full_pass_people": full_pass_people,
        "total_people": total_people,
        "successful_groups": successful_groups,
        "total_groups": group_count,
        "groups_with_constraint_people": len(constraint_group_rates),
        "total_constraint_people": total_constraint_people,
        "online_reward_is_separate": True,
    }


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_exact_mapping(
    label: str,
    value: Any,
    expected: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvalError(f"{label} must be a mapping")
    fields = set(value)
    expected_fields = set(expected)
    if fields != expected_fields:
        raise EvalError(
            f"{label} fields mismatch: missing={sorted(expected_fields - fields)} "
            f"extra={sorted(str(key) for key in fields - expected_fields)}"
        )
    mismatches = {
        key: (expected_value, value.get(key))
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    }
    if mismatches:
        raise EvalError(f"{label} mismatch: {mismatches}")
    return value


def _require_search_runtime_provenance(metadata: Mapping[str, Any]) -> None:
    exact_top_level = {
        "contract_id": SEARCH_PAPER_CONTRACT_ID,
        "contract_sha256": SEARCH_PAPER_CONTRACT_SHA256,
        "system_prompt_sha256": SEARCH_PAPER_SYSTEM_PROMPT_SHA256,
        "contract_mode": "paper_eval",
        "semantic_variant": (
            "paper_metric_evaluation_continue_on_incorrect_one_action_v1"
        ),
        "reward_contract": "evaluation_only_zero_reward_not_for_training",
        "reward_overlay": "none",
        "max_steps": 811,
        "max_total_actions": 811,
        "native_tool_ops": ["search", "get_document"],
        "native_search_k": 5,
        "native_snippet_max_tokens": 512,
        "judge": "memoryarena_browsecomp_gpt_judge_v1",
    }
    mismatches = {
        key: (expected, metadata.get(key))
        for key, expected in exact_top_level.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise EvalError(f"Search runtime contract provenance mismatch: {mismatches}")

    _require_exact_mapping(
        "Search total_action_budget",
        metadata.get("total_action_budget"),
        {
            "limit": 811,
            "enforced_by": "agentmemory_runtime_wrapper",
            "counts": ["native", "memory", "invalid"],
            "legacy_max_steps_field_is_same_limit": True,
            "native_action_allowance": 555,
            "memory_action_allowance": 256,
            "memory_action_allowance_per_phase": 16,
        },
    )
    _require_exact_mapping(
        "Search native_iteration_budget",
        metadata.get("native_iteration_budget"),
        {
            "subquery_per_phase": 35,
            "final_phase": 30,
            "counts": ["native", "invalid"],
            "memory_actions_consume_budget": False,
            "separately_tracked_from_total_action_budget": True,
            "upstream_batched_model_turn_parity": False,
        },
    )
    _require_exact_mapping(
        "Search snippet_tokenizer",
        metadata.get("snippet_tokenizer"),
        SEARCH_SNIPPET_TOKENIZER,
    )

    upstream = metadata.get("upstream_provenance")
    if not isinstance(upstream, Mapping):
        raise EvalError("Search metadata lacks upstream_provenance")
    if set(upstream) != {
        "mode",
        "memoryarena_commit",
        "source_files_sha256",
        "source_bundle_sha256",
    }:
        raise EvalError("Search upstream_provenance fields mismatch")
    source_files = upstream.get("source_files_sha256")
    if (
        upstream.get("mode") != "pinned_pristine_upstream"
        or upstream.get("memoryarena_commit") != SEARCH_MEMORYARENA_COMMIT
        or not isinstance(source_files, Mapping)
        or len(source_files) != SEARCH_UPSTREAM_SOURCE_FILE_COUNT
        or dict(source_files) != SEARCH_UPSTREAM_SOURCE_FILES_SHA256
        or any(
            not isinstance(path, str)
            or not path.endswith(".py")
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for path, digest in source_files.items()
        )
        or _canonical_json_sha256(source_files)
        != SEARCH_UPSTREAM_SOURCE_BUNDLE_SHA256
        or upstream.get("source_bundle_sha256")
        != SEARCH_UPSTREAM_SOURCE_BUNDLE_SHA256
    ):
        raise EvalError("Search upstream executable-source provenance mismatch")

    expected_assets = {
        "mode": "frozen_public_assets",
        "embedding_model": SEARCH_FROZEN_EMBEDDING_MODEL,
        "embedding_dimension": SEARCH_FROZEN_INDEX_DIMENSION,
        "document_count": SEARCH_FROZEN_DOCUMENT_COUNT,
        "index_repository": SEARCH_FROZEN_INDEX_REPOSITORY,
        "index_revision": SEARCH_FROZEN_INDEX_REVISION,
        "index_shards": [dict(item) for item in SEARCH_FROZEN_INDEX_SHARDS],
        "corpus_repository": SEARCH_FROZEN_CORPUS_REPOSITORY,
        "corpus_revision": SEARCH_FROZEN_CORPUS_REVISION,
        "corpus_source_shards": [
            {
                "name": name,
                "sha256": digest,
                "size_bytes": size,
                "row_count": rows,
            }
            for name, digest, size, rows in SEARCH_FROZEN_CORPUS_SOURCE_SHARDS
        ],
        "corpus_sha256": SEARCH_FROZEN_CORPUS_SHA256,
        "corpus_manifest_sha256": SEARCH_FROZEN_CORPUS_MANIFEST_SHA256,
    }
    _require_exact_mapping(
        "Search search_asset_provenance",
        metadata.get("search_asset_provenance"),
        expected_assets,
    )

    embedding = metadata.get("embedding_route_provenance")
    if not isinstance(embedding, Mapping) or set(embedding) != {
        "mode",
        "provider",
        "model",
        "endpoint_sha256",
        "route_variant",
        "config_sha256",
    }:
        raise EvalError("Search embedding_route_provenance fields mismatch")
    embedding_config = {
        "provider": "openai",
        "model": SEARCH_FROZEN_EMBEDDING_MODEL,
        "endpoint_sha256": embedding.get("endpoint_sha256"),
        "route_variant": "paper_eval_openai_embedding_v1",
    }
    if (
        embedding.get("mode") != "explicit_hashed_embedding_route"
        or not _valid_sha256_or_none(embedding.get("endpoint_sha256"))
        or embedding.get("endpoint_sha256") is None
        or any(embedding.get(key) != value for key, value in embedding_config.items())
        or embedding.get("config_sha256")
        != _canonical_json_sha256(embedding_config)
    ):
        raise EvalError("Search embedding route provenance mismatch")

    judge = metadata.get("judge_provenance")
    if not isinstance(judge, Mapping) or set(judge) != {
        "mode",
        "backend",
        "model",
        "max_tokens",
        "endpoint_sha256",
        "prompt_template_sha256",
        "config_sha256",
    }:
        raise EvalError("Search judge_provenance fields mismatch")
    judge_config = {
        "mode": "upstream_memoryarena_judge",
        "backend": "openai_responses",
        "model": SEARCH_JUDGE_MODEL,
        "max_tokens": SEARCH_JUDGE_MAX_TOKENS,
        "endpoint_sha256": judge.get("endpoint_sha256"),
        "prompt_template_sha256": SEARCH_JUDGE_PROMPT_TEMPLATE_SHA256,
    }
    if (
        not _valid_sha256_or_none(judge.get("endpoint_sha256"))
        or judge.get("endpoint_sha256") is None
        or any(judge.get(key) != value for key, value in judge_config.items())
        or judge.get("config_sha256") != _canonical_json_sha256(judge_config)
    ):
        raise EvalError("Search judge provenance mismatch")


def _require_search_paper_metadata(
    metadata: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(metadata, Mapping):
        raise EvalError("Search environment metadata must be a mapping")
    if metadata.get("surface") != SEARCH_PAPER_EVAL_SURFACE:
        raise EvalError("Search paper metrics require the paper-eval runtime surface")
    if metadata.get("domain_id") != "progressive_search":
        raise EvalError("Search paper surface requires domain_id='progressive_search'")
    _require_search_runtime_provenance(metadata)
    _require_public_dataset(
        metadata,
        config="progressive_search",
        record_count=SEARCH_RECORD_COUNT,
        phase_count=SEARCH_PHASE_COUNT,
    )
    metadata_phase_count = metadata.get("phase_count")
    if (
        isinstance(metadata_phase_count, bool)
        or not isinstance(metadata_phase_count, int)
        or metadata_phase_count != SEARCH_PHASE_COUNT
    ):
        raise EvalError(
            "progressive_search metadata phase_count must be "
            f"{SEARCH_PHASE_COUNT}, observed {metadata.get('phase_count')!r}"
        )
    paper = _paper_evaluation_metadata(
        metadata,
        expected_contract=SEARCH_PAPER_METRIC_CONTRACT,
    )
    fields = set(paper)
    if fields != SEARCH_PAPER_METADATA_FIELDS:
        missing = sorted(SEARCH_PAPER_METADATA_FIELDS - fields)
        extra = sorted(str(key) for key in fields - SEARCH_PAPER_METADATA_FIELDS)
        raise EvalError(
            "Search paper_evaluation metadata fields mismatch: "
            f"missing={missing} extra={extra}"
        )
    expected = {
        "id": SEARCH_PAPER_METRIC_CONTRACT,
        "dataset_scope": SEARCH_PAPER_DATASET_SCOPE,
        "available": True,
        "metrics": ["PS", "SR@k", "SR"],
        "metric_scale": "unit_interval",
        "paper_panel_complete": False,
        "public_task_count": SEARCH_RECORD_COUNT,
        "paper_task_count": SEARCH_PAPER_TASK_COUNT,
        "separate_from_online_reward": True,
    }
    mismatches = {
        key: (value, paper.get(key))
        for key, value in expected.items()
        if paper.get(key) != value
    }
    if mismatches:
        raise EvalError(f"Search paper_evaluation metadata mismatch: {mismatches}")
    for key in (
        "available",
        "paper_panel_complete",
        "separate_from_online_reward",
    ):
        if type(paper.get(key)) is not bool:
            raise EvalError(f"Search paper_evaluation metadata {key} must be boolean")
    for key in ("public_task_count", "paper_task_count"):
        if isinstance(paper.get(key), bool) or not isinstance(paper.get(key), int):
            raise EvalError(f"Search paper_evaluation metadata {key} must be integer")
    return paper


def _valid_sha256_or_none(value: Any) -> bool:
    return value is None or (
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    )


def aggregate_search_paper_metrics(
    episodes: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate public221 Search PS, depth-specific SR@k, and final SR."""

    _require_search_paper_metadata(metadata)
    if not episodes:
        raise EvalError("Search paper metrics require completed task evidence")

    process_scores: list[float] = []
    final_successes = 0
    phase_total = 0
    depth_correct: dict[int, int] = {}
    depth_eligible: dict[int, int] = {}
    data_indices: set[int] = set()
    query_ids: set[str] = set()

    for episode_index, episode in enumerate(episodes):
        after = _terminal_env_info(episode, episode_index)
        domain = after.get("domain_evidence")
        if not isinstance(domain, Mapping):
            raise EvalError(
                f"Search episode {episode_index} lacks terminal domain evidence"
            )
        ledger = domain.get("paper_evaluation")
        if not isinstance(ledger, Mapping):
            raise EvalError(
                f"Search episode {episode_index} lacks paper_evaluation ledger"
            )
        fields = set(ledger)
        if fields != SEARCH_PAPER_LEDGER_FIELDS:
            missing = sorted(SEARCH_PAPER_LEDGER_FIELDS - fields)
            extra = sorted(str(key) for key in fields - SEARCH_PAPER_LEDGER_FIELDS)
            raise EvalError(
                f"Search episode {episode_index} paper ledger fields mismatch: "
                f"missing={missing} extra={extra}"
            )
        expected_pairs = {
            "metric_contract": SEARCH_PAPER_METRIC_CONTRACT,
            "dataset_scope": SEARCH_PAPER_DATASET_SCOPE,
            "complete": True,
            "metric_scale": "unit_interval",
            "online_reward_is_separate": True,
        }
        for key, expected in expected_pairs.items():
            if ledger.get(key) != expected:
                raise EvalError(
                    f"Search episode {episode_index} paper ledger {key} mismatch: "
                    f"expected {expected!r}, observed {ledger.get(key)!r}"
                )
        if type(ledger.get("complete")) is not bool or type(
            ledger.get("online_reward_is_separate")
        ) is not bool:
            raise EvalError(
                f"Search episode {episode_index} paper completion flags must be boolean"
            )

        data_idx = episode.get("data_idx")
        if (
            isinstance(data_idx, bool)
            or not isinstance(data_idx, int)
            or not 0 <= data_idx < SEARCH_RECORD_COUNT
        ):
            raise EvalError(f"Search episode {episode_index} has invalid data_idx")
        if data_idx in data_indices:
            raise EvalError(f"Search paper evaluation repeats data_idx {data_idx}")
        data_indices.add(data_idx)
        expected_query_id = str(data_idx)
        expected_phase_count = SEARCH_TASK_PHASE_COUNTS[data_idx]
        initial = episode.get("initial_env_info")
        initial_domain = (
            initial.get("domain_evidence") if isinstance(initial, Mapping) else None
        )
        if not isinstance(initial_domain, Mapping):
            raise EvalError(
                f"Search episode {episode_index} lacks reset task identity"
            )
        for label, info, evidence in (
            ("initial", initial, initial_domain),
            ("terminal", after, domain),
        ):
            if evidence.get("query_id") != expected_query_id:
                raise EvalError(
                    f"Search episode {episode_index} {label} query_id mismatch"
                )
            if evidence.get("contract_mode") != "paper_eval":
                raise EvalError(
                    f"Search episode {episode_index} {label} contract mode mismatch"
                )
            if (
                type(info.get("phase_count")) is not int
                or info.get("phase_count") != expected_phase_count
            ):
                raise EvalError(
                    f"Search episode {episode_index} {label} phase_count mismatch"
                )
        if type(initial.get("phase_index")) is not int or initial.get(
            "phase_index"
        ) != 0:
            raise EvalError(
                f"Search episode {episode_index} did not start at phase zero"
            )

        query_id = ledger.get("query_id")
        if (
            not isinstance(query_id, str)
            or not query_id.strip()
            or query_id != query_id.strip()
        ):
            raise EvalError(f"Search episode {episode_index} has invalid query_id")
        if query_id != expected_query_id:
            raise EvalError(f"Search episode {episode_index} query_id mismatch")
        if domain.get("query_id") != query_id:
            raise EvalError(f"Search episode {episode_index} query_id evidence conflicts")
        if domain.get("contract_mode") != "paper_eval":
            raise EvalError(f"Search episode {episode_index} is not paper_eval evidence")
        if query_id in query_ids:
            raise EvalError(f"Search paper evaluation repeats query_id {query_id!r}")
        query_ids.add(query_id)

        episode_return = episode.get("episode_return")
        if (
            isinstance(episode_return, bool)
            or not isinstance(episode_return, (int, float))
            or not math.isfinite(float(episode_return))
            or float(episode_return) != 0.0
        ):
            raise EvalError(
                f"Search episode {episode_index} paper-eval return must be zero"
            )

        verdicts = ledger.get("phase_verdicts")
        if not isinstance(verdicts, list) or not verdicts:
            raise EvalError(f"Search episode {episode_index} has invalid phase ledger")
        if domain.get("phase_verdict_ledger") != verdicts:
            raise EvalError(
                f"Search episode {episode_index} phase verdict evidence conflicts"
            )
        phase_count = len(verdicts)
        if phase_count != expected_phase_count:
            raise EvalError(
                f"Search episode {episode_index} phase count mismatch: expected "
                f"{expected_phase_count}, observed {phase_count}"
            )
        for count_key in (
            "completed_phase_count",
            "process_score_denominator",
        ):
            count = ledger.get(count_key)
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or count != phase_count
            ):
                raise EvalError(
                    f"Search episode {episode_index} {count_key} disagrees with ledger"
                )

        correct_flags: list[bool] = []
        for phase_index, verdict in enumerate(verdicts):
            if not isinstance(verdict, Mapping):
                raise EvalError(
                    f"Search episode {episode_index} phase {phase_index} is not a mapping"
                )
            verdict_fields = set(verdict)
            if verdict_fields != SEARCH_PHASE_VERDICT_FIELDS:
                missing = sorted(SEARCH_PHASE_VERDICT_FIELDS - verdict_fields)
                extra = sorted(
                    str(key) for key in verdict_fields - SEARCH_PHASE_VERDICT_FIELDS
                )
                raise EvalError(
                    f"Search episode {episode_index} phase {phase_index} fields "
                    f"mismatch: missing={missing} extra={extra}"
                )
            observed_phase_index = verdict.get("phase_index")
            if (
                isinstance(observed_phase_index, bool)
                or not isinstance(observed_phase_index, int)
                or observed_phase_index != phase_index
            ):
                raise EvalError(
                    f"Search episode {episode_index} phase indices are not contiguous"
                )
            expected_kind = "final" if phase_index == phase_count - 1 else "subquery"
            if verdict.get("phase_kind") != expected_kind:
                raise EvalError(
                    f"Search episode {episode_index} phase {phase_index} kind mismatch"
                )
            correct = verdict.get("correct")
            if type(correct) is not bool:
                raise EvalError(
                    f"Search episode {episode_index} phase {phase_index} lacks verdict"
                )
            source = verdict.get("verdict_source")
            if source not in {
                "memoryarena_llm_judge",
                "phase_budget_exhausted_without_submission",
            }:
                raise EvalError(
                    f"Search episode {episode_index} phase {phase_index} has "
                    "unsupported verdict_source"
                )
            answer_sha = verdict.get("answer_sha256")
            judge_sha = verdict.get("judge_response_sha256")
            if not _valid_sha256_or_none(answer_sha) or not _valid_sha256_or_none(
                judge_sha
            ):
                raise EvalError(
                    f"Search episode {episode_index} phase {phase_index} has invalid hash"
                )
            confidence = verdict.get("judge_confidence")
            if confidence is not None and (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(float(confidence))
            ):
                raise EvalError(
                    f"Search episode {episode_index} phase {phase_index} has invalid "
                    "judge_confidence"
                )
            if verdict.get("judge_parse_error") is not False:
                raise EvalError(
                    f"Search episode {episode_index} phase {phase_index} has parse error"
                )
            docids = verdict.get("retrieved_docids")
            if not isinstance(docids, list) or any(
                not isinstance(docid, str) for docid in docids
            ):
                raise EvalError(
                    f"Search episode {episode_index} phase {phase_index} has invalid "
                    "retrieved_docids"
                )
            if source == "memoryarena_llm_judge":
                if answer_sha is None or judge_sha is None:
                    raise EvalError(
                        f"Search episode {episode_index} judged phase lacks hashes"
                    )
            elif (
                correct
                or answer_sha is not None
                or judge_sha is not None
                or confidence is not None
            ):
                raise EvalError(
                    f"Search episode {episode_index} exhausted phase evidence conflicts"
                )
            correct_flags.append(correct)

        correct_count = sum(correct_flags)
        numerator = ledger.get("process_score_numerator")
        if (
            isinstance(numerator, bool)
            or not isinstance(numerator, int)
            or numerator != correct_count
        ):
            raise EvalError(
                f"Search episode {episode_index} process-score numerator conflicts"
            )
        expected_process_score = correct_count / phase_count
        process_score = ledger.get("process_score")
        if (
            isinstance(process_score, bool)
            or not isinstance(process_score, (int, float))
            or not math.isfinite(float(process_score))
            or not math.isclose(
                float(process_score), expected_process_score, rel_tol=1e-12, abs_tol=1e-12
            )
        ):
            raise EvalError(
                f"Search episode {episode_index} process score conflicts with verdicts"
            )

        sr_at_k = ledger.get("sr_at_k")
        expected_depth_keys = {str(depth) for depth in range(1, phase_count + 1)}
        if not isinstance(sr_at_k, Mapping) or set(sr_at_k) != expected_depth_keys:
            raise EvalError(f"Search episode {episode_index} has invalid SR@k depths")
        for depth, correct in enumerate(correct_flags, start=1):
            contribution = sr_at_k[str(depth)]
            if not isinstance(contribution, Mapping) or set(
                contribution
            ) != SEARCH_SR_AT_K_FIELDS:
                raise EvalError(
                    f"Search episode {episode_index} SR@{depth} fields mismatch"
                )
            if (
                type(contribution.get("correct")) is not bool
                or contribution.get("correct") is not correct
                or isinstance(contribution.get("numerator"), bool)
                or not isinstance(contribution.get("numerator"), int)
                or contribution.get("numerator") != int(correct)
                or isinstance(contribution.get("denominator"), bool)
                or not isinstance(contribution.get("denominator"), int)
                or contribution.get("denominator") != 1
            ):
                raise EvalError(
                    f"Search episode {episode_index} SR@{depth} contribution conflicts"
                )

        final_success = ledger.get("final_success")
        final_numerator = ledger.get("final_sr_numerator")
        final_denominator = ledger.get("final_sr_denominator")
        if (
            type(final_success) is not bool
            or final_success is not correct_flags[-1]
            or isinstance(final_numerator, bool)
            or not isinstance(final_numerator, int)
            or final_numerator != int(final_success)
            or isinstance(final_denominator, bool)
            or not isinstance(final_denominator, int)
            or final_denominator != 1
        ):
            raise EvalError(
                f"Search episode {episode_index} final SR evidence conflicts"
            )
        if type(episode.get("episode_success")) is not bool or episode.get(
            "episode_success"
        ) is not final_success:
            raise EvalError(
                f"Search episode {episode_index} terminal success evidence conflicts"
            )
        progress = episode.get("final_phase_progress")
        progress_index = (
            progress.get("phase_index_after")
            if isinstance(progress, Mapping)
            else None
        )
        progress_count = (
            progress.get("phase_count") if isinstance(progress, Mapping) else None
        )
        if (
            not isinstance(progress, Mapping)
            or isinstance(progress_index, bool)
            or not isinstance(progress_index, int)
            or progress_index != phase_count
            or isinstance(progress_count, bool)
            or not isinstance(progress_count, int)
            or progress_count != phase_count
        ):
            raise EvalError(
                f"Search episode {episode_index} terminal phase evidence conflicts"
            )

        process_scores.append(expected_process_score)
        final_successes += int(final_success)
        phase_total += phase_count
        for depth, correct in enumerate(correct_flags, start=1):
            depth_eligible[depth] = depth_eligible.get(depth, 0) + 1
            depth_correct[depth] = depth_correct.get(depth, 0) + int(correct)

    task_count = len(episodes)
    if task_count == SEARCH_RECORD_COUNT:
        if data_indices != set(range(SEARCH_RECORD_COUNT)):
            raise EvalError("Search complete public panel row coverage mismatch")
        if query_ids != {str(index) for index in range(SEARCH_RECORD_COUNT)}:
            raise EvalError("Search complete public panel query identity mismatch")
        if phase_total != SEARCH_PHASE_COUNT:
            raise EvalError(
                "Search complete public panel phase count mismatch: "
                f"expected {SEARCH_PHASE_COUNT}, observed {phase_total}"
            )
    process_score_sum = math.fsum(process_scores)
    return {
        "metric_contract": SEARCH_PAPER_METRIC_CONTRACT,
        "dataset_scope": SEARCH_PAPER_DATASET_SCOPE,
        "metric_scale": "unit_interval",
        "task_count": task_count,
        "phase_count": phase_total,
        "process_score_numerator": process_score_sum,
        "process_score_denominator": task_count,
        "process_score": process_score_sum / task_count,
        "sr_at_k": {
            str(depth): {
                "correct_tasks": depth_correct[depth],
                "eligible_tasks": depth_eligible[depth],
                "rate": depth_correct[depth] / depth_eligible[depth],
            }
            for depth in sorted(depth_eligible)
        },
        "final_sr_numerator": final_successes,
        "final_sr_denominator": task_count,
        "final_success_rate": final_successes / task_count,
        "public_panel_complete": task_count == SEARCH_RECORD_COUNT,
        "paper_panel_complete": False,
        "paper_task_count": SEARCH_PAPER_TASK_COUNT,
        "online_reward_is_separate": True,
    }


def summarize_paper_surface(
    episodes: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach a reproducible paper-column summary to one surface run."""

    summary = summarize_episodes(episodes)
    surface = metadata.get("surface") if isinstance(metadata, Mapping) else None
    if surface not in PAPER_SURFACE_REGISTRY:
        return summary

    registration = resolve_paper_surface(metadata)
    metric_mode = registration["metric_mode"]
    paper_metrics: dict[str, Any] | None = None
    paper_metric_contract: str | None = None
    paper_success_rate: float | None
    expected_task_count: int | None = None

    if metric_mode == "travel_paper_ledger":
        paper_metrics = aggregate_travel_paper_metrics(episodes, metadata)
        paper_metric_contract = paper_metrics["metric_contract"]
        paper_success_rate = float(paper_metrics["sr"]) / 100.0
        expected_task_count = TRAVEL_RECORD_COUNT
    elif metric_mode == "travel_failfast_diagnostic":
        paper_metric_contract = "travel_failfast_diagnostic_only"
        paper_success_rate = None
        expected_task_count = TRAVEL_RECORD_COUNT
    elif metric_mode == "episode_success":
        paper_metric_contract = "episode_success"
        paper_success_rate = float(summary["success_rate"])
        if registration["surface"] == WEBSHOP_V2_SURFACE:
            paper_metrics = aggregate_shopping_panel_evidence(episodes, metadata)
            expected_task_count = SHOPPING_BUNDLE_COUNT
        elif registration["surface"] in FORMAL_SURFACE_DATASETS:
            paper_metrics = aggregate_formal_panel_evidence(episodes, metadata)
            expected_task_count = len(
                FORMAL_TASK_PHASE_COUNTS[registration["surface"]]
            )
        else:  # pragma: no cover - registry construction owns this invariant
            raise EvalError(
                "episode_success surface lacks a frozen panel validator: "
                f"{registration['surface']}"
            )
    elif metric_mode == "search_failfast_diagnostic":
        paper_metric_contract = "search_failfast_diagnostic_only"
        paper_success_rate = None
        expected_task_count = SEARCH_RECORD_COUNT
    elif metric_mode == "search_paper_ledger":
        paper_metrics = aggregate_search_paper_metrics(episodes, metadata)
        paper_metric_contract = paper_metrics["metric_contract"]
        paper_success_rate = float(paper_metrics["final_success_rate"])
        expected_task_count = SEARCH_RECORD_COUNT
    elif metric_mode == "formal_paper_ledger":
        paper_metrics = aggregate_formal_panel_evidence(episodes, metadata)
        paper_metric_contract = paper_metrics["metric_contract"]
        paper_success_rate = float(paper_metrics["final_success_rate"])
        expected_task_count = len(
            FORMAL_TASK_PHASE_COUNTS[registration["surface"]]
        )
    else:  # pragma: no cover - registry construction owns this invariant
        raise EvalError(f"unsupported paper metric mode: {metric_mode!r}")

    coverage = _panel_coverage(
        episodes,
        metadata,
        expected_task_count=expected_task_count,
    )
    attested_panel_complete = coverage["panel_complete"]
    if paper_metrics is not None and "paper_panel_complete" in paper_metrics:
        attested_panel_complete = bool(
            attested_panel_complete
            and paper_metrics.get("paper_panel_complete") is True
        )
    if registration["paper_column"] == "Search":
        # Both registered Search modes use the public 221/256 release.
        attested_panel_complete = False
    eligible = bool(
        registration["canonical_macro_candidate"]
        and attested_panel_complete
        and paper_success_rate is not None
    )
    summary.update(
        {
            "paper_column": registration["paper_column"],
            "paper_surface": registration["surface"],
            "paper_variant": registration["variant"],
            "paper_metric_mode": metric_mode,
            "paper_metric_contract": paper_metric_contract,
            "paper_success_rate": paper_success_rate,
            "paper_metrics": paper_metrics,
            "paper_macro_eligible": eligible,
            "paper_panel_complete": attested_panel_complete,
            **coverage,
        }
    )
    return summary


def token_ids_hash(token_ids: Sequence[int]) -> str:
    """Hash a canonical JSON representation of integer token ids."""

    normalized = [int(item) for item in token_ids]
    payload = json.dumps(
        normalized, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_copy(value: Any) -> Any:
    """Make JSON-shaped evidence independent from mutable HTTP payloads."""

    return copy.deepcopy(value)


class JsonHttp:
    """Small injectable JSON HTTP transport used by the evaluator and tests."""

    def __init__(
        self,
        *,
        timeout: float = 300.0,
        opener: Callable[..., Any] | None = None,
        api_key: str | None = None,
    ) -> None:
        self.timeout = float(timeout)
        self.opener = opener or urlopen
        self.api_key = api_key

    def request(self, method: str, url: str, payload: Mapping[str, Any] | None = None) -> Any:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(url, data=body, headers=headers, method=method)
        try:
            response = self.opener(request, timeout=self.timeout)
            status = int(getattr(response, "status", 200))
            raw = response.read()
        except HTTPError as exc:
            try:
                raw = exc.read()
            except Exception:  # pragma: no cover - defensive for odd clients
                raw = b""
            text = raw.decode("utf-8", errors="replace")
            raise HttpError(method, url, int(exc.code), text) from exc
        except URLError as exc:
            raise EvalError(f"{method} {url} failed: {exc}") from exc
        except OSError as exc:
            raise EvalError(f"{method} {url} failed: {exc}") from exc

        text = raw.decode("utf-8", errors="replace")
        if status < 200 or status >= 300:
            raise HttpError(method, url, status, text)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise EvalError(f"{method} {url} returned non-JSON: {text[:500]}") from exc

    def get(self, url: str) -> Any:
        return self.request("GET", url)

    def post(self, url: str, payload: Mapping[str, Any]) -> Any:
        return self.request("POST", url, payload)


def _trim_base(url: str) -> str:
    return url.rstrip("/")


def _v1_url(model_url: str, path: str) -> str:
    base = _trim_base(model_url)
    if base.endswith("/v1"):
        return base + "/" + path.lstrip("/")
    return base + "/v1/" + path.lstrip("/")


def _server_root(model_url: str) -> str:
    base = _trim_base(model_url)
    return base[:-3] if base.endswith("/v1") else base


def _extract_int_ids(value: Any) -> list[int] | None:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        return None
    return [int(item) for item in value]


def _find_explicit_token_ids(payload: Any, keys: Sequence[str]) -> list[int] | None:
    """Find an explicit id sequence without treating text/tokens as ids."""

    if isinstance(payload, Mapping):
        for key in keys:
            ids = _extract_int_ids(payload.get(key))
            if ids is not None:
                return ids
        # vLLM may wrap tokenization data under ``data`` or ``result``.
        for key in ("data", "result", "output"):
            if key in payload:
                ids = _find_explicit_token_ids(payload[key], keys)
                if ids is not None:
                    return ids
    elif isinstance(payload, list):
        for item in payload:
            ids = _find_explicit_token_ids(item, keys)
            if ids is not None:
                return ids
    return None


class AgentMemoryEnvClient:
    """HTTP-only equivalent of the repository's AgentMemoryEnvClient.

    Keeping this client local to the evidence driver avoids importing the
    training package (and therefore avoids a torch dependency) on eval hosts.
    The endpoint payloads, v3 raw-action behavior, and v2 trailing ``</s>``
    handling match the canonical client.
    """

    def __init__(self, env_url: str, transport: JsonHttp):
        self.base_url = _trim_base(env_url)
        self.transport = transport
        self.metadata = self._get("metadata")
        if not isinstance(self.metadata, Mapping):
            raise EvalError("AgentMemory /metadata must return a JSON object")
        self.metadata = dict(self.metadata)
        self._validate_metadata()
        self.is_v3 = self.metadata.get("formal_schema_version") == FORMAL_SCHEMA_V3
        self.surface = str(self.metadata.get("surface", ""))
        self.system_prompt = str(self.metadata["system_prompt"])
        self.system_prompt_source = str(self.metadata["system_prompt_source"])
        self.env_id: int | None = None
        self.last_submitted_action: str | None = None
        self.info: dict[str, Any] = {}
        self.create()

    def _get(self, path: str) -> Any:
        return self.transport.get(f"{self.base_url}/{path.lstrip('/')}")

    def _post(self, path: str, payload: Mapping[str, Any]) -> Any:
        return self.transport.post(
            f"{self.base_url}/{path.lstrip('/')}", payload
        )

    def _validate_metadata(self) -> None:
        schema = self.metadata.get("formal_schema_version")
        if schema not in (FORMAL_SCHEMA_V3, None):
            raise EvalError(f"Unsupported AgentMemory formal schema: {schema!r}")
        surface = self.metadata.get("surface")
        if schema is None and surface != WEBSHOP_V2_SURFACE:
            raise EvalError(
                "Legacy AgentMemory metadata is accepted only for native WebShop v2"
            )
        system_prompt = self.metadata.get("system_prompt")
        if schema == FORMAL_SCHEMA_V3:
            if not isinstance(system_prompt, str) or not system_prompt.strip():
                raise EvalError("AgentMemory v3 /metadata has no canonical system_prompt")
            expected = self.metadata.get("system_prompt_sha256")
            observed = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
            if expected != observed:
                raise EvalError(
                    "AgentMemory metadata system_prompt_sha256 does not match prompt"
                )
            self.metadata["system_prompt_source"] = "server_metadata"
            return

        if system_prompt is None:
            self.metadata["system_prompt"] = LEGACY_WEBSHOP_SYSTEM_PROMPT
            self.metadata["system_prompt_sha256"] = hashlib.sha256(
                LEGACY_WEBSHOP_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest()
            self.metadata["system_prompt_source"] = "webshop_v2_rollout_fallback"
            return
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise EvalError("Native WebShop system_prompt must be non-empty text")
        if system_prompt != LEGACY_WEBSHOP_SYSTEM_PROMPT:
            raise EvalError(
                "Native WebShop metadata system_prompt disagrees with the "
                "canonical rollout fallback"
            )
        expected = self.metadata.get("system_prompt_sha256")
        observed = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
        if expected is not None and expected != observed:
            raise EvalError(
                "Native WebShop metadata system_prompt_sha256 does not match prompt"
            )
        self.metadata["system_prompt_sha256"] = observed
        self.metadata["system_prompt_source"] = "server_metadata"

    def create(self) -> dict[str, Any]:
        result = self._post("create", {})
        self.env_id = int(result["id"])
        self._set_info(result)
        return _json_copy(result)

    def _set_info(self, result: Mapping[str, Any]) -> None:
        self.info = {
            "observation": result.get("observation", ""),
            "reward": result.get("reward", 0.0),
            "done": bool(result.get("done", False)),
            "env_info": _json_copy(result.get("info", {})),
            "metadata": _json_copy(self.metadata),
        }

    def reset(self, data_idx: int) -> dict[str, Any]:
        if self.env_id is None:
            raise EvalError("Environment was not created")
        result = self._post("reset", {"id": self.env_id, "data_idx": int(data_idx)})
        self._set_info(result)
        return _json_copy(result)

    def step(self, action: str) -> dict[str, Any]:
        if self.env_id is None:
            raise EvalError("Environment was not created")
        if not isinstance(action, str):
            raise EvalError("Model action must be text")
        # This is the only normalization done by the canonical client for v3.
        submitted = action[:-4] if action.endswith("</s>") else action
        if not self.is_v3:
            submitted = extract_webshop_v2_action(submitted)
        self.last_submitted_action = submitted
        result = self._post("step", {"id": self.env_id, "action": submitted})
        self._set_info(result)
        return _json_copy(result)

    def close(self) -> Any:
        if self.env_id is None:
            return None
        result = self._post("close", {"id": self.env_id})
        self.env_id = None
        return _json_copy(result)


_THINK_CLOSE_RE = re.compile(r"</think\s*>", flags=re.IGNORECASE)
_NATIVE_ACTION_RE = re.compile(r"\A(search|click)\[([^\[\]\r\n]+)\]\Z")
_MEMORY_ACTION_NAMES = ("ADD", "UPDATE", "DELETE", "RETRIEVE", "SUMMARY", "FILTER")
_MEMORY_ACTION_RE = re.compile(
    r"\A(" + "|".join(_MEMORY_ACTION_NAMES) + r")\s+(\{.*\})\Z",
    flags=re.DOTALL,
)


def _normalize_webshop_v2_action(candidate: str) -> str | None:
    cleaned = candidate.strip()
    native_match = _NATIVE_ACTION_RE.fullmatch(cleaned)
    if native_match is not None:
        return f"{native_match.group(1)}[{native_match.group(2).strip()}]"
    memory_match = _MEMORY_ACTION_RE.fullmatch(cleaned)
    if memory_match is None:
        return None
    try:
        payload = json.loads(memory_match.group(2))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return f"{memory_match.group(1)} {json.dumps(payload, ensure_ascii=False)}"


def extract_webshop_v2_action(text: str) -> str:
    """Match the legacy ReAct parser's action extraction without dependencies."""

    original = text
    cleaned = text.strip()
    matches = list(_THINK_CLOSE_RE.finditer(cleaned))
    if matches:
        cleaned = cleaned[matches[-1].end() :].strip()
    if "Action:" not in cleaned:
        bare = _normalize_webshop_v2_action(cleaned)
        if bare is not None:
            return bare
    else:
        parsed = _normalize_webshop_v2_action(
            cleaned.rsplit("Action:", 1)[-1]
        )
        if parsed is not None:
            return parsed
    # The canonical client falls back to the original sampled text when its
    # parser returns an empty/invalid action.  Preserve that behavior so the
    # environment, rather than this harness, judges malformed output.
    return original


class OpenAIChatClient:
    def __init__(
        self,
        model_url: str,
        model: str,
        transport: JsonHttp,
        *,
        max_tokens: int,
        temperature: float,
        enable_thinking: bool = False,
    ) -> None:
        self.model_url = _trim_base(model_url)
        self.model = model
        self.transport = transport
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.enable_thinking = bool(enable_thinking)

    def _chat_template_kwargs(self) -> dict[str, bool]:
        return {"enable_thinking": self.enable_thinking}

    def complete(self, messages: Sequence[Mapping[str, str]]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "n": 1,
            "stream": False,
            "chat_template_kwargs": self._chat_template_kwargs(),
        }
        result = self.transport.post(_v1_url(self.model_url, "chat/completions"), payload)
        if not isinstance(result, Mapping):
            raise EvalError("chat/completions returned a non-object JSON payload")
        return _json_copy(result)

    def tokenize(self, messages: Sequence[Mapping[str, str]]) -> tuple[list[int], dict[str, Any], str]:
        payload = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "add_generation_prompt": True,
            "chat_template_kwargs": self._chat_template_kwargs(),
        }
        # vLLM releases have exposed this endpoint both at /tokenize and
        # /v1/tokenize.  Try the canonical server endpoint first, then the
        # OpenAI-prefixed alias; any failure after both attempts is fatal.
        urls = [
            _server_root(self.model_url) + "/tokenize",
            _v1_url(self.model_url, "tokenize"),
        ]
        errors: list[str] = []
        for url in dict.fromkeys(urls):
            try:
                result = self.transport.post(url, payload)
            except (HttpError, EvalError) as exc:
                errors.append(str(exc))
                continue
            ids = _find_explicit_token_ids(result, ("tokens", "token_ids", "prompt_token_ids"))
            if ids is None:
                errors.append(f"{url} returned no integer token id sequence")
                continue
            return ids, _json_copy(result), url
        raise TokenizationError(
            "Authoritative vLLM /tokenize evidence unavailable; refusing to "
            "fabricate prompt token ids. Attempts: " + " | ".join(errors)
        )


def build_latest_observation_messages(
    system_prompt: str, observation: str
) -> list[dict[str, str]]:
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise EvalError("system_prompt must be non-empty")
    if not isinstance(observation, str):
        raise EvalError("latest observation must be text")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": observation},
    ]


def _completion_text(result: Mapping[str, Any]) -> str:
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise EvalError("chat/completions response has no choices[0]")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise EvalError("chat/completions response has no choices[0].message")
    content = message.get("content")
    if isinstance(content, str):
        return content
    # Some OpenAI-compatible servers return structured content parts.
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        if parts:
            return "".join(parts)
    raise EvalError("chat/completions response content is not text")


def _response_token_ids(result: Mapping[str, Any]) -> list[int] | None:
    choices = result.get("choices")
    if isinstance(choices, list) and choices:
        ids = _find_explicit_token_ids(
            choices[0],
            (
                "completion_token_ids",
                "response_token_ids",
                "output_token_ids",
                "token_ids",
            ),
        )
        if ids is not None:
            return ids
    return _find_explicit_token_ids(
        result,
        ("completion_token_ids", "response_token_ids", "output_token_ids"),
    )


def _phase_progress(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    def phase(info: Mapping[str, Any]) -> Any:
        if "phase_index" in info:
            return info.get("phase_index")
        return info.get("current_subtask_index")

    before_index = phase(before)
    after_index = phase(after)
    count = after.get("phase_count", before.get("phase_count"))
    if count is None:
        count = after.get("subtask_count", before.get("subtask_count"))
    advanced = None
    if before_index is not None and after_index is not None:
        try:
            advanced = int(after_index) > int(before_index)
        except (TypeError, ValueError):
            advanced = None
    is_v3 = (
        after.get("formal_schema_version") == FORMAL_SCHEMA_V3
        or before.get("formal_schema_version") == FORMAL_SCHEMA_V3
    )
    if is_v3 and ("progress_score" in before or "progress_score" in after):
        raise EvalError(
            "v3 environment returned ambiguous progress_score; use "
            "workflow_progress for phase traversal and a separate judged "
            "ledger for correctness-based PS"
        )
    server_workflow_progress = after.get(
        "workflow_progress",
        before.get("workflow_progress"),
    )
    workflow_progress = server_workflow_progress
    workflow_progress_source = (
        "server_workflow_progress"
        if server_workflow_progress is not None
        else "derived_phase_index"
    )
    if not is_v3 and workflow_progress is None:
        workflow_progress = after.get("progress_score", before.get("progress_score"))
        if workflow_progress is not None:
            workflow_progress_source = "legacy_progress_score"
    if workflow_progress is None and count not in (None, 0) and after_index is not None:
        try:
            workflow_progress = float(after_index) / float(count)
        except (TypeError, ValueError, ZeroDivisionError):
            workflow_progress = None
    return {
        "phase_index_before": before_index,
        "phase_index_after": after_index,
        "phase_count": count,
        "workflow_progress": workflow_progress,
        "workflow_progress_source": workflow_progress_source,
        "phase_advanced": advanced,
    }


def _reward_ledger(info: Mapping[str, Any], reward: Any) -> dict[str, Any]:
    components = info.get("reward_components")
    if not isinstance(components, list):
        return {
            "reward_components": [],
            "reward_components_present": False,
            "reward_components_sum": None,
            "reward_components_match": None,
        }
    values = []
    valid_components = bool(components)
    for item in components:
        if not isinstance(item, Mapping) or "value" not in item:
            valid_components = False
            continue
        try:
            value = item["value"]
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError
            values.append(float(value))
        except (TypeError, ValueError, OverflowError):
            valid_components = False
    total = (
        sum(values)
        if valid_components and values and all(value == value for value in values)
        else None
    )
    match = None
    if not valid_components:
        match = False
    elif total is not None:
        try:
            match = abs(total - float(reward)) <= 1e-8
        except (TypeError, ValueError):
            match = False
    return {
        "reward_components": _json_copy(components),
        "reward_components_present": True,
        "reward_components_sum": total,
        "reward_components_match": match,
    }


def parse_indices(spec: str) -> list[int]:
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("indices must be a comma-separated list or ranges")
    result: list[int] = []
    seen: set[int] = set()
    for chunk in spec.split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            pieces = token.split("-")
            if len(pieces) != 2 or not all(piece.strip().isdigit() for piece in pieces):
                raise ValueError(f"invalid index range: {token!r}")
            start, end = (int(piece.strip()) for piece in pieces)
            if end < start:
                raise ValueError(f"descending index range: {token!r}")
            values = range(start, end + 1)
        elif token.isdigit():
            values = (int(token),)
        else:
            raise ValueError(f"invalid data index: {token!r}")
        for value in values:
            if value < 0:
                raise ValueError("data indices must be non-negative")
            if value not in seen:
                seen.add(value)
                result.append(value)
    if not result:
        raise ValueError("indices selected no values")
    return result


class EvalRunner:
    def __init__(
        self,
        env: AgentMemoryEnvClient,
        model: OpenAIChatClient,
        *,
        indices: Sequence[int],
        max_policy_turns: int,
        output_dir: Path,
    ) -> None:
        if max_policy_turns < 1:
            raise ValueError("max_policy_turns must be positive")
        self.env = env
        self.model = model
        self.indices = [int(index) for index in indices]
        self.max_policy_turns = int(max_policy_turns)
        self.output_dir = output_dir

    def run(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        episodes: list[dict[str, Any]] = []
        started_at = time.time()
        try:
            for data_idx in self.indices:
                episodes.append(self.run_episode(data_idx))
        finally:
            try:
                self.env.close()
            except Exception as exc:  # cleanup failure must remain visible
                (self.output_dir / "close_error.txt").write_text(
                    f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
                )
        summary = summarize_paper_surface(episodes, self.env.metadata)
        manifest = {
            "schema_version": EVAL_SCHEMA,
            "started_unix": started_at,
            "finished_unix": time.time(),
            "environment": {
                "url": self.env.base_url,
                "metadata": _json_copy(self.env.metadata),
                "system_prompt_source": self.env.system_prompt_source,
            },
            "model": {
                "url": self.model.model_url,
                "model": self.model.model,
                "temperature": self.model.temperature,
                "max_tokens": self.model.max_tokens,
                "enable_thinking": self.model.enable_thinking,
            },
            "indices": list(self.indices),
            "max_policy_turns": self.max_policy_turns,
            "prompt_history_policy": "latest_observation_only",
            "raw_prior_messages_visible": False,
            "prompt_token_ids_exact_contract": "server_tokenize_only",
            "response_token_ids_exact_contract": "explicit_response_ids_only",
            "episodes": episodes,
            "summary": summary,
        }
        write_json(self.output_dir / "manifest.json", manifest)
        return manifest

    def run_episode(self, data_idx: int) -> dict[str, Any]:
        reset_result = self.env.reset(data_idx)
        initial_info = _json_copy(self.env.info.get("env_info", {}))
        observation = str(self.env.info.get("observation", ""))
        steps: list[dict[str, Any]] = []
        episode_return = 0.0
        final_done = False
        final_success = False
        for turn in range(1, self.max_policy_turns + 1):
            before_info = _json_copy(self.env.info.get("env_info", {}))
            messages = build_latest_observation_messages(
                self.env.system_prompt, observation
            )
            prompt_ids, tokenize_json, tokenize_url = self.model.tokenize(messages)
            model_json = self.model.complete(messages)
            model_text = _completion_text(model_json)
            response_ids = _response_token_ids(model_json)
            step_result = self.env.step(model_text)
            after_info = _json_copy(self.env.info.get("env_info", {}))
            if not isinstance(after_info, Mapping):
                raise EvalError("environment step info must be a JSON object")
            raw_reward = step_result.get("reward")
            if (
                isinstance(raw_reward, bool)
                or not isinstance(raw_reward, (int, float))
                or not math.isfinite(float(raw_reward))
            ):
                raise EvalError("environment reward must be finite and numeric")
            if type(step_result.get("done")) is not bool:
                raise EvalError("environment done must be a boolean")
            reward = float(raw_reward)
            done = step_result["done"]
            episode_return += reward
            final_done = done
            # The environment owns success semantics.  A missing or
            # non-boolean field is an evidence failure, never an implicit
            # unsuccessful episode.
            if "episode_success" not in after_info:
                raise EvalError(
                    "environment step is missing authoritative episode_success"
                )
            if type(after_info["episode_success"]) is not bool:
                raise EvalError(
                    "environment episode_success must be a boolean"
                )
            final_success = after_info["episode_success"]
            if final_success and not done:
                raise EvalError(
                    "environment episode_success=True requires done=True"
                )
            ledger = _reward_ledger(after_info, reward)
            step = {
                "turn": turn,
                "request_messages": _json_copy(messages),
                "model_request": {
                    "model": self.model.model,
                    "messages": _json_copy(messages),
                    "temperature": self.model.temperature,
                    "max_tokens": self.model.max_tokens,
                    "n": 1,
                    "stream": False,
                    "chat_template_kwargs": self.model._chat_template_kwargs(),
                },
                "prompt_history_policy": "latest_observation_only",
                "raw_prior_messages_visible": False,
                "prompt_tokenize_url": tokenize_url,
                "prompt_tokenize_request": {
                    "model": self.model.model,
                    "messages": _json_copy(messages),
                    "add_generation_prompt": True,
                    "chat_template_kwargs": self.model._chat_template_kwargs(),
                },
                "prompt_tokenize_response": _json_copy(tokenize_json),
                "prompt_token_ids": list(prompt_ids),
                "prompt_token_ids_hash": token_ids_hash(prompt_ids),
                "prompt_token_ids_exact": True,
                "raw_model_response": _json_copy(model_json),
                "model_text": model_text,
                "response_token_ids": (
                    list(response_ids) if response_ids is not None else None
                ),
                "response_token_ids_hash": (
                    token_ids_hash(response_ids) if response_ids is not None else None
                ),
                "response_token_ids_exact": response_ids is not None,
                "action_submitted": self.env.last_submitted_action,
                "environment_step_request": {
                    "id": self.env.env_id,
                    "action": self.env.last_submitted_action,
                },
                "env_info_before": before_info,
                "env_response": _json_copy(step_result),
                "env_info_after": after_info,
                "reward": reward,
                **ledger,
                "phase_progress": _phase_progress(before_info, after_info),
                "done": done,
                "episode_success": final_success,
            }
            steps.append(step)
            observation = str(step_result.get("observation", ""))
            if done:
                break
        timed_out = not final_done
        episode = {
            "data_idx": int(data_idx),
            "reset_response": _json_copy(reset_result),
            "initial_env_info": initial_info,
            "steps": steps,
            "episode_return": episode_return,
            "done": final_done,
            "episode_success": final_success,
            "timed_out": timed_out,
            "final_phase_progress": (
                steps[-1]["phase_progress"] if steps else _phase_progress(initial_info, initial_info)
            ),
        }
        write_json(
            self.output_dir / f"episode_{int(data_idx):04d}.json",
            episode,
        )
        return episode


def summarize_episodes(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    returns = [float(item.get("episode_return", 0.0)) for item in episodes]
    successes = []
    for index, item in enumerate(episodes):
        if "episode_success" not in item:
            raise EvalError(
                "episode summary is missing authoritative episode_success at "
                f"index {index}"
            )
        if type(item["episode_success"]) is not bool:
            raise EvalError(
                "episode summary episode_success must be boolean at "
                f"index {index}"
            )
        successes.append(item["episode_success"])
    progress_keys: list[str] = []
    known_phase_counts: set[int] = set()
    for episode in episodes:
        progress = episode.get("final_phase_progress")
        if not isinstance(progress, Mapping):
            key = "unknown"
        else:
            phase_index = progress.get("phase_index_after")
            phase_count = progress.get("phase_count")
            try:
                phase_index = int(phase_index)
                phase_count = int(phase_count)
            except (TypeError, ValueError):
                key = "unknown"
            else:
                key = (
                    f"{phase_index}/{phase_count}"
                    if phase_index >= 0
                    and phase_count > 0
                    and phase_index <= phase_count
                    else "unknown"
                )
                if key != "unknown":
                    known_phase_counts.add(phase_count)
        progress_keys.append(key)
    progress_counts: dict[str, int] = {}
    # When every episode shares one authoritative phase count, include empty
    # bins as zero so a diagnostic always answers how many trajectories reached
    # every i/N milestone, not only the milestones that happened to appear.
    if len(known_phase_counts) == 1:
        phase_count = next(iter(known_phase_counts))
        progress_counts.update(
            {f"{phase_index}/{phase_count}": 0 for phase_index in range(phase_count + 1)}
        )
    for key in progress_keys:
        progress_counts[key] = progress_counts.get(key, 0) + 1
    return {
        "episode_count": len(episodes),
        "mean_return": (sum(returns) / len(returns)) if returns else 0.0,
        "success_count": sum(successes),
        "success_rate": (sum(successes) / len(successes)) if successes else 0.0,
        "timeout_count": sum(bool(item.get("timed_out", False)) for item in episodes),
        "final_phase_progress_distribution": progress_counts,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def resolve_max_policy_turns(
    metadata: Mapping[str, Any],
    explicit_limit: int | None,
) -> int:
    if explicit_limit is not None:
        if isinstance(explicit_limit, bool) or explicit_limit < 1:
            raise EvalError("--max-policy-turns must be a positive integer")
        return int(explicit_limit)
    runtime_limit = metadata.get("max_steps")
    if (
        not isinstance(runtime_limit, bool)
        and isinstance(runtime_limit, int)
        and runtime_limit > 0
    ):
        return runtime_limit
    if metadata.get("surface") == WEBSHOP_V2_SURFACE:
        return LEGACY_WEBSHOP_MAX_POLICY_TURNS
    raise EvalError(
        "environment metadata lacks a positive max_steps; pass "
        "--max-policy-turns explicitly"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-url", required=True, help="AgentMemory HTTP server base URL")
    parser.add_argument(
        "--model-url",
        required=True,
        help="vLLM OpenAI-compatible base URL (with or without /v1)",
    )
    parser.add_argument("--model", required=True, help="model id sent to vLLM")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help=(
            "enable native model thinking in both /tokenize and "
            "/chat/completions; disabled by default to match AMG rollout"
        ),
    )
    parser.add_argument(
        "--indices",
        default="0",
        help=(
            "zero-based environment dataset positions, e.g. 0,2-4; Travel "
            "Planner uses positions 0..269 and records source ids 1..270 "
            "separately in evidence"
        ),
    )
    parser.add_argument(
        "--max-policy-turns",
        type=int,
        default=None,
        help=(
            "per-episode policy-turn cap; defaults to the environment's "
            "attested max_steps (56 for legacy WebShop)"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--api-key",
        default=None,
        help="optional bearer token; defaults to no Authorization header",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        indices = parse_indices(args.indices)
        env_transport = JsonHttp(timeout=args.timeout)
        model_transport = JsonHttp(
            timeout=args.timeout, api_key=args.api_key
        )
        env = AgentMemoryEnvClient(args.env_url, env_transport)
        model = OpenAIChatClient(
            args.model_url,
            args.model,
            model_transport,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            enable_thinking=args.enable_thinking,
        )
        runner = EvalRunner(
            env,
            model,
            indices=indices,
            max_policy_turns=resolve_max_policy_turns(
                env.metadata,
                args.max_policy_turns,
            ),
            output_dir=args.output_dir,
        )
        manifest = runner.run()
        print(json.dumps(manifest["summary"], ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"agentmemory eval failed closed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
