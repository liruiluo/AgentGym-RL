# Locked upstream runtime dependency

`trl-0.9.6-py3-none-any.whl` is the unmodified PyPI wheel for veRL v0.9's
official `verl[trl]` optional extra. veRL uses its
`AutoModelForCausalLMWithValueHead` fallback for model families, including
Qwen3.5, that Transformers does not register with
`AutoModelForTokenClassification`.

- Version: `trl==0.9.6` (the latest version allowed by veRL v0.9's
  `TRL_REQUIRES = ["trl<=0.9.6"]`)
- SHA-256: `4753f190c94c11488fcc46ec74b2128e53fbc61d51f0887b7204ec4dc333af4b`
- Source: PyPI (`trl-0.9.6-py3-none-any.whl`)
- License: Apache-2.0; the upstream wheel retains its `LICENSE` metadata.

The launcher adds the wheel itself to its closed `PYTHONPATH`; it does not
replace veRL's value-head implementation or introduce an AMG-specific critic.

`liger_kernel-0.8.2-py3-none-any.whl` is the unmodified PyPI wheel used by
veRL's native `model.use_liger` path. The publication runtime's older 0.6.3
installation imports `HybridCache`, which Transformers 5.5.3 no longer
exports; 0.8.2 supports both that Transformers version and the `qwen3_5`
model type.

- Version: `liger-kernel==0.8.2`
- SHA-256: `84c0a7bc9bf4d4cf8ea5ba89ff84d28686afc94215b220851d9f57dc87852741`
- Source: PyPI (`liger_kernel-0.8.2-py3-none-any.whl`)
- License: BSD-2-Clause; the upstream wheel retains its license metadata.

The launcher places this wheel after the locked TRL wheel and before AMG/veRL
source roots in the same closed `PYTHONPATH`, so the launcher and every Ray
worker resolve the identical upstream Liger implementation without changing
the publication's frozen base runtime or its source lock.
