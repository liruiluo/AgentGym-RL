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
