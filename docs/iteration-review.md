# Iteration Review Protocol

This note is the durable review protocol for Anila iterations. Keep it stable and update it only when the project changes how work should be reviewed. Do not use it as a one-off release report.

## Posture

Anila is a compact native PyTorch language-model training library. A good iteration makes the tokenizer-to-inference path clearer, safer, or easier to modify without hiding the native model, objective, data, checkpoint, and runtime contracts behind a heavy framework.

Prefer changes that increase locality and leverage:

- Put validation in `src/anila/config.py`.
- Keep strict UTF-8 handling in `src/anila/data.py` and `src/anila/tokenization.py`.
- Keep generation semantics in `src/anila/model.py`.
- Route checkpoint artifact reads through `load_checkpoint_payload`.
- Keep runtime state, evaluation RNG isolation, resume state, and training-loader order in `src/anila/training.py`.
- Add optional ecosystem or scaled-runtime support as adapters rather than changing the native checkpoint or objective path.

## Review Questions

Before calling an iteration ready, answer these questions from current files and command output:

1. Does the change help a beginner move through tokenizer training, data preparation, pretraining, post-training, preference/RL training, evaluation, export, or inference?
2. Does each changed file trace directly to the objective, runtime, or interface being improved?
3. Did the change deepen a real module, or did it add a shallow wrapper whose interface is as complex as its implementation?
4. Did any new public claim land in README, docs, status, or changelog without a command, test, or artifact proving it?
5. Did the change alter checkpoint, generation, data loading, or training runtime behavior, and if so, is there focused regression coverage?
6. Can the full verification gate run from one command: `bash scripts/verify.sh`?

## Required Evidence

Every release-minded iteration should leave these surfaces synchronized:

- Version fields in `pyproject.toml`, `uv.lock`, `CHANGELOG.md`, and release/status docs.
- Beginner-facing command paths in `README.md`, `docs/full-flow-quickstart.md`, and `docs/development.md`.
- Data and checkpoint contracts in `docs/data-contracts.md`, `docs/architecture.md`, and tests.
- CI and local verification through `scripts/verify.sh`.

Run:

```bash
bash scripts/verify.sh
```

For broader onboarding or release changes, also run `bash scripts/quickstart-smoke.sh`. For packaging changes, build and inspect the source distribution and wheel, then run the installed console script in an isolated environment.

## Architecture Bar

Use the deep-module vocabulary consistently:

- A module is worth keeping when deleting it would move complexity into several callers instead of making complexity disappear.
- The interface is the test surface. If tests need to reach around the interface to prove behavior, the module is probably shaped wrong.
- A seam is real when behavior varies behind it. One adapter is usually a hypothetical seam; two adapters make the variation concrete.
- Locality matters more than line count. Large modules are acceptable when the project contract says they own a coherent runtime concern and the tests exercise that concern through the same interface as callers.

## Stop Conditions

Do not commit or push an iteration when any of these are true:

- `bash scripts/verify.sh` fails.
- A public doc describes a command that has not been run or covered by an equivalent test.
- A checkpoint read path bypasses `load_checkpoint_payload`.
- Evaluation can perturb training randomness.
- Batched generation can keep emitting after `eos_id`.
- New docs claim distributed training, Hugging Face interoperability, serving compatibility, or production model quality without implemented and tested support.
- Generated artifacts under `runs/`, `dist/`, `.local/`, caches, checkpoints, or logs would be included in the commit.
