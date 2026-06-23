# Contributor Contract

## Scope

Anila is a compact native PyTorch language-model training library. Keep the core readable and dependency-light. Add ecosystem or scaled-runtime support as optional adapters rather than obscuring the native model and objective implementations.

## Project Map

- `src/anila/`: package code for model, data, objectives, runtime, artifacts, evaluation, and CLI.
- `tests/`: unit and tiny end-to-end regression coverage; `tests/test_training_integration.py` owns end-to-end runtime/checkpoint contract verification.
- `configs/quickstart/` and `examples/`: runnable local recipes and their tiny datasets.
- `docs/`, `README.md`, and `CHANGELOG.md`: architecture, operating instructions, and released/unreleased behavior.

## Boundaries

- `src/anila/config.py` owns validation. Invalid explicit input must fail rather than be replaced by a default.
- `src/anila/data.py` and `src/anila/tokenization.py` treat training input as strict UTF-8. Do not silently drop invalid bytes.
- `src/anila/model.py` owns generation semantics. In batched generation, a sequence that emits `eos_id` must remain terminal.
- `src/anila/checkpoint.py` owns external checkpoint deserialization. Route library checkpoint reads through `load_checkpoint_payload`; do not add direct `torch.load` calls for user-supplied artifacts.
- `src/anila/training.py` owns runtime state. Evaluation must not perturb training randomness; new resume checkpoints must preserve built-in training-loader order; older checkpoints without runtime or data state must remain loadable.

## Change Rules

- Keep edits scoped to the objective, runtime, or interface being changed.
- Add a regression test for every fixed behavioral defect.
- Do not commit files under `runs/`, `dist/`, `.local/`, virtual environments, caches, checkpoints, or logs.
- Do not claim distributed training, Hugging Face interoperability, or serving compatibility until implemented and tested.

## Verification

Run before committing:

```bash
bash scripts/verify.sh
```

That wrapper runs `uv lock --check`, `uv run ruff check .`, and `uv run pytest -q` in the same order as CI.

For changes affecting checkpoints, generation, data loading, or training runtime, include the relevant focused regression tests in the change review.
