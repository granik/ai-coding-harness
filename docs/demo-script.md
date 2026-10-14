# Four-minute demo outline

1. Show `harness.example.toml`: fixed repository commit, one writable source
   file, model-visible regression check, and external acceptance check.
2. Run `./scripts/prepare_demo.sh` and show that the starting acceptance check
   fails because `services.add_batch` is missing.
3. Start the harness with Ollama, or clearly state that
   `demo/scripted_repair.jsonl` is being used for a repeatable controller demo.
4. Point out each model turn, tool name, and file path in terminal progress.
5. Show the final `services.py` diff, 23 passing regression tests, passing
   acceptance and AST quality checks, and `verification_passed: true`.
6. Open `report.json`, show denied/retry counters, and explain that a model claim
   cannot replace the final checks.
7. Mention the remaining limitation: local process mode is not a host sandbox;
   Docker is required for unknown code.

