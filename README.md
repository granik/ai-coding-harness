# Stage 1 coding harness

This project is a small end-to-end coding harness for a local Ollama model. It
copies a target repository, gives the model bounded file and check tools, runs
independent final verification, and prints the changed files, unified diff, and
all check results.

The demo target is the public
[Cosmic Python example application](https://github.com/cosmicpython/code) at
commit `17215904806967ee80a71ce0c6f771932382ed0f`. The feature adds service-layer
batch creation and deallocation across the service, repository, and domain
model interfaces. Only `services.py` is writable; the acceptance check stays in
`demo/cosmic_acceptance`, outside every disposable repository copy.

## Quick start

Requirements: Python 3.11+, Git, local Ollama, and Docker for contained check
execution.

```bash
cd ~/Downloads/POC
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[demo]'
./scripts/prepare_demo.sh
ollama pull qwen2.5-coder:7b
docker build -t poc-harness-python:3.11 -f docker/Dockerfile .
python -m coding_harness check-config --config harness.example.toml
python -m coding_harness run --config harness.example.toml
```

Start Ollama in another terminal before `ollama pull`, and keep it running:

```bash
ollama serve
```

The model server must be available at `http://127.0.0.1:11434/api/chat`. No API
key or secret is used. If Docker is unavailable, a reviewed local repository
can use the less isolated process backend:

```bash
python -m coding_harness run --config harness.local.toml
```

For a repeatable demonstration without Ollama, use the recorded JSON replies:

```bash
python -m coding_harness run \
  --config harness.local.toml \
  --scripted demo/scripted_repair.jsonl
```

`harness.local.toml` explicitly enables the resource-limited local process
backend for this trusted demo. It does not provide Docker's filesystem and
network isolation and should not be used for an unknown repository.

Override the task or model without editing the config:

```bash
python -m coding_harness run \
  --config harness.example.toml \
  --model qwen2.5-coder:14b \
  --task "Your small repository task"
```

## Fix a bug in a Git repository

The harness never edits the original checkout. It verifies the exact starting
commit, creates a disposable copy, lets the model work there, and saves a diff.

1. Start from a clean commit and create a branch:

   ```bash
   git -C /absolute/path/to/repository status --short
   git -C /absolute/path/to/repository switch -c harness-fix
   git -C /absolute/path/to/repository rev-parse HEAD
   ```

   The first command must print nothing. Put the commit printed by the last
   command in `project.start_commit`.

2. Copy `harness.example.toml` and configure:

   - `project.source`: absolute or config-relative repository path;
   - `project.task`: the small bug-fix request;
   - `project.writable`: only files the model may change;
   - `project.acceptance_dir`: a directory outside the repository;
   - `[[checks]]`: trusted test commands written as argument arrays.

   Create an acceptance check that fails on the starting commit. Keep it
   outside `project.source`, so the model cannot edit it.

3. Validate the config and run the repair:

   ```bash
   python -m coding_harness check-config --config my-project.toml
   python -m coding_harness run --config my-project.toml
   ```

4. Accept only a report with `"status": "complete"`,
   `"verification_passed": true`, and `"diff_truncated": false`. Review
   the printed diff and every check result. The terminal prints the exact path
   to `report.json`.

5. Apply the verified diff to the original branch:

   ```bash
   TARGET=/absolute/path/to/repository
   REPORT=/absolute/path/to/.harness/runs/<run-id>/report.json
   python -c 'import json,sys; print(json.load(open(sys.argv[1]))["diff"], end="")' "$REPORT" > /tmp/harness.patch
   git -C "$TARGET" apply --check /tmp/harness.patch
   git -C "$TARGET" apply /tmp/harness.patch
   git -C "$TARGET" diff
   ```

6. Run the repository's normal tests once more in the original checkout.
   Review the result, then commit it manually. The harness never pushes,
   merges, deploys, or commits on your behalf.

## What the run produces

Each run creates `.harness/runs/<timestamp-id>/` containing:

- `repo/`: the disposable repository copy;
- `trace.jsonl`: progress, tool decisions, errors, counters, and checks;
- `report.json`: termination reason, changed files, diff, check output, exit
  codes, denied actions, retries, and verification status.

The same information is printed in the terminal. A model summary never
overrides the independent `verification_passed` result.

## Configuration

Copy `harness.example.toml` for another project and change:

- `project.source`, `start_commit`, task, and writable path patterns;
- `project.acceptance_dir`, which must be outside the source repository;
- configured command arrays under `[[checks]]`;
- Docker image, limits, and local Ollama model settings.

Commands are arrays and run without a shell. The model can only select checks
with `model_allowed=true`; it cannot invent an arbitrary command. Every final
run executes all configured checks, including checks hidden from the model.

## Safety boundaries

- The source is copied without `.git`, virtual environments, caches, common
  credential files, symlinks, or environment files.
- Absolute paths, `..`, hidden paths, symlink paths, and writes outside the
  configured patterns are denied in application code.
- Python edits are parsed before an atomic replacement.
- `replace_file` can atomically replace an existing writable file after exact
  substring edits become fragile; it has the same path, size, and Python syntax
  checks as other write tools.
- Tool requests use exact JSON schemas and exact argument sets.
- Successful edits and edit mismatches return bounded current file content; a
  failed final verification also returns current changed text files so a model
  can repair stale edits without spending another action on a read.
- After an exact edit mismatch, the next model schema requires a full-file
  replacement of that same path. The dispatcher enforces this phase even when
  a model ignores its schema; a successful replacement restores normal tools.
- Once every model-visible check has passed after the latest mutation, the
  controller immediately runs all independent checks. A fully verified result
  completes without spending another model turn on a summary.
- Repository and tool text is labelled as untrusted data in the system prompt
  and observations.
- Docker runs configured checks with no network, a read-only container root,
  dropped capabilities, PID/memory/CPU limits, and only the disposable repo
  plus read-only acceptance directory mounted.
- Commands have process-group cancellation, timeouts, and bounded output.
- Push, merge, deployment, arbitrary bash, and host credentials are not
  exposed.

Docker does not make generated code trustworthy. Review the diff and check
output before accepting it. The local process backend can access the host with
the current user permissions, so its use is an explicit trusted-repository
fallback.

## Tests

Core tests need no model or API key:

```bash
python -m unittest discover -s tests -v
```

They cover file tools, path escape, writable scope, syntax rejection, unknown
tools, invalid arguments, invalid model JSON, failed commands, action and retry
limits, bounded output, child-process cancellation, full-file recovery,
schema-ignoring models, early verified completion, external acceptance-file
integrity, and a complete scripted bug fix.

The selected public baseline has 23 local regression tests. The external
acceptance check fails before the feature and passes after the scripted repair;
an external AST quality check also rejects obvious unreachable generated code.

See [architecture](docs/architecture.md) and the
[demo outline](docs/demo-script.md) for the delivery diagrams and video plan.

