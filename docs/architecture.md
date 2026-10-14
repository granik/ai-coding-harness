# Architecture

## Components

```mermaid
flowchart LR
    U[CLI user] --> C[CodingHarness controller]
    C --> M[Local Ollama or ScriptedModel]
    M -->|JSON tool call| C
    C --> D[Dispatcher]
    D --> W[DisposableWorkspace]
    D --> R[CheckRunner]
    R --> X[Docker sandbox]
    X --> P[Disposable repo]
    X --> A[Read-only acceptance checks]
    W --> P
    C -->|untrusted observation| M
    C --> F[Final verifier]
    F --> R
    C --> O[trace.jsonl and report.json]
```

`CodingHarness` owns the bounded loop. `Dispatcher` validates exact action and
argument shapes before calling a tool. `DisposableWorkspace` owns path and
writable-scope checks. `CheckRunner` accepts only trusted commands from TOML;
the model selects a check name rather than submitting shell text.

## One run

```mermaid
sequenceDiagram
    actor User
    participant CLI
    participant Controller
    participant Model
    participant Dispatcher
    participant Sandbox

    User->>CLI: task + trusted config
    CLI->>CLI: verify commit and copy repository
    CLI->>Controller: start bounded run
    loop Up to max_actions
        Controller->>Model: task, schema, recent observations
        Model-->>Controller: one JSON object
        Controller->>Dispatcher: validated action candidate
        Dispatcher-->>Controller: ok, error, or denied
        Controller->>Model: untrusted observation
        opt Exact edit cannot be applied
            Controller->>Model: fixed-path replace_file recovery schema
        end
        opt All model-visible checks passed
            Controller->>Sandbox: run every independent check
            Sandbox-->>Controller: exit codes and bounded output
        end
    end
    Controller->>Sandbox: run every configured final check
    Sandbox-->>Controller: exit codes and bounded output
    Controller-->>CLI: changed files, diff, counters, checks
    CLI-->>User: terminal result + JSON report
```

## Trust boundaries

Trusted inputs are the harness source, TOML configuration, external acceptance
directory, and the human-selected source commit. Model replies, repository
content, and command output are untrusted. The controller never converts text
from an untrusted source into new permissions or an executable command.

The dispatcher enforces recovery phases independently of JSON Schema. After an
exact edit failure, only a complete replacement of the same writable file is
authorized. When all model-visible checks pass after the latest mutation, the
controller immediately runs the full verification set and can finish without
spending another model turn on a summary.

Docker is the intended execution boundary. The process backend retains timeout,
resource, process-group, and output controls but is only a compatibility mode
for reviewed local repositories.

