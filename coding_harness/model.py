"""Local Ollama adapter and deterministic scripted model."""

from __future__ import annotations

import http.client
import json
from pathlib import Path
import socket
import urllib.error
import urllib.parse
import urllib.request

from .config import ModelConfig


class ModelError(RuntimeError):
    pass


class OllamaModel:
    def __init__(self, config: ModelConfig) -> None:
        parsed = urllib.parse.urlsplit(config.endpoint)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ModelError("Ollama endpoint must be local HTTP without credentials")
        self.config = config

    def generate(self, messages: list[dict[str, str]], schema: dict) -> str:
        payload = {
            "model": self.config.name,
            "messages": messages,
            "stream": False,
            "think": False,
            "format": schema,
            "options": {
                "temperature": 0,
                "num_ctx": self.config.context_tokens,
                "num_predict": self.config.predict_tokens,
                "repeat_last_n": 64,
                "repeat_penalty": 1.05,
            },
        }
        request = urllib.request.Request(
            self.config.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.timeout_seconds
            ) as response:
                raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise ModelError("Ollama response exceeded 2,000,000 bytes")
            body = json.loads(raw.decode("utf-8"))
            content = body["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("message.content is not text")
            return content
        except urllib.error.HTTPError as error:
            detail = error.read(4_000).decode("utf-8", errors="replace")
            raise ModelError(
                f"Ollama returned HTTP {error.code}: {detail or error.reason}"
            ) from error
        except urllib.error.URLError as error:
            raise ModelError(f"Cannot reach Ollama: {error.reason}") from error
        except (TimeoutError, socket.timeout, http.client.HTTPException) as error:
            raise ModelError(f"Ollama request failed: {error}") from error
        except (KeyError, TypeError, json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ModelError(f"Invalid Ollama response: {error}") from error


class ScriptedModel:
    """Repeatable model used by tests and recorded demonstrations."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict[str, str]]] = []
        self.schemas: list[dict] = []

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "ScriptedModel":
        replies = []
        for number, line in enumerate(
            Path(path).read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ModelError(f"Invalid scripted JSON on line {number}: {error}") from error
            replies.append(json.dumps(value, ensure_ascii=False))
        return cls(replies)

    def generate(self, messages: list[dict[str, str]], schema: dict) -> str:
        self.calls.append([dict(message) for message in messages])
        self.schemas.append(schema)
        if not self.replies:
            raise ModelError("Scripted model has no reply left")
        return self.replies.pop(0)

