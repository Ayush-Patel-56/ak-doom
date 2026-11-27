"""Minimal client for the hackathon's OpenAI-compatible LLM API (Creo), for our TESTING tools only.

Never imported by submission/: evaluation has no network, and the agent's decisions must come from our
own trained network. This file only powers offline review tools (scripts/replay_review.py,
scripts/mock_review.py).

Settings (environment variables win over the file):
    CREO_API_KEY    secret key, never committed, never printed
    CREO_BASE_URL   e.g. https://<host>/v1 (the part before /chat/completions)
    CREO_MODEL      default model name (list them with `models`)
File: ~/.creo.env with KEY=VALUE lines (outside the repo). Standard library only: no new dependency.

    uv run python scripts/creo.py models          # which models the account can use
    uv run python scripts/creo.py ping            # one tiny request: checks key, URL and model
    uv run python scripts/creo.py usage           # tokens used so far (runs/llm/usage.jsonl)

Dry run (no key needed, nothing sent): set CREO_DRY_RUN=1 or pass dry_run=True; each request body is
written to runs/llm/dry-run/ so the prompts can be checked before spending credits.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LLM_DIR = REPO / "runs" / "llm"
USAGE_LOG = LLM_DIR / "usage.jsonl"
DRY_DIR = LLM_DIR / "dry-run"
ENV_FILE = Path.home() / ".creo.env"
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


class CreoError(RuntimeError):
    pass


def load_settings(env_file: Path = ENV_FILE) -> dict[str, str]:
    """CREO_* settings: ~/.creo.env first, then the environment overrides it."""
    out: dict[str, str] = {}
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    for k in ("CREO_API_KEY", "CREO_BASE_URL", "CREO_MODEL", "CREO_DRY_RUN"):
        if os.environ.get(k):
            out[k] = os.environ[k]
    return out


class Creo:
    def __init__(self, model: str | None = None, dry_run: bool | None = None, timeout: float = 180.0,
                 max_retries: int = 4, settings: dict | None = None):
        s = load_settings() if settings is None else settings
        self.key = s.get("CREO_API_KEY", "")
        self.base = s.get("CREO_BASE_URL", "").rstrip("/")
        self.model = model or s.get("CREO_MODEL", "")
        env_dry = s.get("CREO_DRY_RUN", "") not in ("", "0", "false", "False")
        self.dry_run = env_dry if dry_run is None else dry_run
        if not self.dry_run and not (self.key and self.base):
            raise CreoError("set CREO_API_KEY and CREO_BASE_URL (environment or ~/.creo.env), or use --dry-run")
        self.timeout, self.max_retries = timeout, max_retries
        self._dry_n = 0

    # ------------------------------------------------------------------ http
    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        url = f"{self.base}{path}"
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                   "User-Agent": "ak-doom-review-tools"}
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(url, data=data, method=method, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:500]
                if self.key:
                    detail = detail.replace(self.key, "***")
                if e.code in RETRY_STATUS and attempt < self.max_retries:
                    wait = float(e.headers.get("retry-after") or 2 ** attempt)
                    time.sleep(min(wait, 60.0))
                    continue
                raise CreoError(f"HTTP {e.code} from {path}: {detail}") from None
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise CreoError(f"cannot reach {self.base}: {e}") from None
        raise CreoError("unreachable")

    # ------------------------------------------------------------------ api
    def models(self) -> list[str]:
        if self.dry_run:
            return []
        data = self._request("GET", "/models")
        return sorted(m.get("id", "?") for m in data.get("data", []))

    def chat(self, messages: list[dict], *, max_tokens: int = 1500, temperature: float = 0.2,
             json_mode: bool = False, tag: str = "chat") -> str:
        """One chat completion; returns the reply text (in dry run: a placeholder)."""
        if not self.model and not self.dry_run:
            raise CreoError("no model: set CREO_MODEL or pass --model (see `creo.py models`)")
        body = {"model": self.model or "<model>", "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.dry_run:
            DRY_DIR.mkdir(parents=True, exist_ok=True)
            self._dry_n += 1
            path = DRY_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{tag}-{self._dry_n:03d}.json"
            path.write_text(json.dumps(body, indent=1, ensure_ascii=False), encoding="utf-8")
            return "{}" if json_mode else f"[dry run: request saved to {path.relative_to(REPO)}]"
        try:
            data = self._request("POST", "/chat/completions", body)
        except CreoError as e:
            if json_mode and "HTTP 400" in str(e):  # some servers reject response_format: ask in the prompt only
                body.pop("response_format")
                data = self._request("POST", "/chat/completions", body)
            else:
                raise
        self._log_usage(tag, data.get("usage") or {})
        choices = data.get("choices") or []
        if not choices:
            raise CreoError(f"no choices in reply: {str(data)[:300]}")
        return (choices[0].get("message") or {}).get("content") or ""

    def _log_usage(self, tag: str, usage: dict) -> None:
        LLM_DIR.mkdir(parents=True, exist_ok=True)
        row = {"t": round(time.time()), "tag": tag, "model": self.model,
               "prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0)}
        with USAGE_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def parse_json_reply(text: str) -> dict:
    """Best-effort JSON from a model reply (handles ```json fences and text around the object)."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        start, end = t.find("{"), t.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(t[start:end + 1])
            except json.JSONDecodeError:
                pass
    return {"_unparsed": text[:2000]}


def usage_summary() -> dict:
    rows = []
    if USAGE_LOG.is_file():
        rows = [json.loads(l) for l in USAGE_LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_tag: dict[str, list[int]] = {}
    for r in rows:
        acc = by_tag.setdefault(r["tag"], [0, 0, 0])
        acc[0] += 1
        acc[1] += r["prompt_tokens"]
        acc[2] += r["completion_tokens"]
    return {"requests": len(rows), "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
            "completion_tokens": sum(r["completion_tokens"] for r in rows),
            "by_tag": {k: dict(requests=v[0], prompt_tokens=v[1], completion_tokens=v[2]) for k, v in by_tag.items()}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["models", "ping", "usage", "settings"])
    ap.add_argument("--model")
    a = ap.parse_args(argv)
    if a.command == "usage":
        print(json.dumps(usage_summary(), indent=1))
        return 0
    if a.command == "settings":
        s = load_settings()
        print(json.dumps({"CREO_BASE_URL": s.get("CREO_BASE_URL", "(not set)"), "CREO_MODEL": s.get("CREO_MODEL", "(not set)"),
                          "CREO_API_KEY": "set" if s.get("CREO_API_KEY") else "(not set)",
                          "env file": str(ENV_FILE) + (" (found)" if ENV_FILE.is_file() else " (missing)")}, indent=1))
        return 0
    client = Creo(model=a.model)
    if a.command == "models":
        for m in client.models():
            print(m)
        return 0
    reply = client.chat([{"role": "user", "content": "Reply with exactly: pong"}], max_tokens=10, temperature=0, tag="ping")
    print(f"model {client.model!r} replied: {reply.strip()!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
