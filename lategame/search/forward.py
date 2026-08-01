"""Python wrapper over the persistent forward-model driver (Lever 11 / R-PREDICT).

Holds one long-lived ``node forward_driver.js`` subprocess and exchanges NDJSON
request/response lines with it. Each agent owns one ``ForwardModel`` so node startup +
dex/generator load is paid once, not per decision. Commands: ``reconstruct`` (build a
full battle from a determinization spec -> serialized state + observable digest) and
``step`` (fork a serialized state, apply a hypothetical (p1, p2) choice -> per-side delta
+ new request, for feeding a poke-env Battle and embedding the leaf).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

_DEFAULT_SHOWDOWN = "third_party/pokemon-showdown"
_DRIVER = Path(__file__).with_name("forward_driver.js")


class ForwardModel:
    """A persistent Showdown forward model: reconstruct a state, then fork/step it."""

    def __init__(
        self,
        showdown_dir: str | Path = _DEFAULT_SHOWDOWN,
        node: str = "node",
    ) -> None:
        if not _DRIVER.exists():
            raise RuntimeError(f"forward driver missing: {_DRIVER}")
        if not (Path(showdown_dir) / "dist").is_dir():
            raise RuntimeError(
                f"'{showdown_dir}/dist' not found -- build the vendored simulator first "
                "(run `bash scripts/setup_server.sh`)."
            )
        self._proc = subprocess.Popen(
            [node, str(_DRIVER), str(showdown_dir)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def _rpc(self, msg: dict[str, Any]) -> dict[str, Any]:
        if self._proc.poll() is not None or self._proc.stdin is None or self._proc.stdout is None:
            err = self._proc.stderr.read() if self._proc.stderr else ""
            raise RuntimeError(f"forward driver died: {err[:500]}")
        self._proc.stdin.write(json.dumps(msg) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            err = self._proc.stderr.read() if self._proc.stderr else ""
            raise RuntimeError(f"forward driver returned no output: {err[:500]}")
        res: dict[str, Any] = json.loads(line)
        if "error" in res:
            raise RuntimeError(f"forward driver error: {res['error'][:500]}")
        return res

    def ping(self) -> bool:
        return bool(self._rpc({"cmd": "ping"}).get("pong"))

    def reconstruct(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Build a battle from ``spec``; returns ``{"state", "digest"}``."""
        return self._rpc({"cmd": "reconstruct", "spec": spec})

    def step(
        self, state: Any, p1: str | None, p2: str | None
    ) -> dict[str, Any]:
        """Fork ``state``, apply (p1, p2) choices; returns per-side delta + request + state."""
        return self._rpc({"cmd": "step", "state": state, "p1": p1, "p2": p2})

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                if self._proc.stdin is not None:
                    self._proc.stdin.close()
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self._proc.kill()

    def __enter__(self) -> ForwardModel:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
