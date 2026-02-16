"""HTTP client for Rubik simulator server."""

from __future__ import annotations

import json
from urllib import request

import numpy as np


class RubikAPIClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8000, timeout: float = 10.0):
        self.base = f"http://{host}:{port}"
        self.timeout = timeout

    def _call(self, method: str, path: str, payload: dict | None = None) -> dict:
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = request.Request(url=f"{self.base}{path}", method=method, data=data, headers=headers)
        with request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def health(self) -> dict:
        return self._call("GET", "/health")

    def get_state(self) -> np.ndarray:
        out = self._call("GET", "/state")
        return np.asarray(out["state"], dtype=np.float64)

    def reset(self, state: np.ndarray | list | None = None) -> dict:
        payload = {} if state is None else {"state": np.asarray(state).astype(int).tolist()}
        return self._call("POST", "/reset", payload)

    def scramble(self, steps: int, seed: int | None = None) -> dict:
        payload = {"steps": int(steps), "seed": seed}
        return self._call("POST", "/scramble", payload)

    def step(self, action: int) -> dict:
        return self._call("POST", "/step", {"action": int(action)})

    def step_animated(self, action: int, duration_ms: int | None = None) -> dict:
        payload = {"action": int(action)}
        if duration_ms is not None:
            payload["duration_ms"] = int(duration_ms)
        return self._call("POST", "/step_animated", payload)

    def solved(self) -> bool:
        out = self._call("GET", "/solved")
        return bool(out["solved"])
