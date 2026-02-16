"""HTTP API server for Rubik simulator."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

from .engine import RubikEngine
from .solved_check import is_solved_orientation_invariant
from .state_codec import StateValidationError, state_to_json_one_hot


class RubikHTTPServer:
    def __init__(
        self,
        engine: RubikEngine,
        host: str = "127.0.0.1",
        port: int = 8000,
        mode: str = "headless",
        step_animator: Callable[[int, int | None], Any] | None = None,
    ):
        self.engine = engine
        self.mode = mode
        self._lock = threading.RLock()
        self.step_animator = step_animator

        handler_cls = self._build_handler()
        self.httpd = ThreadingHTTPServer((host, port), handler_cls)
        self.host, self.port = self.httpd.server_address

    def _build_handler(self):
        parent = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "RubikSim/1.0"

            def log_message(self, fmt: str, *args):
                return

            def _send_json(self, code: int, payload: dict[str, Any]):
                body = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _read_json(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                if length == 0:
                    return {}
                data = self.rfile.read(length)
                try:
                    obj = json.loads(data.decode("utf-8"))
                except json.JSONDecodeError as exc:
                    raise StateValidationError(f"Invalid JSON body: {exc}") from exc
                if not isinstance(obj, dict):
                    raise StateValidationError("JSON body must be an object")
                return obj

            def do_GET(self):
                with parent._lock:
                    if self.path == "/health":
                        self._send_json(
                            200,
                            {
                                "mode": parent.mode,
                                "cube_size": parent.engine.cube_size,
                                "ready": True,
                            },
                        )
                        return

                    if self.path == "/state":
                        self._send_json(200, parent.engine.state_payload())
                        return

                    if self.path == "/solved":
                        self._send_json(200, {"solved": parent.engine.is_solved()})
                        return

                self._send_json(404, {"error": "Not Found"})

            def do_POST(self):
                try:
                    body = self._read_json()
                    with parent._lock:
                        if self.path == "/state":
                            state = body.get("state")
                            if state is None:
                                raise StateValidationError("Missing required field: state")
                            parent.engine.set_state(state)
                            self._send_json(200, parent.engine.state_payload())
                            return

                        if self.path == "/reset":
                            state = body.get("state") if "state" in body else None
                            parent.engine.reset(state=state)
                            self._send_json(200, parent.engine.state_payload())
                            return

                        if self.path == "/scramble":
                            if "steps" not in body:
                                raise StateValidationError("Missing required field: steps")
                            steps = body["steps"]
                            seed = body.get("seed")
                            if seed is not None and not isinstance(seed, int):
                                raise StateValidationError("seed must be an integer or null")
                            state, actions = parent.engine.scramble(steps=steps, seed=seed)
                            self._send_json(
                                200,
                                {
                                    "state": state_to_json_one_hot(state),
                                    "actions": actions,
                                    "step_count": parent.engine.step_count,
                                    "scrambled": not is_solved_orientation_invariant(state),
                                },
                            )
                            return

                        if self.path == "/step":
                            if "action" not in body:
                                raise StateValidationError("Missing required field: action")
                            action = body["action"]
                            if not isinstance(action, int):
                                raise StateValidationError("action must be an integer in range 0..11")
                            state = parent.engine.step(action)
                            self._send_json(
                                200,
                                {
                                    "state": state_to_json_one_hot(state),
                                    "action": action,
                                    "solved": parent.engine.is_solved(),
                                    "step_count": parent.engine.step_count,
                                },
                            )
                            return
                        if self.path == "/step_animated":
                            if "action" not in body:
                                raise StateValidationError("Missing required field: action")
                            action = body["action"]
                            if not isinstance(action, int):
                                raise StateValidationError("action must be an integer in range 0..11")

                            duration_ms = body.get("duration_ms")
                            if duration_ms is not None and not isinstance(duration_ms, int):
                                raise StateValidationError("duration_ms must be an integer or null")

                            if parent.step_animator is not None:
                                try:
                                    state = parent.step_animator(action, duration_ms=duration_ms)
                                except ValueError as exc:
                                    raise StateValidationError(str(exc)) from exc
                            else:
                                state = parent.engine.step(action)

                            self._send_json(
                                200,
                                {
                                    "state": state_to_json_one_hot(state),
                                    "action": action,
                                    "solved": parent.engine.is_solved(),
                                    "step_count": parent.engine.step_count,
                                },
                            )
                            return

                except StateValidationError as exc:
                    self._send_json(400, {"error": str(exc)})
                    return

                self._send_json(404, {"error": "Not Found"})

        return Handler

    def serve_forever(self):
        self.httpd.serve_forever()

    def start_background(self, daemon: bool = True) -> threading.Thread:
        thread = threading.Thread(target=self.serve_forever, daemon=daemon)
        thread.start()
        return thread

    def shutdown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
