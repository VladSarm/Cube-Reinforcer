"""CLI entrypoint for Rubik simulator."""

from __future__ import annotations

import argparse
import json

from .engine import RubikEngine
from .server import RubikHTTPServer


def _load_state(state_json: str | None, state_file: str | None):
    if state_json and state_file:
        raise ValueError("Use only one of --state-json or --state-file")
    if state_json:
        return json.loads(state_json)
    if state_file:
        with open(state_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rubik 3x3 simulator")
    sub = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--host", default="127.0.0.1")
    common.add_argument("--port", type=int, default=8000)
    common.add_argument("--cube-size", type=int, default=3, choices=[3])
    common.add_argument("--state-json", type=str, default=None)
    common.add_argument("--state-file", type=str, default=None)

    headless = sub.add_parser("headless", parents=[common], help="Run headless HTTP simulator")
    headless.add_argument("--scramble-steps", type=int, default=20)

    gui = sub.add_parser("gui", parents=[common], help="Run pygame GUI with HTTP server")
    gui.add_argument("--scramble-steps", type=int, default=20)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    initial_state = _load_state(args.state_json, args.state_file)
    engine = RubikEngine(cube_size=args.cube_size, initial_state=initial_state)

    if args.mode == "headless":
        server = RubikHTTPServer(engine=engine, host=args.host, port=args.port, mode="headless")
        if args.scramble_steps > 0 and initial_state is None:
            engine.scramble(args.scramble_steps)
        print(f"Rubik headless server listening on http://{server.host}:{server.port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.shutdown()
        return

    if args.mode == "gui":
        from .gui import RubikGUI

        app = RubikGUI(engine=engine, host=args.host, port=args.port, scramble_steps=args.scramble_steps)
        app.run()
        return

    parser.error(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
