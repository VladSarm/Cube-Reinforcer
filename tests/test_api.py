import json
import threading
import time
import unittest
from urllib import error, request

from rubik_sim.actions import solved_state
from rubik_sim.engine import RubikEngine
from rubik_sim.server import RubikHTTPServer
from rubik_sim.state_codec import state_to_json_one_hot


def http_json(method: str, url: str, payload: dict | None = None):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = request.Request(url=url, method=method, data=data, headers=headers)
    with request.urlopen(req, timeout=2.0) as resp:
        body = resp.read().decode("utf-8")
        return resp.status, json.loads(body)


class TestAPI(unittest.TestCase):
    def setUp(self):
        self.engine = RubikEngine()
        self.server = RubikHTTPServer(engine=self.engine, host="127.0.0.1", port=0, mode="headless")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        time.sleep(0.05)
        self.base = f"http://{self.server.host}:{self.server.port}"

    def tearDown(self):
        self.server.shutdown()
        self.thread.join(timeout=1.0)

    def test_step_applies_action_and_returns_payload(self):
        status, out = http_json("POST", f"{self.base}/step", {"action": 8})
        self.assertEqual(status, 200)
        self.assertIn("state", out)
        self.assertIn("solved", out)
        self.assertEqual(out["action"], 8)
        self.assertEqual(len(out["state"]), 54)
        self.assertEqual(len(out["state"][0]), 6)
        self.assertEqual(sum(out["state"][0]), 1)

    def test_state_roundtrip_set_get(self):
        colors = solved_state().tolist()
        colors[0], colors[9] = colors[9], colors[0]
        target = state_to_json_one_hot(colors)

        status, _ = http_json("POST", f"{self.base}/state", {"state": target})
        self.assertEqual(status, 200)

        status, out = http_json("GET", f"{self.base}/state")
        self.assertEqual(status, 200)
        self.assertEqual(out["state"], target)

    def test_scramble_changes_state_and_reports_actions(self):
        initial = state_to_json_one_hot(solved_state())
        status, out = http_json("POST", f"{self.base}/scramble", {"steps": 12, "seed": 7})
        self.assertEqual(status, 200)
        self.assertEqual(len(out["actions"]), 12)
        self.assertEqual(out["step_count"], 12)
        self.assertNotEqual(out["state"], initial)

    def test_invalid_action_returns_400(self):
        req = request.Request(
            url=f"{self.base}/step",
            method="POST",
            data=json.dumps({"action": 99}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with self.assertRaises(error.HTTPError) as ctx:
            request.urlopen(req, timeout=2.0)
        self.assertEqual(ctx.exception.code, 400)

    def test_step_animated_endpoint(self):
        status, out = http_json("POST", f"{self.base}/step_animated", {"action": 2, "duration_ms": 150})
        self.assertEqual(status, 200)
        self.assertEqual(out["action"], 2)
        self.assertIn("state", out)


if __name__ == "__main__":
    unittest.main()
