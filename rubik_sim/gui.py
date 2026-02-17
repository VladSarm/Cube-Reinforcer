"""Pygame GUI for the 2x2 Rubik simulator."""

from __future__ import annotations

import math
import sys
import threading
import traceback

import numpy as np
import pygame

from rubik_rl.checkpoint import CheckpointManager, load_sparse_latest, load_sparse_torch_latest
from .actions import (
    ACTION_6_TO_12,
    ACTION_NAMES,
    ACTION_TABLE,
    CLOCKWISE_ANGLE_DEG,
    FACE_AXIS_LAYER,
    FACE_INDEX,
    FACE_SPECS,
    MOVE_PERMUTATIONS,
    STICKER_LABELS,
    STICKER_MODEL,
    SLOT_INDEX_TO_CELL_NAME,
)
from .engine import RubikEngine
from .server import RubikHTTPServer
from .state_codec import encode_one_hot

# Sparse policy (6 actions): used when "Eval Sparse" is on
def _sparse_state_from_engine(engine, history_snapshot) -> np.ndarray:
    from rubik_rl.sparse_state import piece_permutation, sparse_state_from_perm
    perm = piece_permutation(history_snapshot)
    return sparse_state_from_perm(perm)

COLOR_MAP = {
    0: (245, 245, 245),
    1: (220, 30, 30),
    2: (30, 160, 30),
    3: (240, 220, 40),
    4: (255, 140, 20),
    5: (30, 90, 220),
}
SHOW_INDEX_BLACK = (25, 25, 25)  # color for --show-index highlighted sticker

BG = (18, 22, 30)
LINE = (28, 32, 42)
TEXT = (220, 225, 235)
BUTTON = (52, 60, 78)
BUTTON_BORDER = (92, 110, 140)

# By key code (works with US layout)
KEY_TO_ACTION = {
    pygame.K_u: 0,
    pygame.K_j: 1,
    pygame.K_d: 2,
    pygame.K_c: 3,
    pygame.K_l: 4,
    pygame.K_k: 5,
    pygame.K_r: 6,
    pygame.K_e: 7,
    pygame.K_f: 8,
    pygame.K_g: 9,
    pygame.K_b: 10,
    pygame.K_n: 11,
}

# By scancode (physical key; works with any layout: Russian, etc.)
# SDL / USB HID usage page 0x07: A=4, B=5, ..., U=24, ...
SCANCODE_TO_ACTION = {
    24: 0,   # U -> U+/U-
    13: 1,   # J
    7: 2,    # D
    6: 3,    # C
    15: 4,   # L
    14: 5,   # K
    21: 6,   # R
    8: 7,    # E
    9: 8,    # F
    10: 9,   # G
    5: 10,   # B
    17: 11,  # N
}

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

# Unfolded cube net: 6x6 matrix; 0 = no cell, 1 = filled cell, string = sticker label (index name)
NET_MATRIX = [
    [1, "F", "E1", "F1", 0, 0],
    ["G", "E", "A1", "B1", 0, 0],
    ["E2", "A2", "A", "B", "B2", "F2"],
    ["G2", "C2", "C", "D", "D2", 1],
    [0, 0, "C1", "D1", 0, 0],
    [0, 0, "G1", 1, 0, 0],
]
LABEL_TO_INDEX = {label: i for i, label in enumerate(STICKER_LABELS)}
NET_FILLED_COLOR = (40, 40, 50)  # color for cells with value 1


def _rotation_matrix_float(axis: str, angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


class RubikGUI:
    def __init__(
        self,
        engine: RubikEngine,
        host: str = "127.0.0.1",
        port: int = 8000,
        scramble_steps: int = 20,
        show_index: int | None = None,
    ):
        self.engine = engine
        self.scramble_steps = scramble_steps
        self.show_index = show_index  # if set, sticker at this flat index (0..23) is drawn black
        self._anim_cond = threading.Condition()
        self._pending_anim_request: dict | None = None
        self._active_anim_request: dict | None = None

        self.server = RubikHTTPServer(
            engine=engine,
            host=host,
            port=port,
            mode="gui",
            step_animator=self.animate_step_blocking,
        )

        pygame.init()
        self.size = (1280, 800)
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("Rubik 2x2 Simulator")
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("monospace", 18)
        self.small_font = pygame.font.SysFont("monospace", 14)
        self.sticker_font = pygame.font.SysFont("monospace", 11)

        self.scramble_btn = pygame.Rect(40, 560, 180, 44)
        self.reset_btn = pygame.Rect(240, 560, 180, 44)
        self.eval_btn = pygame.Rect(440, 560, 200, 44)
        self.eval_sparse_btn = pygame.Rect(40, 610, 200, 44)
        self.eval_sparse_torch_btn = pygame.Rect(250, 610, 220, 44)
        self.eval_anti_repeat_checkbox = pygame.Rect(664, 570, 24, 24)

        self.yaw = -0.75
        self.pitch = 0.45
        self.camera_distance = 6.0
        self.focal = 520.0

        self.dragging = False
        self.last_mouse = (0, 0)

        self.anim_duration_ms = 220
        self._default_anim_duration_ms = 220
        self.animating = False
        self.anim_action = -1
        self.anim_start_ms = 0
        self.anim_base_state: np.ndarray | None = None

        self.eval_enabled = False
        self.eval_policy = None
        self.eval_checkpoint_dir = "checkpoints"
        self.eval_sparse_enabled = False
        self.eval_sparse_policy = None
        self.eval_sparse_checkpoint_dir = "checkpoints_sparse"
        self.eval_sparse_torch_enabled = False
        self.eval_sparse_torch_policy = None
        self.eval_sparse_torch_checkpoint_dir = "checkpoints_sparse_torch"
        self.eval_anti_repeat_enabled = False
        self.eval_action_history: list[int] = []
        self.eval_runs_solved = 0
        self.eval_runs_total = 0

    def _apply_camera(self, point: np.ndarray) -> np.ndarray:
        rot_y = _rotation_matrix_float("y", self.yaw)
        rot_x = _rotation_matrix_float("x", self.pitch)
        return rot_x @ (rot_y @ point)

    def _project(self, point_view: np.ndarray) -> tuple[int, int] | None:
        denom = self.camera_distance - point_view[2]
        if denom <= 0.2:
            return None
        x = self.size[0] * 0.5 + self.focal * point_view[0] / denom
        y = self.size[1] * 0.54 - self.focal * point_view[1] / denom
        return int(x), int(y)

    @staticmethod
    def _sticker_vertices(meta: dict) -> list[np.ndarray]:
        face = meta["face"]
        row = int(meta["row"])
        col = int(meta["col"])

        spec = FACE_SPECS[face]
        n = np.array(spec["normal"], dtype=np.float64)
        r = np.array(spec["right"], dtype=np.float64)
        up = np.array(spec["up"], dtype=np.float64)

        a0 = -1.0 + col
        a1 = a0 + 1.0
        b_top = 1.0 - row
        b_bottom = b_top - 1.0

        p00 = n + a0 * r + b_top * up
        p10 = n + a1 * r + b_top * up
        p11 = n + a1 * r + b_bottom * up
        p01 = n + a0 * r + b_bottom * up

        epsilon = 0.03
        return [p00 + epsilon * n, p10 + epsilon * n, p11 + epsilon * n, p01 + epsilon * n]

    @staticmethod
    def _smoothstep01(t: float) -> float:
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)

    def _start_action_animation(self, action: int, duration_ms: int | None = None):
        if self.animating:
            print("[GUI] _start_action_animation: ignored (already animating)", file=sys.stderr)
            return
        self.anim_duration_ms = int(duration_ms) if duration_ms is not None else self._default_anim_duration_ms
        self.animating = True
        self.anim_action = int(action)
        self.anim_start_ms = pygame.time.get_ticks()
        self.anim_base_state = self.engine.get_state()
        print(f"[GUI] _start_action_animation: started action={action} ({ACTION_NAMES[action]})", file=sys.stderr)

    def _update_animation(self):
        if not self.animating:
            return
        elapsed = pygame.time.get_ticks() - self.anim_start_ms
        if elapsed >= self.anim_duration_ms:
            state = None
            action_done = self.anim_action
            try:
                state = self.engine.step(self.anim_action)
                print(f"[GUI] _update_animation: step done action={action_done} history_len={len(self.engine.history)}", file=sys.stderr)
            except Exception as e:
                traceback.print_exc(file=sys.stderr)
                print(f"[GUI] _update_animation: engine.step() failed: {e}", file=sys.stderr)
            finally:
                self.animating = False
                self.anim_action = -1
                self.anim_base_state = None
                self.anim_duration_ms = self._default_anim_duration_ms
            if state is not None:
                with self._anim_cond:
                    if self._active_anim_request is not None:
                        self._active_anim_request["state"] = state.copy()
                        self._active_anim_request["done"] = True
                        self._anim_cond.notify_all()
                        self._active_anim_request = None

    def _process_pending_animation_request(self):
        if self.animating:
            return
        with self._anim_cond:
            if self._pending_anim_request is None:
                return
            req = self._pending_anim_request
            self._pending_anim_request = None
            self._active_anim_request = req

        self._start_action_animation(req["action"], duration_ms=req["duration_ms"])

    def animate_step_blocking(self, action: int, duration_ms: int | None = None) -> np.ndarray:
        if not isinstance(action, int) or action < 0 or action >= 12:
            raise ValueError("action must be an integer in range 0..11")
        if duration_ms is not None and duration_ms <= 0:
            raise ValueError("duration_ms must be positive")

        req = {
            "action": int(action),
            "duration_ms": int(duration_ms) if duration_ms is not None else self._default_anim_duration_ms,
            "done": False,
            "state": None,
        }

        with self._anim_cond:
            while self._pending_anim_request is not None:
                self._anim_cond.wait(timeout=0.05)
            self._pending_anim_request = req
            self._anim_cond.notify_all()

            while not req["done"]:
                self._anim_cond.wait(timeout=0.05)

        return req["state"].copy()

    def _animation_transform(self, meta: dict) -> tuple[np.ndarray | None, np.ndarray]:
        if not self.animating or self.anim_base_state is None:
            return None, np.array(meta["normal"], dtype=np.float64)

        face, direction = ACTION_TABLE[self.anim_action]
        axis, layer_sign = FACE_AXIS_LAYER[face]
        axis_idx = AXIS_INDEX[axis]
        cubie = meta["cubie"]
        if int(cubie[axis_idx]) != layer_sign:
            return None, np.array(meta["normal"], dtype=np.float64)

        elapsed = pygame.time.get_ticks() - self.anim_start_ms
        t = self._smoothstep01(elapsed / self.anim_duration_ms)
        full_deg = CLOCKWISE_ANGLE_DEG[face] if direction > 0 else -CLOCKWISE_ANGLE_DEG[face]
        rot = _rotation_matrix_float(axis, math.radians(full_deg) * t)
        normal = rot @ np.array(meta["normal"], dtype=np.float64)
        return rot, normal

    def _piece_permutation(self) -> np.ndarray:
        """Cumulative permutation: perm[i] = solved-state index of the piece now at slot i."""
        perm = np.arange(24, dtype=np.int32)
        # Snapshot to avoid "list changed size during iteration" if server thread appends to history
        history_snapshot = list(self.engine.history)
        for action in history_snapshot:
            if 0 <= action < len(MOVE_PERMUTATIONS):
                perm = perm[MOVE_PERMUTATIONS[action]]
        return perm

    def _draw_cube(self):
        if self.animating and self.anim_base_state is not None:
            state = self.anim_base_state.reshape(6, 4)
        else:
            state = self.engine.get_state().reshape(6, 4)

        # Labels follow the cell: label at slot i = label of the piece currently at slot i
        piece_at = self._piece_permutation()

        draw_items = []
        for meta in STICKER_MODEL:
            face_idx = FACE_INDEX[meta["face"]]
            sticker_idx = int(meta["row"]) * 2 + int(meta["col"])
            color_id = int(state[face_idx, sticker_idx])

            poly_world = self._sticker_vertices(meta)
            rot_anim, normal_world = self._animation_transform(meta)
            if rot_anim is not None:
                poly_world = [rot_anim @ p for p in poly_world]

            poly_view = [self._apply_camera(p) for p in poly_world]
            normal_view = self._apply_camera(normal_world)

            if normal_view[2] <= 0.0:
                continue

            poly_screen = [self._project(p) for p in poly_view]
            if any(pt is None for pt in poly_screen):
                continue

            if self.show_index is not None and meta["idx"] == self.show_index:
                color = SHOW_INDEX_BLACK
            else:
                color = COLOR_MAP[color_id]
            depth = float(sum(p[2] for p in poly_view) / 4.0)
            # Piece at this slot came from solved index piece_at[meta["idx"]]
            idx = int(piece_at[meta["idx"]])
            label = STICKER_LABELS[max(0, min(idx, 23))]
            draw_items.append((depth, poly_screen, color, label))

            draw_items.sort(key=lambda x: x[0])
        for _, poly, color, label in draw_items:
            pygame.draw.polygon(self.screen, color, poly)
            pygame.draw.polygon(self.screen, LINE, poly, 2)
            # Draw slot label at polygon center (labels follow their cell on moves)
            cx = sum(p[0] for p in poly) / 4
            cy = sum(p[1] for p in poly) / 4
            text_color = (255, 255, 255) if color == SHOW_INDEX_BLACK else (28, 28, 36)
            surf = self.sticker_font.render(label, True, text_color)
            self.screen.blit(surf, (cx - surf.get_width() // 2, cy - surf.get_height() // 2))

    def _draw_net(self):
        """Draw the unfolded cube net: cell names (a..g, H), neutral background; green only if correct piece at that cell."""
        piece_at = self._piece_permutation()  # piece_at[slot] = solved-index of piece now at that slot

        cell_size = 40
        origin_x = 720
        origin_y = 140
        net_correct_green = (30, 160, 30)  # green when piece_at[idx] == idx (correct piece at correct cell)

        for r in range(6):
            for c in range(6):
                val = NET_MATRIX[r][c]
                if val == 0:
                    continue
                rect = pygame.Rect(origin_x + c * cell_size, origin_y + r * cell_size, cell_size, cell_size)
                if val == 1:
                    # Fixed corner H (marked 1 on net): always neutral, label "H"
                    pygame.draw.rect(self.screen, NET_FILLED_COLOR, rect)
                    label = "H"
                    txt = self.sticker_font.render(label, True, TEXT)
                    self.screen.blit(txt, (rect.centerx - txt.get_width() // 2, rect.centery - txt.get_height() // 2))
                else:
                    idx = LABEL_TO_INDEX.get(val)
                    if idx is not None:
                        cell_name = SLOT_INDEX_TO_CELL_NAME[idx]
                        # Green only if the piece that belongs at this cell is at this cell
                        is_correct = int(piece_at[idx]) == idx
                        bg = net_correct_green if is_correct else NET_FILLED_COLOR
                        pygame.draw.rect(self.screen, bg, rect)
                        txt = self.sticker_font.render(
                            cell_name, True, (28, 28, 36) if is_correct else TEXT
                        )
                        self.screen.blit(txt, (rect.centerx - txt.get_width() // 2, rect.centery - txt.get_height() // 2))
                    else:
                        pygame.draw.rect(self.screen, NET_FILLED_COLOR, rect)
                        txt = self.sticker_font.render(str(val), True, TEXT)
                        self.screen.blit(txt, (rect.centerx - txt.get_width() // 2, rect.centery - txt.get_height() // 2))
                pygame.draw.rect(self.screen, LINE, rect, 1)

    def _draw_buttons(self):
        eval_label = "Eval: ON" if self.eval_enabled else "Eval: OFF"
        eval_sparse_label = "Eval Sparse: ON" if self.eval_sparse_enabled else "Eval Sparse: OFF"
        eval_sparse_torch_label = "Eval SpTorch: ON" if self.eval_sparse_torch_enabled else "Eval SpTorch: OFF"
        for rect, label in (
            (self.scramble_btn, "Scramble"),
            (self.reset_btn, "Reset"),
            (self.eval_btn, eval_label),
            (self.eval_sparse_btn, eval_sparse_label),
            (self.eval_sparse_torch_btn, eval_sparse_torch_label),
        ):
            pygame.draw.rect(self.screen, BUTTON, rect, border_radius=8)
            pygame.draw.rect(self.screen, BUTTON_BORDER, rect, width=2, border_radius=8)
            txt = self.font.render(label, True, TEXT)
            self.screen.blit(txt, (rect.centerx - txt.get_width() // 2, rect.centery - txt.get_height() // 2))

        pygame.draw.rect(self.screen, BUTTON, self.eval_anti_repeat_checkbox, border_radius=4)
        pygame.draw.rect(self.screen, BUTTON_BORDER, self.eval_anti_repeat_checkbox, width=2, border_radius=4)
        if self.eval_anti_repeat_enabled:
            c = self.eval_anti_repeat_checkbox.center
            pygame.draw.line(self.screen, TEXT, (c[0] - 6, c[1]), (c[0] - 1, c[1] + 6), 2)
            pygame.draw.line(self.screen, TEXT, (c[0] - 1, c[1] + 6), (c[0] + 7, c[1] - 6), 2)
        label = self.small_font.render("Anti-repeat x5", True, TEXT)
        self.screen.blit(label, (self.eval_anti_repeat_checkbox.right + 10, self.eval_anti_repeat_checkbox.y + 3))

    def _draw_hud(self):
        payload = self.engine.state_payload()
        header = self.font.render(
            f"HTTP {self.server.host}:{self.server.port} | steps={payload['step_count']} | solved={not payload['scrambled']}",
            True,
            TEXT,
        )
        controls = self.small_font.render(
            "Drag with mouse to rotate view | Keys: U/J D/C L/K R/E F/G B/N | Scramble/Reset | ESC",
            True,
            TEXT,
        )

        self.screen.blit(header, (24, 18))
        self.screen.blit(controls, (24, 48))

        y = 84
        for action_idx, name in enumerate(ACTION_NAMES):
            txt = self.small_font.render(f"{action_idx}: {name}", True, TEXT)
            self.screen.blit(txt, (24 + (action_idx % 4) * 120, y + (action_idx // 4) * 20))

    def _load_eval_policy(self):
        ckpt = CheckpointManager(self.eval_checkpoint_dir)
        policy, episode = ckpt.load_latest()
        if policy is None:
            return None, None
        return policy, episode

    def _toggle_eval(self):
        if self.eval_enabled:
            self.eval_enabled = False
            self.eval_action_history = []
            self.eval_runs_total += 1
            print("eval_mode disabled")
            return
        self.eval_sparse_enabled = False
        policy, episode = self._load_eval_policy()
        if policy is None:
            print(f"eval_mode cannot enable: no checkpoints found in '{self.eval_checkpoint_dir}'")
            return
        self.eval_policy = policy
        self.eval_enabled = True
        self.eval_action_history = []
        print(f"eval_mode enabled (checkpoint episode={episode})")

    def _toggle_eval_sparse(self):
        if self.eval_sparse_enabled:
            self.eval_sparse_enabled = False
            self.eval_action_history = []
            self.eval_runs_total += 1
            print("eval_sparse disabled")
            return
        self.eval_enabled = False
        policy, episode = load_sparse_latest(self.eval_sparse_checkpoint_dir)
        if policy is None:
            print(f"eval_sparse cannot enable: no checkpoints in '{self.eval_sparse_checkpoint_dir}'")
            return
        self.eval_sparse_policy = policy
        self.eval_sparse_enabled = True
        self.eval_action_history = []
        print(f"eval_sparse enabled (checkpoint episode={episode})")

    def _toggle_eval_sparse_torch(self):
        if self.eval_sparse_torch_enabled:
            self.eval_sparse_torch_enabled = False
            self.eval_action_history = []
            self.eval_runs_total += 1
            print("eval_sparse_torch disabled")
            return
        self.eval_enabled = False
        self.eval_sparse_enabled = False
        policy, episode = load_sparse_torch_latest(self.eval_sparse_torch_checkpoint_dir)
        if policy is None:
            print(f"eval_sparse_torch cannot enable: no checkpoints in '{self.eval_sparse_torch_checkpoint_dir}'")
            return
        self.eval_sparse_torch_policy = policy
        self.eval_sparse_torch_enabled = True
        self.eval_action_history = []
        print(f"eval_sparse_torch enabled (checkpoint episode={episode})")

    @staticmethod
    def _is_bad_pattern(history: list[int], candidate: int) -> bool:
        """True if adding candidate creates a wasteful pattern:
        - repeat-3: last 2 actions == candidate (a,a,a)
        - oscillate: last 3 actions form a,b,a and candidate==b where b==a^1  (a,b,a,b)
        """
        h = history
        # repeat-3: [..., a, a, candidate] where a == candidate
        if len(h) >= 2 and h[-1] == candidate and h[-2] == candidate:
            return True
        # oscillate: [..., a, b, a, candidate==b] where b == a^1
        if len(h) >= 3 and h[-1] == h[-3] and h[-2] == (h[-3] ^ 1) and candidate == h[-2]:
            return True
        return False

    def _best_allowed_action(self, probs: np.ndarray, history: list[int]) -> int:
        """Return highest-prob action that doesn't create a bad pattern."""
        order = np.argsort(probs)[::-1]
        for cand in order:
            if not self._is_bad_pattern(history, int(cand)):
                return int(cand)
        return int(order[0])  # fallback: all bad, pick best anyway

    def _eval_tick(self):
        eval_active = self.eval_enabled or self.eval_sparse_enabled or self.eval_sparse_torch_enabled
        if not eval_active:
            return
        if self.animating:
            return
        if self.engine.is_solved():
            steps = len(self.eval_action_history)
            self.eval_runs_solved += 1
            self.eval_runs_total += 1
            rate = self.eval_runs_solved / self.eval_runs_total
            if self.eval_sparse_torch_enabled:
                mode = "eval_sparse_torch"
            elif self.eval_sparse_enabled:
                mode = "eval_sparse"
            else:
                mode = "eval"
            print(f"{mode} finished: solved in {steps} steps | session success_rate={self.eval_runs_solved}/{self.eval_runs_total} ({rate:.1%})")
            self.eval_enabled = False
            self.eval_sparse_enabled = False
            self.eval_sparse_torch_enabled = False
            self.eval_action_history = []
            return

        if self.eval_sparse_torch_enabled:
            if self.eval_sparse_torch_policy is None:
                self.eval_sparse_torch_enabled = False
                self.eval_action_history = []
                return
            history_snapshot = list(self.engine.history)
            state_sparse = _sparse_state_from_engine(self.engine, history_snapshot)
            hist_oh = self.eval_sparse_torch_policy.history_one_hot(self.eval_action_history)
            action_6, probs = self.eval_sparse_torch_policy.sample_action(state_sparse, hist_oh)
            if self.eval_anti_repeat_enabled:
                action_6 = self._best_allowed_action(probs, self.eval_action_history)
            self._start_action_animation(ACTION_6_TO_12[action_6], duration_ms=self._default_anim_duration_ms)
            self.eval_action_history.append(int(action_6))
            if len(self.eval_action_history) > 8:
                self.eval_action_history = self.eval_action_history[-8:]
            return

        if self.eval_sparse_enabled:
            if self.eval_sparse_policy is None:
                self.eval_sparse_enabled = False
                self.eval_action_history = []
                return
            history_snapshot = list(self.engine.history)
            state_sparse = _sparse_state_from_engine(self.engine, history_snapshot)
            hist_oh = self.eval_sparse_policy.history_one_hot(self.eval_action_history)
            action_6, probs = self.eval_sparse_policy.sample_action(state_sparse, hist_oh)
            if self.eval_anti_repeat_enabled:
                action_6 = self._best_allowed_action(probs, self.eval_action_history)
            self._start_action_animation(ACTION_6_TO_12[action_6], duration_ms=self._default_anim_duration_ms)
            self.eval_action_history.append(int(action_6))
            if len(self.eval_action_history) > 8:
                self.eval_action_history = self.eval_action_history[-8:]
            return

        if self.eval_policy is None:
            self.eval_enabled = False
            self.eval_action_history = []
            return
        state_one_hot = encode_one_hot(self.engine.get_state()).astype(np.float64)
        action, probs = self.eval_policy.sample_action(state_one_hot)
        if self.eval_anti_repeat_enabled:
            recent = self.eval_action_history[-4:]
            if len(recent) == 4 and all(a == action for a in recent):
                alt = self._second_best_action(probs, action)
                if alt != action:
                    action = alt
        self._start_action_animation(action, duration_ms=self._default_anim_duration_ms)
        self.eval_action_history.append(int(action))
        if len(self.eval_action_history) > 8:
            self.eval_action_history = self.eval_action_history[-8:]

    def _scramble_6actions(self, steps: int) -> None:
        """Scramble using only the 6-action subset (H face fixed). Avoids inverse of last action."""
        rng = np.random.default_rng()
        actions_6 = np.array(ACTION_6_TO_12, dtype=np.int32)
        prev_action: int | None = None
        for _ in range(steps):
            if prev_action is not None:
                inverse_action = prev_action ^ 1
                candidates = actions_6[actions_6 != inverse_action]
            else:
                candidates = actions_6
            action = int(rng.choice(candidates))
            self.engine.step(action)
            prev_action = action

    def run(self):
        self.server.start_background(daemon=True)
        running = True

        while running:
            try:
                self.clock.tick(60)
                self._process_pending_animation_request()
                self._update_animation()
                self._eval_tick()
            except Exception:
                traceback.print_exc(file=sys.stderr)
                print("[GUI] main loop tick failed", file=sys.stderr)

            for event in pygame.event.get():
                try:
                    if event.type == pygame.QUIT:
                        running = False

                    elif event.type == pygame.KEYDOWN:
                        action = KEY_TO_ACTION.get(event.key)
                        if action is None and getattr(event, "scancode", None) is not None:
                            action = SCANCODE_TO_ACTION.get(event.scancode)
                        if event.key in (pygame.K_ESCAPE, pygame.K_q):
                            running = False
                        elif action is not None:
                            self._start_action_animation(action)

                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        if self.scramble_btn.collidepoint(event.pos):
                            if not self.animating:
                                self._scramble_6actions(self.scramble_steps)
                                self.eval_action_history = []
                        elif self.reset_btn.collidepoint(event.pos):
                            if not self.animating:
                                self.engine.reset()
                                self.eval_action_history = []
                        elif self.eval_btn.collidepoint(event.pos):
                            self._toggle_eval()
                        elif self.eval_sparse_btn.collidepoint(event.pos):
                            self._toggle_eval_sparse()
                        elif self.eval_sparse_torch_btn.collidepoint(event.pos):
                            self._toggle_eval_sparse_torch()
                        elif self.eval_anti_repeat_checkbox.collidepoint(event.pos):
                            self.eval_anti_repeat_enabled = not self.eval_anti_repeat_enabled
                            print(f"eval_anti_repeat_x5={'on' if self.eval_anti_repeat_enabled else 'off'}")
                        else:
                            self.dragging = True
                            self.last_mouse = event.pos

                    elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                        self.dragging = False

                    elif event.type == pygame.MOUSEMOTION and self.dragging:
                        dx = event.pos[0] - self.last_mouse[0]
                        dy = event.pos[1] - self.last_mouse[1]
                        self.last_mouse = event.pos
                        self.yaw += dx * 0.01
                        self.pitch += dy * 0.01
                        self.pitch = max(-1.2, min(1.2, self.pitch))
                except Exception:
                    traceback.print_exc(file=sys.stderr)
                    print("[GUI] event handler failed", file=sys.stderr)

            self.screen.fill(BG)
            self._draw_hud()
            try:
                self._draw_cube()
                self._draw_net()
            except Exception:
                traceback.print_exc(file=sys.stderr)
                print("[GUI] _draw_cube/_draw_net failed", file=sys.stderr)
            self._draw_buttons()
            pygame.display.flip()

        self.server.shutdown()
        pygame.quit()
