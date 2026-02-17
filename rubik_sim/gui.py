"""Pygame GUI for the 3x3 Rubik simulator."""

from __future__ import annotations

import math
import threading

import numpy as np
import pygame

from rubik_rl.checkpoint import CheckpointManager
from .actions import (
    ACTION_NAMES,
    ACTION_TABLE,
    CLOCKWISE_ANGLE_DEG,
    FACE_AXIS_LAYER,
    FACE_INDEX,
    FACE_SIZE,
    FACE_SPECS,
    STICKER_MODEL,
)
from .engine import RubikEngine
from .server import RubikHTTPServer
from .state_codec import encode_one_hot

COLOR_MAP = {
    0: (245, 245, 245),
    1: (220, 30, 30),
    2: (30, 160, 30),
    3: (240, 220, 40),
    4: (255, 140, 20),
    5: (30, 90, 220),
}

BG = (18, 22, 30)
LINE = (28, 32, 42)
TEXT = (220, 225, 235)
BUTTON = (52, 60, 78)
BUTTON_BORDER = (92, 110, 140)

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

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


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
    ):
        self.engine = engine
        self.scramble_steps = scramble_steps
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
        self.size = (960, 640)
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("Rubik 3x3 Simulator")
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("monospace", 18)
        self.small_font = pygame.font.SysFont("monospace", 14)

        self.scramble_btn = pygame.Rect(40, 560, 180, 44)
        self.reset_btn = pygame.Rect(240, 560, 180, 44)
        self.eval_btn = pygame.Rect(440, 560, 200, 44)
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
        self.eval_anti_repeat_enabled = False
        self.eval_action_history: list[int] = []

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

        # Map 3x3 row/col to local face coordinates in [-1, 1] with cell width 2/3.
        cell = 2.0 / FACE_SIZE
        a0 = -1.0 + col * cell
        a1 = a0 + cell
        b_top = 1.0 - row * cell
        b_bottom = b_top - cell

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
            return
        self.anim_duration_ms = int(duration_ms) if duration_ms is not None else self._default_anim_duration_ms
        self.animating = True
        self.anim_action = int(action)
        self.anim_start_ms = pygame.time.get_ticks()
        self.anim_base_state = self.engine.get_state()

    def _update_animation(self):
        if not self.animating:
            return
        elapsed = pygame.time.get_ticks() - self.anim_start_ms
        if elapsed >= self.anim_duration_ms:
            state = self.engine.step(self.anim_action)
            self.animating = False
            self.anim_action = -1
            self.anim_base_state = None
            self.anim_duration_ms = self._default_anim_duration_ms

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

    def _draw_cube(self):
        if self.animating and self.anim_base_state is not None:
            state = self.anim_base_state.reshape(6, FACE_SIZE * FACE_SIZE)
        else:
            state = self.engine.get_state().reshape(6, FACE_SIZE * FACE_SIZE)

        draw_items = []
        for meta in STICKER_MODEL:
            face_idx = FACE_INDEX[meta["face"]]
            sticker_idx = int(meta["row"]) * FACE_SIZE + int(meta["col"])
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

            depth = float(sum(p[2] for p in poly_view) / 4.0)
            draw_items.append((depth, poly_screen, COLOR_MAP[color_id]))

        draw_items.sort(key=lambda x: x[0])
        for _, poly, color in draw_items:
            pygame.draw.polygon(self.screen, color, poly)
            pygame.draw.polygon(self.screen, LINE, poly, 2)

    def _draw_buttons(self):
        eval_label = "Eval: ON" if self.eval_enabled else "Eval: OFF"
        for rect, label in (
            (self.scramble_btn, "Scramble"),
            (self.reset_btn, "Reset"),
            (self.eval_btn, eval_label),
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
            print("eval_mode disabled")
            return

        policy, episode = self._load_eval_policy()
        if policy is None:
            print(f"eval_mode cannot enable: no checkpoints found in '{self.eval_checkpoint_dir}'")
            return
        self.eval_policy = policy
        self.eval_enabled = True
        self.eval_action_history = []
        print(f"eval_mode enabled (checkpoint episode={episode})")

    @staticmethod
    def _second_best_action(probs: np.ndarray, best_action: int) -> int:
        order = np.argsort(probs)
        for idx in range(len(order) - 1, -1, -1):
            cand = int(order[idx])
            if cand != best_action:
                return cand
        return int(best_action)

    def _eval_tick(self):
        if not self.eval_enabled:
            return
        if self.animating:
            return
        if self.engine.is_solved():
            self.eval_enabled = False
            self.eval_action_history = []
            print("eval_mode finished: cube solved")
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

    def run(self):
        self.server.start_background(daemon=True)
        running = True

        while running:
            self.clock.tick(60)
            self._process_pending_animation_request()
            self._update_animation()
            self._eval_tick()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key in KEY_TO_ACTION:
                        self._start_action_animation(KEY_TO_ACTION[event.key])

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.scramble_btn.collidepoint(event.pos):
                        if not self.animating:
                            self.engine.scramble(self.scramble_steps)
                            self.eval_action_history = []
                    elif self.reset_btn.collidepoint(event.pos):
                        if not self.animating:
                            self.engine.reset()
                            self.eval_action_history = []
                    elif self.eval_btn.collidepoint(event.pos):
                        self._toggle_eval()
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

            self.screen.fill(BG)
            self._draw_hud()
            self._draw_cube()
            self._draw_buttons()
            pygame.display.flip()

        self.server.shutdown()
        pygame.quit()
