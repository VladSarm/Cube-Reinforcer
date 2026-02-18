"""Action and geometry utilities for the 2x2 Rubik simulator."""

from __future__ import annotations

from collections import deque

import numpy as np

FACE_ORDER = ("U", "R", "F", "D", "L", "B")
FACE_INDEX = {face: i for i, face in enumerate(FACE_ORDER)}
N_FACES = 6
STICKERS_PER_FACE = 4
STATE_SIZE = N_FACES * STICKERS_PER_FACE

# Face specification from outside view.
FACE_SPECS = {
    "U": {"normal": (0, 1, 0), "right": (1, 0, 0), "up": (0, 0, -1)},
    "R": {"normal": (1, 0, 0), "right": (0, 0, -1), "up": (0, 1, 0)},
    "F": {"normal": (0, 0, 1), "right": (1, 0, 0), "up": (0, 1, 0)}, # frontal face
    "D": {"normal": (0, -1, 0), "right": (1, 0, 0), "up": (0, 0, 1)},
    "L": {"normal": (-1, 0, 0), "right": (0, 0, 1), "up": (0, 1, 0)},
    "B": {"normal": (0, 0, -1), "right": (-1, 0, 0), "up": (0, 1, 0)},
}

# Action index -> (face, direction)
# direction: +1 means clockwise from the face viewpoint, -1 means counter-clockwise.
ACTION_TABLE = [
    ("U", +1),
    ("U", -1),
    ("D", +1),
    ("D", -1),
    ("L", +1),
    ("L", -1),
    ("R", +1),
    ("R", -1),
    ("F", +1),
    ("F", -1),
    ("B", +1),
    ("B", -1),
]

ACTION_NAMES = [f"{face}{'+' if direction > 0 else '-'}" for face, direction in ACTION_TABLE]

# Clockwise turn from face viewpoint expressed as world-axis rotation angle.
CLOCKWISE_ANGLE_DEG = {
    "U": -90,
    "D": +90,
    "L": +90,
    "R": -90,
    "F": -90,
    "B": +90,
}

FACE_AXIS_LAYER = {
    "U": ("y", +1),
    "D": ("y", -1),
    "L": ("x", -1),
    "R": ("x", +1),
    "F": ("z", +1),
    "B": ("z", -1),
}


def solved_state() -> np.ndarray:
    """Return the canonical solved flat state of length 24."""
    return np.repeat(np.arange(N_FACES, dtype=np.int8), STICKERS_PER_FACE) #[0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4, 5,5,5,5]


def action_name(action: int) -> str:
    return ACTION_NAMES[action]


def _rotation_matrix(axis: str, angle_deg: int) -> np.ndarray:
    """Return integer rotation matrix for ±90 around x/y/z axes."""
    if axis == "x" and angle_deg == +90:
        return np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.int8)
    if axis == "x" and angle_deg == -90:
        return np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.int8)
    if axis == "y" and angle_deg == +90:
        return np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.int8)
    if axis == "y" and angle_deg == -90:
        return np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.int8)
    if axis == "z" and angle_deg == +90:
        return np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int8)
    if axis == "z" and angle_deg == -90:
        return np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.int8)
    raise ValueError(f"Unsupported rotation: axis={axis}, angle={angle_deg}")


def _build_sticker_model() -> tuple[list[dict[str, np.ndarray]], dict[tuple[int, int, int], str]]:
    stickers: list[dict[str, np.ndarray]] = []
    normal_to_face: dict[tuple[int, int, int], str] = {}

    for face in FACE_ORDER:
        spec = FACE_SPECS[face]
        n = np.array(spec["normal"], dtype=np.int8)
        r = np.array(spec["right"], dtype=np.int8)
        up = np.array(spec["up"], dtype=np.int8)
        normal_to_face[tuple(int(v) for v in n)] = face

        for row in range(2):
            for col in range(2):
                col_sign = -1 if col == 0 else +1
                row_sign = +1 if row == 0 else -1
                center = 2 * n + col_sign * r + row_sign * up
                cubie = center - n
                idx = FACE_INDEX[face] * STICKERS_PER_FACE + row * 2 + col
                stickers.append(
                    {
                        "idx": idx,
                        "face": face,
                        "row": row,
                        "col": col,
                        "center": center,
                        "normal": n,
                        "cubie": cubie,
                    }
                )

    stickers.sort(key=lambda s: s["idx"])
    return stickers, normal_to_face


_STICKERS, _NORMAL_TO_FACE = _build_sticker_model()


def _face_row_col_from_center(face: str, center: np.ndarray) -> tuple[int, int]:
    spec = FACE_SPECS[face]
    n = np.array(spec["normal"], dtype=np.int8)
    r = np.array(spec["right"], dtype=np.int8)
    up = np.array(spec["up"], dtype=np.int8)

    offset = center - 2 * n
    col_sign = int(np.dot(offset, r))
    row_sign = int(np.dot(offset, up))

    if col_sign not in (-1, 1) or row_sign not in (-1, 1):
        raise ValueError(f"Invalid center for face {face}: {center}")

    col = 0 if col_sign == -1 else 1
    row = 0 if row_sign == +1 else 1
    return row, col


def _axis_info_for_face(face: str) -> tuple[str, int]:
    return FACE_AXIS_LAYER[face]


def _generate_face_turn_permutation(face: str, direction: int) -> np.ndarray:
    axis, layer_sign = _axis_info_for_face(face)
    angle = CLOCKWISE_ANGLE_DEG[face] if direction > 0 else -CLOCKWISE_ANGLE_DEG[face]
    rot = _rotation_matrix(axis, angle)

    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    perm = np.empty(STATE_SIZE, dtype=np.int32)

    for sticker in _STICKERS:
        old_idx = int(sticker["idx"])
        center = sticker["center"]
        normal = sticker["normal"]
        cubie = center - normal

        if int(cubie[axis_idx]) == layer_sign:
            new_center = rot @ center
            new_normal = rot @ normal
        else:
            new_center = center
            new_normal = normal

        face_new = _NORMAL_TO_FACE[tuple(int(v) for v in new_normal)]
        row_new, col_new = _face_row_col_from_center(face_new, new_center)
        new_idx = FACE_INDEX[face_new] * STICKERS_PER_FACE + row_new * 2 + col_new
        perm[new_idx] = old_idx

    return perm


def _generate_move_permutations() -> np.ndarray:
    perms = np.empty((len(ACTION_TABLE), STATE_SIZE), dtype=np.int32)
    for action, (face, direction) in enumerate(ACTION_TABLE):
        perms[action] = _generate_face_turn_permutation(face, direction)
    return perms


def _matrix_key(mat: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in mat.reshape(-1))


def _generate_global_orientation_matrices() -> list[np.ndarray]:
    gens = [_rotation_matrix("x", +90), _rotation_matrix("y", +90), _rotation_matrix("z", +90)]
    identity = np.eye(3, dtype=np.int8)

    mats: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    q: deque[np.ndarray] = deque([identity])

    while q:
        mat = q.popleft()
        key = _matrix_key(mat)
        if key in seen:
            continue
        seen.add(key)
        mats.append(mat)
        for g in gens:
            q.append(g @ mat)

    if len(mats) != 24:
        raise RuntimeError(f"Expected 24 orientation matrices, got {len(mats)}")
    return mats


def _generate_orientation_permutations() -> np.ndarray:
    mats = _generate_global_orientation_matrices()
    perms = np.empty((len(mats), STATE_SIZE), dtype=np.int32)

    for i, mat in enumerate(mats):
        perm = np.empty(STATE_SIZE, dtype=np.int32)
        for sticker in _STICKERS:
            old_idx = int(sticker["idx"])
            new_center = mat @ sticker["center"]
            new_normal = mat @ sticker["normal"]

            face_new = _NORMAL_TO_FACE[tuple(int(v) for v in new_normal)]
            row_new, col_new = _face_row_col_from_center(face_new, new_center)
            new_idx = FACE_INDEX[face_new] * STICKERS_PER_FACE + row_new * 2 + col_new
            perm[new_idx] = old_idx
        perms[i] = perm

    return perms


MOVE_PERMUTATIONS = _generate_move_permutations()
ORIENTATION_PERMUTATIONS = _generate_orientation_permutations()

# Read-only sticker metadata for rendering and animation.
STICKER_MODEL = tuple(
    {
        "idx": int(s["idx"]),
        "face": s["face"],
        "row": int(s["row"]),
        "col": int(s["col"]),
        "center": tuple(int(v) for v in s["center"]),
        "normal": tuple(int(v) for v in s["normal"]),
        "cubie": tuple(int(v) for v in s["cubie"]),
    }
    for s in _STICKERS
)
