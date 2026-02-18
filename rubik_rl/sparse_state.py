"""Sparse state (7 bits a..g) and vectorized reward for Rubik 2x2 training.

State space: cells a..g (7 corners). In solved state: A→a, B→b, C→c, D→d, E→e, F→f, G→g.
The 8th corner H is the static one (marked 1 on the net); slot 13, not in a..g.

Reward per cell: if the correct piece is in that cell → 0. Else by how many of the
cell's 3 faces contain that piece: 2 faces → -0.6, 1 face → -0.3, 0 faces → -1.
"""

from __future__ import annotations

import numpy as np

from rubik_sim.actions import ACTION_6_TO_12, MOVE_PERMUTATIONS, STATE_SIZE

# Fixed corner is H (slot 13 on the net; not in KEY_SLOTS). Наблюдаемое состояние — поля a..g.
# In solved state: cell a = slot 0 (A), b=1 (B), c=2 (C), d=3 (D), e=14 (E), f=15 (F), g=12 (G).
CELL_NAMES = ("a", "b", "c", "d", "e", "f", "g")  # state[i] = поле CELL_NAMES[i]
KEY_SLOTS = np.array([0, 1, 2, 3, 14, 15, 12], dtype=np.int32)   # a,b,c,d,e,f,g
TARGET_PIECE = np.array([0, 1, 2, 3, 14, 15, 12], dtype=np.int32)  # A,B,C,D,E,F,G

# Planes (slot sets). abcd = face U; abeg = band; ef = two slots on D.
PLANE_ABCD = frozenset({0, 1, 2, 3})
PLANE_ABEG = frozenset({0, 1, 12, 14})
PLANE_EF = frozenset({14, 15})

# For each cell 0..6 (a..g), the planes (faces) that contain that cell.
# Reward: 2 planes → -0.6, 1 plane → -0.3, 0 planes → -1; at target slot → 0.
CELL_PLANES: tuple[tuple[frozenset[int], ...], ...] = (
    (PLANE_ABCD, PLANE_ABEG),   # a: 2 planes
    (PLANE_ABCD, PLANE_ABEG),   # b: 2 planes
    (PLANE_ABCD,),             # c: 1 plane
    (PLANE_ABCD,),             # d: 1 plane
    (PLANE_ABEG,),             # e: 1 plane
    (PLANE_EF,),               # f: 1 plane
    (PLANE_ABEG,),             # g: 1 plane
)

# Reward: 2 faces → -0.6, 1 face → -0.3, 0 faces (wrong place) → -1
R_CORRECT = 0.0
R_WRONG = -1.0
R_IN_ONE_PLANE = -0.3
R_IN_TWO_PLANES = -0.6

SPARSE_DIM = 7

# In solved state (identity), slot KEY_SLOTS[i] holds piece TARGET_PIECE[i]: A on a, B on b, ...
assert KEY_SLOTS.size == SPARSE_DIM and np.array_equal(KEY_SLOTS, TARGET_PIECE)

# Curriculum order: first corner above static H (D→d), then +F, then +B, then the rest (a,c,e,g).
# Cell indices: 0=a, 1=b, 2=c, 3=d, 4=e, 5=f, 6=g.
# Stage 1: D in d; 2: +F; 3: +B; 4–7: +a, +c, +e, +g.
CURRICULUM_ORDER = np.array([3, 5, 1, 0, 2, 4, 6], dtype=np.int32)  # d, f, b, a, c, e, g

# Shaping: per corner, reward when the piece that belongs there is in a slot.
# reward_shape[i, s] = 1.0 if slot s is target for cell i (piece in place), else 0.33 if s on one of cell i's faces, else 0.
REWARD_SHAPE_MATRIX = np.zeros((SPARSE_DIM, STATE_SIZE), dtype=np.float32)
for i in range(SPARSE_DIM):
    target_slot = KEY_SLOTS[i]
    planes_i = CELL_PLANES[i]
    for s in range(STATE_SIZE):
        if s == target_slot:
            REWARD_SHAPE_MATRIX[i, s] = 1.0
        elif any(s in pl for pl in planes_i):
            REWARD_SHAPE_MATRIX[i, s] = 0.33
        # else 0


def piece_permutation(history: list[int]) -> np.ndarray:
    """Cumulative permutation from 12-action history. perm[slot] = solved-index of piece now at that slot."""
    perm = np.arange(STATE_SIZE, dtype=np.int32)
    for action in history:
        if 0 <= action < len(MOVE_PERMUTATIONS):
            perm = perm[MOVE_PERMUTATIONS[action]]
    return perm


def piece_permutation_6(history_6: list[int]) -> np.ndarray:
    """Same as piece_permutation but history is 6-action indices (angle H fixed)."""
    perm = np.arange(STATE_SIZE, dtype=np.int32)
    for a6 in history_6:
        if 0 <= a6 < len(ACTION_6_TO_12):
            a12 = ACTION_6_TO_12[a6]
            perm = perm[MOVE_PERMUTATIONS[a12]]
    return perm


def slot_of_piece(perm: np.ndarray) -> np.ndarray:
    """Return array of length 24: slot_of_piece[piece_idx] = slot where that piece is."""
    inv = np.empty(STATE_SIZE, dtype=np.int32)
    for slot in range(STATE_SIZE):
        inv[perm[slot]] = slot
    return inv


def sparse_state_from_perm(perm: np.ndarray) -> np.ndarray:
    """Наблюдаемое состояние: вектор длины 7 (поля abcdefg). out[i] = 1, если в ячейке CELL_NAMES[i] правильный угол, иначе 0."""
    assert len(CELL_NAMES) == SPARSE_DIM
    out = np.zeros(SPARSE_DIM, dtype=np.float64)
    for i in range(SPARSE_DIM):
        slot = KEY_SLOTS[i]
        target = TARGET_PIECE[i]
        if perm[slot] == target:
            out[i] = 1.0
    return out


def vectorized_reward(perm: np.ndarray) -> np.ndarray:
    """Reward for each cell a..g. For each cell we look at the piece that should be there (A..G).
    If that piece is at the target slot → 0. Else count how many of the cell's planes contain
    that piece's current slot: 2 → -0.6, 1 → -0.3, 0 → -1."""
    slot_of = slot_of_piece(perm)
    rewards = np.zeros(SPARSE_DIM, dtype=np.float64)

    for cell in range(SPARSE_DIM):
        slot = KEY_SLOTS[cell]
        target = TARGET_PIECE[cell]
        # Correct piece at target cell (3/3 faces) → 0
        if perm[slot] == target:
            rewards[cell] = R_CORRECT
            continue
        # Wrong: where is the correct piece? Count how many of this cell's planes contain it.
        where_piece = slot_of[target]
        n_planes = sum(1 for pl in CELL_PLANES[cell] if where_piece in pl)
        if n_planes >= 2:
            rewards[cell] = R_IN_TWO_PLANES
        elif n_planes == 1:
            rewards[cell] = R_IN_ONE_PLANE
        else:
            rewards[cell] = R_WRONG

    return rewards


def is_solved(perm: np.ndarray) -> bool:
    """True if cube is solved (perm is identity)."""
    return bool(np.array_equal(perm, np.arange(STATE_SIZE, dtype=perm.dtype)))


def scalar_reward(perm: np.ndarray) -> float:
    """Чёткий сигнал: +10 за сборку, -0.1 за каждый ход (без сборки). Иначе агент тонет в -200."""
    return 10.0 if is_solved(perm) else -0.1
