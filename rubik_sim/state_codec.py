"""State validation and codec helpers."""

from __future__ import annotations

import numpy as np

from .actions import N_FACES, STATE_SIZE, STICKERS_PER_FACE


class StateValidationError(ValueError):
    """Raised when an input state is invalid."""


def _validate_color_ids(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.int16).reshape(-1)
    if arr.size != STATE_SIZE:
        raise StateValidationError(f"State must have {STATE_SIZE} stickers, got {arr.size}")

    if np.any(arr < 0) or np.any(arr >= N_FACES):
        raise StateValidationError("State contains invalid color IDs; allowed values are 0..5")

    counts = np.bincount(arr, minlength=N_FACES)
    expected = np.full(N_FACES, STICKERS_PER_FACE, dtype=np.int64)
    if not np.array_equal(counts, expected):
        raise StateValidationError(
            "Invalid sticker counts; each color 0..5 must appear exactly 4 times"
        )

    return arr.astype(np.int8, copy=True)


def _validate_one_hot(one_hot: np.ndarray) -> np.ndarray:
    arr = np.asarray(one_hot)

    if arr.ndim == 1:
        if arr.size != STATE_SIZE * N_FACES:
            raise StateValidationError(
                f"One-hot state must have {STATE_SIZE * N_FACES} values when flattened"
            )
        arr = arr.reshape(STATE_SIZE, N_FACES)

    if arr.shape != (STATE_SIZE, N_FACES):
        raise StateValidationError(
            f"One-hot state must have shape ({STATE_SIZE}, {N_FACES}), got {arr.shape}"
        )

    arr = arr.astype(np.int8, copy=False)
    if np.any((arr != 0) & (arr != 1)):
        raise StateValidationError("One-hot state must contain only 0/1 values")

    row_sums = arr.sum(axis=1)
    if not np.all(row_sums == 1):
        raise StateValidationError("Each sticker one-hot vector must contain exactly one 1")

    colors = np.argmax(arr, axis=1).astype(np.int8)
    return _validate_color_ids(colors)


def validate_state(state: list[int] | list[list[int]] | np.ndarray) -> np.ndarray:
    """Validate state and return canonical flat color IDs (length 24)."""
    arr = np.asarray(state)

    if arr.ndim == 2:
        return _validate_one_hot(arr)

    if arr.ndim == 1:
        if arr.size == STATE_SIZE:
            return _validate_color_ids(arr)
        if arr.size == STATE_SIZE * N_FACES:
            return _validate_one_hot(arr)

    raise StateValidationError(
        "State must be either color IDs of length 24 or one-hot with shape (24,6)"
    )


def encode_one_hot(state: list[int] | np.ndarray) -> np.ndarray:
    colors = validate_state(state)
    one_hot = np.zeros((STATE_SIZE, N_FACES), dtype=np.int8)
    one_hot[np.arange(STATE_SIZE), colors.astype(np.int64)] = 1
    return one_hot


def state_to_json_one_hot(state: list[int] | np.ndarray) -> list[list[int]]:
    return encode_one_hot(state).astype(int).tolist()


def flat_to_faces(state: list[int] | list[list[int]] | np.ndarray) -> np.ndarray:
    arr = validate_state(state)
    return arr.reshape(N_FACES, STICKERS_PER_FACE)


def faces_to_flat(faces: np.ndarray) -> np.ndarray:
    arr = np.asarray(faces, dtype=np.int16)
    if arr.shape != (N_FACES, STICKERS_PER_FACE):
        raise StateValidationError(
            f"Faces array must have shape ({N_FACES}, {STICKERS_PER_FACE}), got {arr.shape}"
        )
    return validate_state(arr.reshape(-1))
