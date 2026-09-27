"""Atomic checkpoint writes and RNG preservation; no changes to optimization."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import random

import numpy as np
import torch


def capture_rng_state():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


@contextmanager
def preserve_rng_state():
    """Keep per-epoch test iteration from changing the next training RNG draw."""
    state = capture_rng_state()
    try:
        yield
    finally:
        restore_rng_state(state)


def atomic_torch_save(state, filename):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        with temporary.open('wb') as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(filename, payload):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        with temporary.open('w') as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_resume_args(saved, current, names):
    differences = [name for name in names if getattr(saved, name, None) != getattr(current, name, None)]
    if differences:
        raise ValueError('Resume would change the training protocol: ' + ', '.join(differences))
