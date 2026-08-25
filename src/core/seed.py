from __future__ import annotations

import random


def _validate_seed(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"seed 必须是非负整数: {seed!r}")


def set_seed(seed: int, deterministic: bool = False) -> None:
    _validate_seed(seed)
    random.seed(seed)

    try:
        import numpy
    except ImportError:
        numpy = None
    if numpy is not None:
        numpy.random.seed(seed)

    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
