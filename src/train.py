from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.args import ConfigError, build_config
from src.core.seed import set_seed
from src.engine.train_engine import run_training


def main() -> None:
    try:
        config = build_config(train_mode=True)
    except ConfigError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        set_seed(config.seed)
        run_training(config)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"训练失败: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
