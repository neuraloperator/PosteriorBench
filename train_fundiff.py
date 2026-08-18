from __future__ import annotations

from training_fundiff.train import train_from_config
from utils.yaml_config import Config, process_arguments


def main() -> None:
    conf = Config(process_arguments())
    train_from_config(conf)


if __name__ == "__main__":
    main()

