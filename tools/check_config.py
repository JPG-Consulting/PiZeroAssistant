"""Validate a configuration file loads correctly."""

import argparse

from voiceassistant.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    config = load_config(args.path)
    print("Loaded config:", config)


if __name__ == "__main__":
    main()
