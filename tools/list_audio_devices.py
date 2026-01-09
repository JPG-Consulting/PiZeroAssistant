"""List audio devices to assist configuration."""

import sounddevice as sd


def main() -> None:
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        print(f"{idx}: {device['name']}")


if __name__ == "__main__":
    main()
