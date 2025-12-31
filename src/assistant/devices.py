from __future__ import annotations

import alsaaudio
import sys


def list_capture_devices() -> list[dict]:
    """
    Returns a list of ALSA capture devices with:
      - card index
      - device index
      - human-readable name
      - hw string usable in config.yaml
    """
    devices = []

    cards = alsaaudio.cards()
    for card_index, card_name in enumerate(cards):
        try:
            pcm_devices = alsaaudio.pcms(
                alsaaudio.PCM_CAPTURE,
                card=card_index
            )
        except Exception:
            continue

        for dev_name in pcm_devices:
            # dev_name is usually something like 'device 0: USB Audio'
            # Extract device index if possible
            dev_index = None
            for token in dev_name.split():
                if token.isdigit():
                    dev_index = int(token)
                    break

            hw = (
                f"hw:{card_index},{dev_index}"
                if dev_index is not None
                else f"hw:{card_index}"
            )

            devices.append({
                "card_index": card_index,
                "card_name": card_name,
                "device_name": dev_name,
                "hw": hw,
            })

    return devices


def main() -> int:
    devices = list_capture_devices()

    if not devices:
        print("No ALSA capture devices found.", file=sys.stderr)
        return 1

    print("Available ALSA capture devices:\n")

    for d in devices:
        print(f"Card {d['card_index']}: {d['card_name']}")
        print(f"  Device : {d['device_name']}")
        print(f"  Config : {d['hw']}")
        print()

    print("Use the value under 'Config' in config/config.yaml:")
    print()
    print("audio:")
    print("  device: hw:X,Y")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
