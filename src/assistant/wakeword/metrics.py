from __future__ import annotations

import time


class WakeWordMetrics:
    """
    Collects wake-word related metrics independent of model.
    """

    def __init__(self):
        self.start_time = time.time()

        self.triggers = 0
        self.suppressed = 0
        self.last_trigger_ts = None

    def on_trigger(self):
        now = time.time()
        self.triggers += 1

        if self.last_trigger_ts is not None:
            delta = now - self.last_trigger_ts
            print(f"[metrics] time since last trigger: {delta:.1f}s")

        self.last_trigger_ts = now

    def on_suppressed(self):
        self.suppressed += 1

    def report(self):
        uptime = time.time() - self.start_time
        hours = uptime / 3600.0 if uptime > 0 else 0.0

        print(
            "[metrics] uptime={:.2f}h triggers={} suppressed={} rate={:.2f}/h".format(
                hours,
                self.triggers,
                self.suppressed,
                (self.triggers / hours) if hours > 0 else 0.0,
            )
        )
