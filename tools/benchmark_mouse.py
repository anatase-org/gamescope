"""Small Linux uinput helper; only the benchmark supervisor needs root."""
import fcntl
import math
import os
import struct
import threading
import time


class Mouse:
    def __init__(self, width=1920, height=1080, rate=240, radius=160, period=2):
        if not all(math.isfinite(v) for v in (rate, radius, period)) or rate <= 0 or rate > 2000 or radius < 0 or radius >= min(width, height) / 2 or period <= 0:
            raise ValueError('invalid mouse rate, radius, or period')
        self.rate, self.radius, self.period = rate, radius, period
        self.enabled = False
        self.closed = False
        self.events = 0
        self.fd = os.open('/dev/uinput', os.O_WRONLY | os.O_CLOEXEC)
        try:
            for event in (1, 2):  # EV_KEY, EV_REL
                fcntl.ioctl(self.fd, 0x40045564, event)
            for key in (272, 273):  # BTN_LEFT, BTN_RIGHT
                fcntl.ioctl(self.fd, 0x40045565, key)
            for axis in (0, 1):
                fcntl.ioctl(self.fd, 0x40045566, axis)
            device = struct.pack('=80sHHHHI', b'gamescope benchmark mouse', 3, 0x1209, 1, 1, 0)
            os.write(self.fd, device + bytes(64 * 4 * 4))
            fcntl.ioctl(self.fd, 0x5501)  # UI_DEV_CREATE
            time.sleep(0.5)  # Allow libinput to discover the new device.
            self.move(-32767, -32767)
            time.sleep(0.05)
            self.move(width // 2, height // 2)
            self.x = self.y = 0
            self.started = time.monotonic()
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
        except BaseException:
            os.close(self.fd)
            raise

    def move(self, x, y):
        for code, value in ((0, x), (1, y)):
            if value:
                os.write(self.fd, struct.pack('llHHi', 0, 0, 2, code, value))
        os.write(self.fd, struct.pack('llHHi', 0, 0, 0, 0, 0))
        self.events += 1

    def _run(self):
        target = time.monotonic()
        while not self.closed:
            if self.enabled:
                angle = (time.monotonic() - self.started) * 2 * math.pi / self.period
                x = round(self.radius * math.cos(angle))
                y = round(self.radius * math.sin(angle))
                self.move(x - self.x, y - self.y)
                self.x, self.y = x, y
            target += 1 / self.rate
            time.sleep(max(0, target - time.monotonic()))
            if target < time.monotonic() - 1 / self.rate:
                target = time.monotonic()

    def close(self):
        self.closed = True
        self.thread.join(timeout=2)
        fcntl.ioctl(self.fd, 0x5502)  # UI_DEV_DESTROY
        os.close(self.fd)


if __name__ == '__main__':
    import argparse
    import signal
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rate', type=float, default=240)
    parser.add_argument('--radius', type=float, default=160)
    parser.add_argument('--period', type=float, default=2)
    parser.add_argument('--duration', type=float, default=30)
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error('duration must be positive')
    def interrupted(*_):
        raise KeyboardInterrupt
    for sig in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(sig, interrupted)
    mouse = Mouse(rate=args.rate, radius=args.radius, period=args.period)
    try:
        mouse.enabled = True
        time.sleep(args.duration)
    except KeyboardInterrupt:
        pass
    finally:
        mouse.close()
