import time
from collections import deque
from contextlib import contextmanager
from threading import Lock

_events = deque(maxlen=60)
_lock = Lock()


def record(name, ms, backend, detail=""):
    with _lock:
        _events.appendleft(
            {"time": time.strftime("%H:%M:%S"), "name": name, "ms": round(ms, 1), "backend": backend, "detail": detail}
        )


@contextmanager
def timed(name, backend, detail=""):
    start = time.perf_counter()
    try:
        yield
    finally:
        record(name, (time.perf_counter() - start) * 1000, backend, detail)


def recent():
    with _lock:
        return list(_events)
