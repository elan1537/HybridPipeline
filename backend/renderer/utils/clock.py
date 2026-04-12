"""
Monotonic clock utility for consistent timestamp measurement.

Provides a monotonic millisecond clock that is epoch-aligned with time.time()
so that existing cross-clock math (client ↔ server) continues to work,
while interval measurements are guaranteed monotonic (immune to NTP adjustments).

Usage:
    from renderer.utils.clock import monotonic_ms
    ts = monotonic_ms()  # returns float, milliseconds
"""

import time

_origin_ns = time.monotonic_ns()
_origin_wall_ms = time.time() * 1000.0


def monotonic_ms() -> float:
    """CPU-side monotonic milliseconds, epoch-aligned with time.time().

    This is a CPU clock — it does NOT wait for GPU work to complete.
    For GPU-accurate boundaries, call torch.cuda.synchronize() before
    reading this clock.
    """
    return (time.monotonic_ns() - _origin_ns) / 1e6 + _origin_wall_ms
