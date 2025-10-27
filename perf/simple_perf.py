import time
from typing import Any, Callable


def perf_indicator(label: str, unit: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to print a short performance indicator for an operation.
    The wrapped function should return either:
      - result (any)           -> assumes count = 1
      - (result, count: int)   -> uses provided count for throughput
    """

    def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def _wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_s = time.perf_counter() - t0

            count = 1
            payload = result
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], (int, float)):
                payload, count = result  # unpack (payload, count)

            rate_per_min = (count / elapsed_s) * 60.0 if elapsed_s > 0 else float("inf")
            print(
                f"[{label}] {int(count)} {unit} in {elapsed_s*1000:.2f} ms ({rate_per_min:.1f} {unit}/min)"
            )
            return payload

        return _wrapper

    return _decorator


