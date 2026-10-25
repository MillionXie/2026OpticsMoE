"""Explicit bounded lab waiting windows; GPU point limit stays two hours."""
import math
import time


def valid_window(deadline,queue_seconds,now=None):
    now=time.time() if now is None else now
    return (queue_seconds in (7200,14400,21600) and math.isfinite(deadline)
            and 0<deadline-now<=queue_seconds)


def point_deadline(queue_deadline,now=None):
    now=time.time() if now is None else now
    return min(queue_deadline,now+7199)
