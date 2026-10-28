"""Serialize a worker's existing foreground/background MPS check transaction.

No command retry, exception suppression, cached snapshot or relaxed guard.
The original check body (including source/foreign/client checks) is unchanged.
"""
import threading


def serialized_guard_class(original):
    if getattr(original, '_serialized_mps_check', False):
        return original

    class SerializedMPSGuard(original):
        _serialized_mps_check = True

        def __init__(self, *args, **kwargs):
            self._mps_check_lock = threading.RLock()
            super().__init__(*args, **kwargs)

        def check(self):
            with self._mps_check_lock:
                return super().check()

    return SerializedMPSGuard
