"""Opt-in public-test plateau control; never changes model structure or best selection."""
import math


class PlateauController:
    def __init__(self, patience=3, min_delta=0.0001, factor=0.5,
                 max_reductions=2, min_epoch=6):
        if patience < 1 or min_epoch < 1 or max_reductions < 0:
            raise ValueError("Invalid plateau counts")
        if not math.isfinite(min_delta) or min_delta < 0 or not 0 < factor < 1:
            raise ValueError("Invalid plateau threshold/factor")
        self.patience, self.min_delta, self.factor = patience, min_delta, factor
        self.max_reductions, self.min_epoch = max_reductions, min_epoch
        self.best = -math.inf
        self.bad_tests = self.reductions = 0
        self.multiplier = 1.0
        self.stopped = False

    def observe(self, epoch, cc):
        if not math.isfinite(cc):
            raise ValueError("Non-finite public-test CC")
        if self.stopped:
            return "stop"
        if cc > self.best + self.min_delta:
            self.best, self.bad_tests = cc, 0
            return "improved"
        # Warmstart anchors best; grace-period tests do not consume patience.
        if epoch < self.min_epoch:
            return "grace"
        self.bad_tests += 1
        if self.bad_tests < self.patience:
            return "wait"
        self.bad_tests = 0
        if self.reductions < self.max_reductions:
            self.reductions += 1
            self.multiplier *= self.factor
            return "reduce_lr"
        self.stopped = True
        return "stop"

    def scale_epoch_rates(self, optimizer):
        """Call once AFTER staged_epoch recomputes base rates (no compounding)."""
        for group in optimizer.param_groups:
            group["lr"] *= self.multiplier
