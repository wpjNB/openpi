"""Admission for periodic reservations on one non-preemptive worker."""

from collections.abc import Sequence
import dataclasses
import math


@dataclasses.dataclass(frozen=True)
class Reservation:
    hz: float
    max_latency: float

    @property
    def period(self) -> float:
        return 1 / self.hz

    @property
    def deadline(self) -> float:
        return min(self.max_latency, self.period)


def can_reserve(reservations: Sequence[Reservation], *, cost: float, max_utilization: float) -> bool:
    """Conservative non-preemptive EDF demand check for constrained deadlines.

    Charge the entire collect budget plus inference to each job, even if prefetch
    overlaps some of that cost. Account for one job already in service. This is
    budget-based admission: runtime overruns can still cause reported misses.
    """
    if not reservations:
        return True
    if any(task.deadline < cost for task in reservations):
        return False
    utilization = math.fsum(task.hz * cost for task in reservations)
    if utilization > max_utilization:
        return False
    if len(reservations) == 1:
        return True

    # For t above this bound, sum(dbf(t)) + cost <= t by the linear
    # demand bound. Check only deadlines up to the bound. A bounded
    # iteration count conservatively rejects extreme period ratios.
    if utilization >= 1:
        return False
    intercept = math.fsum(cost * (1 - task.deadline * task.hz) for task in reservations)
    bound = (cost + intercept) / (1 - utilization)
    for task in reservations:
        points = math.floor((bound - task.deadline) * task.hz) + 1
        if points > 4096:
            return False
        for i in range(points):
            instant = task.deadline + i * task.period
            demand = sum(
                max(0, math.floor((instant - other.deadline) * other.hz + 1e-10) + 1) for other in reservations
            )
            blocking = int(any(other.deadline > instant for other in reservations))
            if (demand + blocking) * cost > instant:
                return False
    return True
