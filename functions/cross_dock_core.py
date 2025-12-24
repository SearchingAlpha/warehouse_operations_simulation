"""Core simulation definitions for the cross-docking model."""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# Prefer SimPy; if unavailable, use a minimal built-in discrete-event engine.
try:  # pragma: no cover - tiny import shim
    import simpy  # type: ignore
    SIMPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    SIMPY_AVAILABLE = False
    import heapq

    class Event:
        def __init__(self, env):
            self.env = env
            self.callbacks: List = []
            self.triggered = False
            self.value = None

        def succeed(self, value=None):
            if self.triggered:
                return self
            self.triggered = True
            self.value = value
            for cb in list(self.callbacks):
                cb(self)
            return self


    class Timeout(Event):
        def __init__(self, env, delay):
            super().__init__(env)
            self.delay = delay


    class Process(Event):
        def __init__(self, env, generator):
            super().__init__(env)
            self._generator = generator
            self._pending = None
            env._schedule(env.now, self._step)

        def _resume(self, event: Event):
            self._pending = event.value
            self.env._schedule(self.env.now, self._step)

        def _step(self):
            if self.triggered:
                return
            try:
                if self._pending is None:
                    yielded = next(self._generator)
                else:
                    yielded = self._generator.send(self._pending)
                    self._pending = None
            except StopIteration as exc:
                self.succeed(getattr(exc, "value", None))
                return

            self.env._handle_yield(self, yielded)


    class Store:
        def __init__(self, env, capacity=None):
            self.env = env
            self.capacity = capacity
            self.items: List = []
            self.get_waiters: List[Event] = []
            self.put_waiters: List[Tuple[Event, object]] = []

        def put(self, item):
            ev = Event(self.env)
            if self.capacity is not None and len(self.items) >= self.capacity:
                self.put_waiters.append((ev, item))
            else:
                if self.get_waiters:
                    getter = self.get_waiters.pop(0)
                    getter.succeed(item)
                else:
                    self.items.append(item)
                ev.succeed(None)
            return ev

        def get(self):
            ev = Event(self.env)
            if self.items:
                item = self.items.pop(0)
                ev.succeed(item)
                self._flush_put_waiters()
            else:
                self.get_waiters.append(ev)
            return ev

        def _flush_put_waiters(self):
            # If capacity allows, pull waiting puts into the store
            while self.put_waiters and (self.capacity is None or len(self.items) < self.capacity):
                put_ev, item = self.put_waiters.pop(0)
                if self.get_waiters:
                    getter = self.get_waiters.pop(0)
                    getter.succeed(item)
                else:
                    self.items.append(item)
                put_ev.succeed(None)


    class Environment:
        def __init__(self):
            self.now = 0.0
            self._queue: List[Tuple[float, int, Callable]] = []
            self._counter = 0

        def _schedule(self, time: float, callback: Callable):
            self._counter += 1
            heapq.heappush(self._queue, (time, self._counter, callback))

        def _handle_yield(self, proc: Process, yielded):
            if isinstance(yielded, Event):
                if yielded.triggered:
                    proc._resume(yielded)
                else:
                    yielded.callbacks.append(proc._resume)
            elif isinstance(yielded, (int, float)):
                self.timeout(float(yielded)).callbacks.append(proc._resume)
            else:  # immediate
                proc._resume(Event(self))

        def process(self, generator):
            return Process(self, generator)

        def timeout(self, delay: float):
            ev = Timeout(self, delay)
            self._schedule(self.now + delay, lambda: ev.succeed(None))
            return ev

        def run(self, until):
            while self._queue and self.now < until:
                time, _, callback = heapq.heappop(self._queue)
                if time > until:
                    self.now = until
                    break
                self.now = time
                callback()
            self.now = until

    # Expose fallback module-like object
    class _SimPyModule:
        Environment = Environment
        Event = Event
        Timeout = Timeout
        Process = Process
        Store = Store

    simpy = _SimPyModule()


@dataclass
class ArrivalSchedule:
    """Piecewise constant arrival schedule.

    Each tuple is (start_minute, end_minute, rate_per_minute) with start inclusive,
    end exclusive. Use ascending, non-overlapping intervals.
    """

    segments: List[Tuple[float, float, float]]

    def rate_at(self, t: float) -> float:
        for start, end, rate in self.segments:
            if start <= t < end:
                return rate
        return 0.0


@dataclass
class SimulationConfig:
    # core timing
    sim_duration_min: float = 480
    warmup_min: float = 0

    # arrival
    arrival_type: str = "poisson"  # "poisson" or "schedule"
    mean_arrival_rate_per_min: float = 10.0
    arrival_schedule: Optional[ArrivalSchedule] = None

    # routing
    n_routes: int = 5
    route_probabilities: List[float] = field(default_factory=lambda: [0.2] * 5)

    # resources
    n_inbound_workers: int = 3
    inbound_service_time_mean: float = 0.5

    conveyor_rate_items_per_min: float = 40.0
    # If provided, overrides rate-based service time
    conveyor_service_time_mean: Optional[float] = None

    n_sort_workers: int = 4
    sort_service_time_mean: float = 0.6

    per_route_n_load_workers: Sequence[int] | int = 1
    load_service_time_mean: float = 1.5

    # capacities (None -> infinite)
    inbound_queue_cap: Optional[int] = None
    per_route_queue_cap: Optional[int] = None

    random_seed: int = 42

    def validate(self) -> None:
        if abs(sum(self.route_probabilities) - 1.0) > 1e-6:
            raise ValueError("route_probabilities must sum to 1")
        if len(self.route_probabilities) != self.n_routes:
            raise ValueError("route_probabilities length must equal n_routes")
        if self.arrival_type not in {"poisson", "schedule"}:
            raise ValueError("arrival_type must be 'poisson' or 'schedule'")
        if self.arrival_type == "schedule" and not self.arrival_schedule:
            raise ValueError("arrival_schedule required when arrival_type='schedule'")

    def loader_counts(self) -> List[int]:
        if isinstance(self.per_route_n_load_workers, int):
            return [self.per_route_n_load_workers] * self.n_routes
        if len(self.per_route_n_load_workers) != self.n_routes:
            raise ValueError("per_route_n_load_workers must be scalar or length n_routes")
        return list(self.per_route_n_load_workers)


class LevelMonitor:
    """Tracks time-averaged levels (queues/WIP) and max values."""

    def __init__(self, env: simpy.Environment, name: str):
        self.env = env
        self.name = name
        self.start_time = env.now
        self.last_time = 0.0
        self.last_value = 0.0
        self.area = 0.0
        self.max_value = 0.0

    def record(self, value: float) -> None:
        now = self.env.now
        dt = now - self.last_time
        self.area += self.last_value * dt
        self.last_time = now
        self.last_value = value
        self.max_value = max(self.max_value, value)

    def reset(self) -> None:
        # Reset area accounting at current time; keep current value so continuity holds.
        self.area = 0.0
        self.start_time = self.env.now
        self.last_time = self.env.now
        self.max_value = self.last_value

    def average(self, horizon: Optional[float] = None) -> float:
        elapsed = self.env.now - self.start_time if horizon is None else horizon
        if elapsed <= 0:
            return 0.0
        # Add area from last point to now using current value.
        adjusted_area = self.area + self.last_value * (self.env.now - self.last_time)
        return adjusted_area / elapsed


class UtilizationTracker:
    """Tracks busy time for a multi-server station."""

    def __init__(self, env: simpy.Environment, n_servers: int, name: str):
        self.env = env
        self.n_servers = n_servers
        self.name = name
        self.busy_time = [0.0 for _ in range(n_servers)]
        self.current_start = [None for _ in range(n_servers)]

    def start(self, idx: int) -> None:
        self.current_start[idx] = self.env.now

    def stop(self, idx: int) -> None:
        start = self.current_start[idx]
        if start is None:
            return
        self.busy_time[idx] += self.env.now - start
        self.current_start[idx] = None

    def utilization(self, horizon: float) -> float:
        if horizon <= 0:
            return 0.0
        return sum(self.busy_time) / (self.n_servers * horizon)


class CrossDockSimulation:
    def __init__(self, config: SimulationConfig):
        config.validate()
        random.seed(config.random_seed)
        self.config = config
        self.env = simpy.Environment()
        self.metrics: Dict[str, float] = {}

        # Monitors
        self.inbound_queue_monitor = LevelMonitor(self.env, "inbound_queue")
        self.conveyor_queue_monitor = LevelMonitor(self.env, "conveyor_queue")
        self.sort_queue_monitor = LevelMonitor(self.env, "sort_queue")
        self.route_queue_monitors = [LevelMonitor(self.env, f"route_{i}_queue") for i in range(config.n_routes)]
        self.wip_monitor = LevelMonitor(self.env, "wip")

        # Utilization trackers
        self.inbound_util = UtilizationTracker(self.env, config.n_inbound_workers, "inbound")
        conveyor_servers = 1  # single conveyor lane approximation
        self.conveyor_util = UtilizationTracker(self.env, conveyor_servers, "conveyor")
        self.sort_util = UtilizationTracker(self.env, config.n_sort_workers, "sort")
        self.loader_utils = [UtilizationTracker(self.env, c, f"route_{i}_load") for i, c in enumerate(config.loader_counts())]

        # Stores
        self.inbound_queue = simpy.Store(self.env, capacity=config.inbound_queue_cap)
        self.conveyor_queue = simpy.Store(self.env)
        self.sort_queue = simpy.Store(self.env)
        self.route_queues = [simpy.Store(self.env, capacity=config.per_route_queue_cap) for _ in range(config.n_routes)]

        # Data collection
        self.completed_times: List[float] = []
        self.arrival_count = 0
        self.completion_count = 0
        self.dropped_count = 0

    # ------------------------- arrival process
    def arrival_process(self):
        cfg = self.config
        env = self.env

        def next_interarrival() -> float:
            if cfg.arrival_type == "poisson":
                return random.expovariate(cfg.mean_arrival_rate_per_min)
            # scheduled arrival
            rate = cfg.arrival_schedule.rate_at(env.now)
            return math.inf if rate <= 0 else random.expovariate(rate)

        while env.now < cfg.sim_duration_min:
            dt = next_interarrival()
            if math.isinf(dt):
                # No more arrivals in this segment; advance slightly to exit loop if beyond duration
                yield env.timeout(1)
                continue
            yield env.timeout(dt)
            if env.now > cfg.sim_duration_min:
                break
            route = random.choices(range(cfg.n_routes), weights=cfg.route_probabilities, k=1)[0]
            item = {"arrival_time": env.now, "route": route}
            self.arrival_count += 1
            self.wip_monitor.record(self.wip_monitor.last_value + 1)
            # Put into inbound queue (blocks if full)
            yield self.inbound_queue.put(item)
            self.inbound_queue_monitor.record(len(self.inbound_queue.items))

    # ------------------------- station workers
    def start_inbound_workers(self):
        for idx in range(self.config.n_inbound_workers):
            self.env.process(self.inbound_worker(idx))

    def inbound_worker(self, idx: int):
        while True:
            item = yield self.inbound_queue.get()
            self.inbound_queue_monitor.record(len(self.inbound_queue.items))
            self.inbound_util.start(idx)
            mean = self.config.inbound_service_time_mean
            service_time = random.expovariate(1 / mean) if mean > 0 else 0
            yield self.env.timeout(service_time)
            self.inbound_util.stop(idx)
            yield self.conveyor_queue.put(item)
            self.conveyor_queue_monitor.record(len(self.conveyor_queue.items))

    def start_conveyor(self):
        # Single server approximating throughput rate
        self.env.process(self.conveyor_server(0))

    def conveyor_server(self, idx: int):
        mean = self.config.conveyor_service_time_mean
        if mean is None:
            mean = 1 / self.config.conveyor_rate_items_per_min if self.config.conveyor_rate_items_per_min > 0 else 0
        while True:
            item = yield self.conveyor_queue.get()
            self.conveyor_queue_monitor.record(len(self.conveyor_queue.items))
            self.conveyor_util.start(idx)
            service_time = random.expovariate(1 / mean) if mean > 0 else 0
            yield self.env.timeout(service_time)
            self.conveyor_util.stop(idx)
            yield self.sort_queue.put(item)
            self.sort_queue_monitor.record(len(self.sort_queue.items))

    def start_sort_workers(self):
        for idx in range(self.config.n_sort_workers):
            self.env.process(self.sort_worker(idx))

    def sort_worker(self, idx: int):
        while True:
            item = yield self.sort_queue.get()
            self.sort_queue_monitor.record(len(self.sort_queue.items))
            self.sort_util.start(idx)
            mean = self.config.sort_service_time_mean
            service_time = random.expovariate(1 / mean) if mean > 0 else 0
            yield self.env.timeout(service_time)
            self.sort_util.stop(idx)
            route = item["route"]
            yield self.route_queues[route].put(item)
            self.route_queue_monitors[route].record(len(self.route_queues[route].items))

    def start_loaders(self):
        for route_idx, worker_count in enumerate(self.config.loader_counts()):
            for worker_idx in range(worker_count):
                self.env.process(self.loader(route_idx, worker_idx))

    def loader(self, route_idx: int, worker_idx: int):
        util = self.loader_utils[route_idx]
        store = self.route_queues[route_idx]
        monitor = self.route_queue_monitors[route_idx]
        while True:
            item = yield store.get()
            monitor.record(len(store.items))
            util.start(worker_idx)
            mean = self.config.load_service_time_mean
            service_time = random.expovariate(1 / mean) if mean > 0 else 0
            yield self.env.timeout(service_time)
            util.stop(worker_idx)
            self.completion_count += 1
            self.wip_monitor.record(self.wip_monitor.last_value - 1)
            lead_time = self.env.now - item["arrival_time"]
            self.completed_times.append(lead_time)

    # ------------------------- warmup management
    def _apply_warmup_reset(self):
        if self.config.warmup_min <= 0:
            return

        def reset_monitors():
            yield self.env.timeout(self.config.warmup_min)
            self.inbound_queue_monitor.reset()
            self.conveyor_queue_monitor.reset()
            self.sort_queue_monitor.reset()
            for m in self.route_queue_monitors:
                m.reset()
            self.wip_monitor.reset()
            # Utilization trackers: drop busy time accrued before warmup
            for tracker in [self.inbound_util, self.conveyor_util, self.sort_util, *self.loader_utils]:
                tracker.busy_time = [0.0 for _ in tracker.busy_time]
                tracker.current_start = [self.env.now if start is not None else None for start in tracker.current_start]
            # Drop pre-warmup counts
            self.arrival_count = 0
            self.completion_count = 0
            self.completed_times = []

        self.env.process(reset_monitors())

    # ------------------------- run
    def run(self) -> Dict[str, float]:
        # kick off processes
        self.env.process(self.arrival_process())
        self.start_inbound_workers()
        self.start_conveyor()
        self.start_sort_workers()
        self.start_loaders()
        self._apply_warmup_reset()

        self.env.run(until=self.config.sim_duration_min)

        horizon = self.config.sim_duration_min - self.config.warmup_min
        avg_inbound_q = self.inbound_queue_monitor.average()
        avg_conveyor_q = self.conveyor_queue_monitor.average()
        avg_sort_q = self.sort_queue_monitor.average()
        avg_route_q = [m.average() for m in self.route_queue_monitors]
        avg_route_q_total = sum(avg_route_q) / len(avg_route_q) if avg_route_q else 0

        avg_wip = self.wip_monitor.average()

        util_inbound = self.inbound_util.utilization(horizon)
        util_conveyor = self.conveyor_util.utilization(horizon)
        util_sort = self.sort_util.utilization(horizon)
        util_loaders = [u.utilization(horizon) for u in self.loader_utils]

        lead_mean = statistics.mean(self.completed_times) if self.completed_times else 0
        lead_p50 = statistics.median(self.completed_times) if self.completed_times else 0
        try:
            lead_p90 = statistics.quantiles(self.completed_times, n=10)[8]
        except Exception:
            lead_p90 = lead_mean

        completion_rate_per_hour = self.completion_count / (horizon / 60) if horizon > 0 else 0

        # Bottleneck heuristic: highest utilization + queues growing
        util_pairs = {
            "inbound": util_inbound,
            "conveyor": util_conveyor,
            "sort": util_sort,
            "loader_max": max(util_loaders) if util_loaders else 0,
        }
        top_util = max(util_pairs.items(), key=lambda x: x[1])
        queue_flags = []
        if self.inbound_queue_monitor.max_value > 1:
            queue_flags.append("inbound")
        if self.conveyor_queue_monitor.max_value > 1:
            queue_flags.append("conveyor")
        if self.sort_queue_monitor.max_value > 1:
            queue_flags.append("pre-sort")
        for idx, monitor in enumerate(self.route_queue_monitors):
            if monitor.max_value > 1:
                queue_flags.append(f"route_{idx}")

        bottleneck_hint = {
            "highest_utilization": top_util,
            "queues_showing_growth": queue_flags,
        }

        results = {
            "total_arrived": self.arrival_count,
            "total_completed": self.completion_count,
            "completion_rate_per_hour": completion_rate_per_hour,
            "avg_inbound_queue": avg_inbound_q,
            "max_inbound_queue": self.inbound_queue_monitor.max_value,
            "avg_conveyor_queue": avg_conveyor_q,
            "max_conveyor_queue": self.conveyor_queue_monitor.max_value,
            "avg_sort_queue": avg_sort_q,
            "max_sort_queue": self.sort_queue_monitor.max_value,
            "avg_route_queue": avg_route_q_total,
            "max_route_queue": max([m.max_value for m in self.route_queue_monitors], default=0),
            "avg_wip": avg_wip,
            "max_wip": self.wip_monitor.max_value,
            "util_inbound": util_inbound,
            "util_conveyor": util_conveyor,
            "util_sort": util_sort,
            "util_loaders": util_loaders,
            "lead_time_mean": lead_mean,
            "lead_time_p50": lead_p50,
            "lead_time_p90": lead_p90,
            "bottleneck": bottleneck_hint,
        }
        self.metrics = results
        return results


__all__ = [
    "ArrivalSchedule",
    "SimulationConfig",
    "CrossDockSimulation",
]
