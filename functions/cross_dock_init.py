"""Module holding tunable initialization helpers."""

from .cross_dock_core import SimulationConfig


def build_baseline_config() -> SimulationConfig:
    """Produce the default configuration for example runs."""
    return SimulationConfig(
        sim_duration_min=480,  # minutes (60-1440)
        warmup_min=60,  # minutes (0-120)
        arrival_type="poisson",  # "poisson" or "schedule"
        mean_arrival_rate_per_min=12,  # orders per minute (1-30)
        n_routes=6,  # number of downstream routes (>=1)
        route_probabilities=[0.15, 0.2, 0.15, 0.2, 0.2, 0.1],  # length == n_routes, sum to 1.0
        n_inbound_workers=3,  # inbound workers (>=1)
        inbound_service_time_mean=0.6,  # minutes per inbound worker (0-5)
        conveyor_rate_items_per_min=45,  # items/min processed by conveyor (10-60)
        n_sort_workers=4,  # sort labor headcount (>=1)
        sort_service_time_mean=0.7,  # minutes at sort (0.1-2)
        per_route_n_load_workers=1,  # per-route load workers (scalar or list matching n_routes)
        load_service_time_mean=1.8,  # minutes to load (0.5-3)
        per_route_queue_cap=50,  # queue capacity per route (None for infinite)
        random_seed=7,  # deterministic simulation seed
    )


__all__ = ["build_baseline_config"]
