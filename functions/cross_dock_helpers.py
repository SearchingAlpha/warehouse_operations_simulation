"""Helper routines for running scenarios, sweeps, and persisting output."""

from __future__ import annotations

import copy
import csv
import itertools
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .cross_dock_core import CrossDockSimulation, SimulationConfig


def print_summary(results: Dict[str, float]) -> None:
    print(
        f"Arrived={results['total_arrived']} | Completed={results['total_completed']} | "
        f"Throughput={results['completion_rate_per_hour']:.1f}/hr"
    )
    print(
        f"Queues avg/max -> inbound: {results['avg_inbound_queue']:.2f}/{results['max_inbound_queue']:.0f}, "
        f"conveyor: {results['avg_conveyor_queue']:.2f}/{results['max_conveyor_queue']:.0f}, "
        f"pre-sort: {results['avg_sort_queue']:.2f}/{results['max_sort_queue']:.0f}, "
        f"route: {results['avg_route_queue']:.2f}/{results['max_route_queue']:.0f}"
    )
    print(
        f"Utilization -> inbound: {results['util_inbound']:.2f}, conveyor: {results['util_conveyor']:.2f}, "
        f"sort: {results['util_sort']:.2f}, loaders(max): {max(results['util_loaders'], default=0):.2f}"
    )
    print(
        f"Lead time mean/p50/p90 (min): {results['lead_time_mean']:.2f}/"
        f"{results['lead_time_p50']:.2f}/{results['lead_time_p90']:.2f}"
    )
    print(
        f"Bottleneck hint -> highest util: {results['bottleneck']['highest_utilization']}, "
        f"queues: {results['bottleneck']['queues_showing_growth']}"
    )


def run_scenario(config: SimulationConfig, label: str) -> Dict[str, float]:
    sim = CrossDockSimulation(config)
    results = sim.run()
    print(f"\n--- Scenario: {label} ---")
    print_summary(results)
    return results


def run_parameter_sweep(
    base_config: SimulationConfig,
    sort_workers: Sequence[int],
    conveyor_rates: Sequence[float],
    inbound_workers: Sequence[int],
) -> List[Tuple[str, Dict[str, float]]]:
    scenarios: List[Tuple[str, Dict[str, float]]] = []
    for sw, cr, iw in itertools.product(sort_workers, conveyor_rates, inbound_workers):
        cfg = copy.deepcopy(base_config)
        cfg.n_sort_workers = sw
        cfg.conveyor_rate_items_per_min = cr
        cfg.conveyor_service_time_mean = None  # ensure rate-derived
        cfg.n_inbound_workers = iw
        cfg.random_seed = base_config.random_seed  # deterministic across scenarios
        label = f"sort={sw}, conv={cr}/min, inbound={iw}"
        res = CrossDockSimulation(cfg).run()
        scenarios.append((label, res))

    scenarios.sort(key=lambda x: (-x[1]["completion_rate_per_hour"], x[1]["lead_time_mean"]))

    print("\n=== Parameter sweep (ranked) ===")
    for label, res in scenarios:
        print(
            f"{label:30s} | throughput={res['completion_rate_per_hour']:.1f}/hr | "
            f"lead_mean={res['lead_time_mean']:.2f} min | bottleneck={res['bottleneck']['highest_utilization']}"
        )
    return scenarios


def _serialize_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value)
    return value


def _sanitize_label(label: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in label)
    return cleaned.strip("_") or "scenario"


def export_results_to_csv(
    results: Dict[str, float],
    config: SimulationConfig,
    label: str,
    directory: Optional[Path] = None,
) -> Path:
    """Persist scenario results and config metadata for later analysis."""
    directory = Path(directory or Path("data"))
    directory.mkdir(parents=True, exist_ok=True)
    config_dict = asdict(config)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    fieldnames = ["timestamp", "scenario_label"]
    fieldnames += [f"config_{key}" for key in config_dict.keys()]
    fieldnames += [f"metric_{key}" for key in results.keys()]

    row: Dict[str, object] = {
        "timestamp": timestamp,
        "scenario_label": label,
    }
    for key, value in config_dict.items():
        row[f"config_{key}"] = _serialize_value(value)
    for key, value in results.items():
        row[f"metric_{key}"] = _serialize_value(value)

    label_token = _sanitize_label(label)
    filename = directory / f"data_{timestamp}_{label_token}.csv"
    with filename.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    return filename


__all__ = [
    "print_summary",
    "run_scenario",
    "run_parameter_sweep",
    "export_results_to_csv",
]
