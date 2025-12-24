"""Entry point for running cross-dock scenarios and exporting the resulting data."""

import copy

from cross_dock_sim import (
    build_baseline_config,
    export_results_to_csv,
    run_parameter_sweep,
    run_scenario,
)


def main() -> None:
    """Run the sample scenarios and capture their outputs."""
    baseline = build_baseline_config()
    baseline_results = run_scenario(baseline, "Baseline")
    export_results_to_csv(baseline_results, baseline, "Baseline")

    stress = copy.deepcopy(baseline)
    stress.mean_arrival_rate_per_min = 18
    stress_results = run_scenario(stress, "Stress (higher arrivals)")
    export_results_to_csv(stress_results, stress, "Stress (higher arrivals)")

    fix = copy.deepcopy(stress)
    fix.n_sort_workers = 6
    fix_results = run_scenario(fix, "Fix (more sort labor)")
    export_results_to_csv(fix_results, fix, "Fix (more sort labor)")

    run_parameter_sweep(
        baseline,
        sort_workers=[3, 4, 5, 6],
        conveyor_rates=[30, 45, 60],
        inbound_workers=[2, 3, 4],
    )


if __name__ == "__main__":
    main()
