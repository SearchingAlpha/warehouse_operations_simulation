You are a senior operations-research engineer. Build a super-simple, flexible simulation of a cross-dock facility in Python.

Context:
- The facility receives parcels, does a minimal inbound handling, then sorts/consolidates them into outbound “routes” (lanes/trucks).
- Goal: get rough numbers for (1) normal throughput capacity, (2) bottlenecks, (3) shift sizing (people + conveyor capacity).
- Keep it intentionally simple: we want directional insight, not perfect realism.

Hard requirements:
1) Super simple: few processes, few parameters, minimal dependencies.
2) Flexible: changing inputs (people, conveyors, service rates, arrival rates, number of routes, shift length) must be easy (config dict / dataclass).
3) Clear outputs: throughput, WIP/queue sizes, utilization, and bottleneck identification.
4) Provide runnable code + an example run.

Modeling approach (preferred):
- Use a discrete-event simulation (DES) with SimPy.
- Time unit: minutes.
- Entities: parcels.
- Resources:
  - Inbound unloaders (workers) OR inbound dock capacity
  - Conveyor capacity as a rate limit (items/min) or as a resource with service time
  - Sorters (workers) who scan and divert parcels to routes
  - Per-route staging lane capacity (optional simple cap) and route “loaders” (workers) OR fixed route dispatch cadence
- Flow (keep minimal):
  Arrivals -> Inbound queue -> Unload/induct -> Conveyor -> Sorting -> Route queue -> Load/outbound
- Routing:
  Each parcel gets a destination route sampled from a configurable probability distribution across N routes.

Simplifications (explicitly state them):
- Ignore travel distances, breaks, complex labor rules.
- Use simple service times (constant or exponential) with configurable mean.
- Optional: allow deterministic arrivals or Poisson arrivals.

Inputs (must be in one config object):
- sim_duration_min, warmup_min (optional)
- arrival process: type (“poisson” or “schedule”), mean_rate_per_min or list of (start,end,rate)
- n_routes and route_probabilities (must sum to 1)
- resources:
  - n_inbound_workers, inbound_service_time_mean
  - conveyor_rate_items_per_min (or conveyor_service_time_mean)
  - n_sort_workers, sort_service_time_mean
  - per_route_n_load_workers (either 1 value for all routes or list per route)
  - load_service_time_mean
- optional capacities:
  - inbound_queue_cap, per_route_queue_cap
- random_seed

Outputs (print + return as dict):
- total_arrived, total_completed, completion_rate_per_hour
- average and max queue lengths for each buffer (inbound, pre-sort, per-route)
- average and peak WIP
- average utilization of each resource (inbound workers, sort workers, loaders; conveyor if modeled as resource)
- lead time (time in system): mean/p50/p90
- bottleneck heuristic:
  - identify the station(s) with highest utilization and/or growing queues.

Experiments / scenarios:
- Provide functions to run a scenario and to run a small parameter sweep:
  - vary n_sort_workers, conveyor_rate_items_per_min, and n_inbound_workers
  - output a simple table ranking scenarios by throughput and average lead time.

Code structure:
- One file, readable, heavily commented.
- Use dataclasses for config.
- No external files. Avoid heavy plotting; optional simple matplotlib chart is okay but not required.
- Include a “main” section that runs:
  a) baseline scenario
  b) one stress scenario (higher arrivals)
  c) one bottleneck-fix scenario (increase the constrained resource)

Quality checks:
- Validate route probabilities.
- Handle both scalar and per-route list inputs for loaders.
- Keep everything deterministic under seed.

Deliverables:
1) Explanation of assumptions and how to interpret outputs.
2) The full Python code using SimPy.
3) Example output from running baseline + two scenarios (show representative printed metrics).

If SimPy is not available, fall back to a minimal event loop implementation (but SimPy is strongly preferred).
