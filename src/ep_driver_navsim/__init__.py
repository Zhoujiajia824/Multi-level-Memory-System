"""NAVSIM v2.2 integration. SDK modules stay importable under Python 3.9.

The numerical planner runs in a separate Python 3.12 process; importing this
package must not import nine_grid or alter the official evaluator.
"""

OFFICIAL_TAG = "v2.2"
OFFICIAL_COMMIT = "359c7f72304bfa8273e754224a213d3751bd2340"
INPUT_PROTOCOL = "ep-driver-navsim-privileged-history-1.1"
LEGACY_INPUT_PROTOCOL = "ep-driver-navsim-privileged-history-1.0"
INPUT_CONDITION = "privileged_current_history_tracks_hd_map_route_traffic_lights"
