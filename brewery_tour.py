"""
Optimal Brewery Tour Route Planning (TSP)
=========================================

Research question:
    Given a set of breweries in a selected region, what is the shortest
    possible round-trip route visiting every brewery exactly once and
    returning to the starting point? (Travelling Salesman Problem)

Approach:
    1. Load the cleaned brewery dataset and filter for a chosen region
       (by state/province and optionally by country)
    2. Drop any breweries without GPS coordinates — they cannot be
       placed on a route
    3. Build a pairwise distance matrix using the Haversine formula,
       which calculates great-circle distances (straight-line km) between
       two points on the Earth's surface given their lat/lon coordinates.
       Plain Euclidean distance would be inaccurate here because the
       Earth is a sphere, not a flat plane.
    4. Solve the TSP:
         - Exact solution  (Dynamic Programming, Held-Karp algorithm)
           for small n (≤ EXACT_MAX_N). Time complexity is O(n² x 2ⁿ),
           so it becomes infeasible quickly — above ~13 stops it takes
           too long to run.
         - Heuristic solution (Local Search / 2-opt) for larger n.
           Not guaranteed to find the global optimum, but typically
           comes very close in a fraction of the time.
    5. Output:
         - Console: ordered list of stops with distance to next stop
         - CSV: full route table
         - HTML: interactive map with stop markers and route line

Why these two solvers?
    The exact DP solver guarantees the globally shortest route but
    scales exponentially with n. The local search heuristic scales
    polynomially and is good enough for planning real trips, where road
    distances and travel time matter more than the theoretical optimum
    on straight-line distances anyway.

Installation:
    pip install python-tsp plotly pandas numpy
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_local_search
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

INPUT_CSV = "breweries_clean.csv"
OUTPUT_ROUTE_CSV = "brewery_tour_route.csv"
OUTPUT_MAP = "map_brewery_tour.html"

# Region filter — change these to plan a tour in a different area.
# FILTER_COLUMN can be "state_province" or "city".
# Set FILTER_COUNTRY to None to skip the country filter.
FILTER_COLUMN = "city"
FILTER_VALUE = "Berlin"
FILTER_COUNTRY = "Germany"

OUTPUT_DIR = Path("tour_output") / FILTER_VALUE
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Maximum number of stops for the exact DP solver.
# Above this threshold the heuristic solver is used instead.
# Rule of thumb: keep this at 12–13 to avoid multi-minute runtimes.
EXACT_MAX_N = 12

EARTH_RADIUS_KM = 6371.0

# ── Haversine distance matrix ─────────────────────────────────────────────────


def haversine_distance_matrix(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Compute an n x n matrix of pairwise great-circle distances in km.

    The Haversine formula accounts for the curvature of the Earth.
    Using plain Euclidean distance on lat/lon degree values would
    introduce significant errors, especially at higher latitudes where
    one degree of longitude covers much less distance than at the equator.

    Parameters
    ----------
    lats : array of shape (n,) — latitude values in decimal degrees
    lons : array of shape (n,) — longitude values in decimal degrees

    Returns
    -------
    dist_matrix : array of shape (n, n) — distances in km
    """
    # Convert degrees to radians (required for trigonometric functions)
    lat_rad = np.radians(lats)
    lon_rad = np.radians(lons)

    # Broadcast to create all pairwise combinations:
    # lat1[i, j] = latitude of point i, lat2[i, j] = latitude of point j
    lat1 = lat_rad[:, np.newaxis]  # shape (n, 1)
    lat2 = lat_rad[np.newaxis, :]  # shape (1, n)
    lon1 = lon_rad[:, np.newaxis]
    lon2 = lon_rad[np.newaxis, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Core Haversine formula:
    # 'a' is the square of half the chord length between the two points
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    # np.clip guards against floating-point rounding errors that could
    # push 'a' slightly outside [0, 1], which would make arcsin undefined
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    return EARTH_RADIUS_KM * c


# ── Load & filter data ────────────────────────────────────────────────────────

df = pd.read_csv(INPUT_CSV, dtype={"postal_code": str})

# Rows without coordinates cannot be placed on a map or in a distance
# matrix, so we drop them before filtering.
# Closed breweries are removed as well.
df = df.dropna(subset=["latitude", "longitude"]).copy()
df = df.loc[df.loc[:, "brewery_type"] != "closed"].copy()

# Build a boolean mask for the chosen region
mask = df.loc[:, FILTER_COLUMN] == FILTER_VALUE
if FILTER_COUNTRY is not None:
    # Combine with country filter using bitwise AND (&)
    # 'and' would not work element-wise on a pandas Series
    mask &= df.loc[:, "country"] == FILTER_COUNTRY

region_df = df.loc[mask].reset_index(drop=True)
n = len(region_df)

print(f"Breweries found in '{FILTER_VALUE}': {n}")

if n < 2:
    raise ValueError(
        f"Not enough breweries in '{FILTER_VALUE}' with known coordinates "
        f"(found {n}, need at least 2). Try a different region or filter."
    )

# ── Build distance matrix ─────────────────────────────────────────────────────

print("Computing pairwise Haversine distance matrix ...")

dist_matrix = haversine_distance_matrix(
    region_df.loc[:, "latitude"].values,
    region_df.loc[:, "longitude"].values,
)

print(f"Distance matrix shape: {dist_matrix.shape}")

# ── Solve TSP ─────────────────────────────────────────────────────────────────

if n <= EXACT_MAX_N:
    print(f"\nn={n} <= {EXACT_MAX_N} -> using exact solver (Dynamic Programming)")
    print("This guarantees the globally shortest route.")
    permutation, total_distance = solve_tsp_dynamic_programming(dist_matrix)
else:
    print(f"\nn={n} > {EXACT_MAX_N} -> using heuristic solver (Local Search / 2-opt)")
    print("Result is near-optimal but not guaranteed to be globally shortest.")
    permutation, total_distance = solve_tsp_local_search(dist_matrix)

print(f"\nTotal round-trip distance: {total_distance:.1f} km")

# ── Build route table ─────────────────────────────────────────────────────────

# 'permutation' is a list of row indices into region_df, in visit order.
# We reorder region_df accordingly and assign stop numbers 1, 2, 3, ...
route_df = region_df.loc[permutation].reset_index(drop=True)
route_df.loc[:, "stop_number"] = range(1, len(route_df) + 1)

# Calculate the distance from each stop to the next one.
# The last stop loops back to the first stop to close the round trip.
next_indices = permutation[1:] + [permutation[0]]
leg_distances = [dist_matrix[i, j] for i, j in zip(permutation, next_indices)]
route_df.loc[:, "distance_to_next_km"] = np.round(leg_distances, 1)

# Keep only the relevant output columns in a logical reading order
output_cols = [
    "stop_number",
    "name",
    "brewery_type",
    "city",
    "state_province",
    "country",
    "latitude",
    "longitude",
    "distance_to_next_km",
]
route_df = route_df.loc[:, output_cols]

print("\nOptimal route:")
print(route_df.to_string(index=False))

route_df.to_csv(OUTPUT_DIR / OUTPUT_ROUTE_CSV, index=False)
print(f"\n-> Route saved to {OUTPUT_DIR / OUTPUT_ROUTE_CSV}")

# ── Visualisation ─────────────────────────────────────────────────────────────

fig = go.Figure()

# Route line must be added first so it renders behind the markers
route_lats = route_df.loc[:, "latitude"].tolist() + [route_df.loc[0, "latitude"]]
route_lons = route_df.loc[:, "longitude"].tolist() + [route_df.loc[0, "longitude"]]

fig.add_trace(
    go.Scattermapbox(
        lon=route_lons,
        lat=route_lats,
        mode="lines",
        line=dict(width=2, color="#4c72b0"),
        name="Route",
        showlegend=False,
    )
)

# Brewery markers with hover info
fig.add_trace(
    go.Scattermapbox(
        lon=route_df.loc[:, "longitude"],
        lat=route_df.loc[:, "latitude"],
        mode="markers+text",
        text=route_df.loc[:, "stop_number"].astype(str),
        textposition="top center",
        marker=dict(size=8, color="#c44e52"),
        customdata=route_df.loc[
            :, ["city", "brewery_type", "distance_to_next_km"]
        ].values,
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "City: %{customdata[0]}<br>"
            "Type: %{customdata[1]}<br>"
            "Distance to next: %{customdata[2]} km"
            "<extra></extra>"
        ),
        hovertext=route_df.loc[:, "name"],
        name="Breweries",
    )
)

fig.update_layout(
    mapbox_style="carto-positron",
    mapbox=dict(
        center=dict(
            lat=route_df.loc[:, "latitude"].mean(),
            lon=route_df.loc[:, "longitude"].mean(),
        ),
        zoom=11,
    ),
    title=(
        f"Optimal Brewery Tour: {FILTER_VALUE} "
        f"({n} stops, {total_distance:.1f} km round trip)"
    ),
    margin=dict(l=0, r=0, t=40, b=0),
)

fig.write_html(OUTPUT_DIR / OUTPUT_MAP)
print(f"-> Interactive map saved to {OUTPUT_DIR / OUTPUT_MAP}")

print("\nDone.")
