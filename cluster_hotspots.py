import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import DBSCAN
from pathlib import Path

INPUT_CSV = "breweries_clean.csv"
OUTPUT_CSV = "breweries_with_clusters.csv"
OUTPUT_MAP_HOTSPOTS = "map_brewery_hotspots.html"
OUTPUT_MAP_ALL = "map_breweries_all.html"
OUTPUT_SUMMARY = "hotspot_summary.csv"
save_path = Path("output")  # output directory for results
save_path.mkdir(exist_ok=True)  # ensure output directory exists

# DBSCAN parameters
EPS_KM = 25  # max distance between points in the same cluster
MIN_SAMPLES = 5  # minimum number of breweries to form a cluster
EARTH_RADIUS_KM = 6371.0

df = pd.read_csv(INPUT_CSV, dtype={"postal_code": str})
df_coords = df.dropna(subset=["latitude", "longitude"]).copy()

print(f"Breweries with coordinates: {len(df_coords)} of {len(df)}")

# Convert coordinates to radians (required by the Haversine metric)
coords_rad = np.radians(df_coords.loc[:, ["latitude", "longitude"]].values)

# Convert eps from kilometers to 'radian distance' (Haversine works on normalized distances on the unit sphere)
eps_rad = EPS_KM / EARTH_RADIUS_KM

# DBSCAN clustering with Haversine metric
db = DBSCAN(
    eps=eps_rad,
    min_samples=MIN_SAMPLES,
    metric="haversine",
    algorithm="ball_tree",
)

# Fit and predict cluster labels
df_coords.loc[:, "cluster"] = db.fit_predict(coords_rad)

# print(df_coords.loc[:, "cluster"].value_counts())

# How many clusters?
n_clusters = df_coords.loc[:, "cluster"].nunique() - (
    1 if -1 in df_coords.loc[:, "cluster"].values else 0
)
print(f"\nClusters found (hotspots): {n_clusters}")

# Amount of noise points (not in any cluster)
n_noise = df_coords.loc[:, "cluster"].eq(-1).sum()

print(
    f"Breweries outside of clusters (Noise): {n_noise} "
    f"({n_noise / len(df_coords) * 100:.1f}%)"
)

# Summarize clusters: size, center, top country/state/city
cluster_summary = []
for cluster_id, group in df_coords.groupby("cluster"):
    # Ignore noise points (cluster_id == -1) for the summary
    if cluster_id == -1:
        continue
    # Calculate summary statistics for the cluster and append to the list as a dictionary
    cluster_summary.append(
        {
            # unique cluster ID
            "cluster": cluster_id,
            # number of breweries in the cluster
            "n_breweries": len(group),
            # average latitude of the cluster
            "center_lat": group.loc[:, "latitude"].mean(),
            # average longitude of the cluster
            "center_lon": group.loc[:, "longitude"].mean(),
            # most common country in the cluster
            "top_country": group.loc[:, "country"].mode().iloc[0],
            # most common state/province in the cluster
            "top_state_province": group.loc[:, "state_province"].mode().iloc[0],
            # most common city in the cluster
            "top_city": group.loc[:, "city"].mode().iloc[0],
        }
    )

# Create a summary DataFrame and sort by number of breweries in the cluster
summary_df = pd.DataFrame(cluster_summary).sort_values("n_breweries", ascending=False)

# Print the top 15 hotspots
print("\nTop 15 Hotspots:")
print(summary_df.head(15).to_string(index=False))

# Save the summary DataFrame to a CSV file
summary_df.to_csv(save_path / OUTPUT_SUMMARY, index=False)
print(f"\n-> {OUTPUT_SUMMARY} saved")

# Save the original DataFrame with cluster labels to a new CSV file
df_coords.to_csv(save_path / OUTPUT_CSV, index=False)
print(f"-> {OUTPUT_CSV} saved")

# Visualization: clusters on a map (noise grey, clusters colored)
plot_df = df_coords.copy()
# Create a new column for cluster labels, where -1 (noise) is labeled as "Noise (no hotspot)" and others as "Cluster {cluster_id}"
plot_df.loc[:, "cluster_label"] = plot_df.loc[:, "cluster"].apply(
    lambda c: "Noise (no hotspot)" if c == -1 else f"Cluster {c}"
)

# Only show top 15 clusters, label the rest as "other cluster"
# get the cluster IDs of the top 15 clusters in a list
top_clusters = summary_df.loc[:, "cluster"].head(15).tolist()
# Change values in column "cluster_label" to "other cluster" if the cluster is not in the top 15 clusters and not noise
plot_df_top_clusters = plot_df.copy()
plot_df_top_clusters.loc[:, "cluster_label"] = plot_df_top_clusters.apply(
    lambda row: (
        row["cluster_label"]
        if row["cluster"] in top_clusters or row["cluster"] == -1
        else "other cluster"
    ),
    axis=1,
)

# Create a scatter geo plot with Plotly Express, coloring points by their cluster label
# Define a color map for the clusters, with "Noise (no hotspot)" as light grey and "other cluster" as light blue, while the top clusters will get default colors
color_map = {
    "Noise (no hotspot)": "lightgrey",
    "other cluster": "lightblue",
}
# Create scatter plot with only the top clusters colored
fig = px.scatter_geo(
    plot_df_top_clusters,
    lon="longitude",
    lat="latitude",
    color="cluster_label",
    color_discrete_map=color_map,
    projection="natural earth",
    opacity=0.6,
    hover_name="name",
    hover_data=["city", "state_province", "country"],
    title=f"Brewery-Hotspots (DBSCAN, eps={EPS_KM}km, " f"min_samples={MIN_SAMPLES})",
)

# Save the map as an HTML file
fig.write_html(save_path / OUTPUT_MAP_HOTSPOTS)
print(f"-> {OUTPUT_MAP_HOTSPOTS} saved")

# Create a scatter geo plot with all breweries plotted
fig = px.scatter_geo(
    plot_df.dropna(subset=["longitude", "latitude"]),
    lon="longitude",
    lat="latitude",
    color="cluster_label",
    color_discrete_map=color_map,
    projection="natural earth",
    opacity=0.5,
    hover_name="name",
    hover_data=["city", "state_province", "country"],
    title="All Breweries",
)

# Save the map as an HTML file
fig.write_html(save_path / OUTPUT_MAP_ALL)
print(f"-> {OUTPUT_MAP_ALL} saved")
