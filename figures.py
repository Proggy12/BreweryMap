from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("breweries_clean.csv", dtype={"postal_code": str})
save_path = Path("figs")
save_path.mkdir(exist_ok=True)

# Brewery type
type_counts = df.loc[:, "brewery_type"].value_counts()

fig, ax = plt.subplots(figsize=(8, 5))
type_counts.plot(kind="bar", ax=ax, color="#4c72b0")
ax.set_title("Distribution of Brewery Types")
ax.set_xlabel("Brewery Type")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(save_path / "plot_brewery_type.png", dpi=150)
plt.close()

# Top-10 countries
country_counts = df.loc[:, "country"].value_counts().head(10)

fig, ax = plt.subplots(figsize=(8, 5))
country_counts.plot(kind="bar", ax=ax, color="#55a868")
ax.set_title("Top 10 Countries by Number of Breweries")
ax.set_xlabel("Country")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(save_path / "plot_top10_countries.png", dpi=150)
plt.close()

# Top-10 states/provinces
state_counts = df.loc[:, "state_province"].value_counts().head(10)

fig, ax = plt.subplots(figsize=(8, 5))
state_counts.plot(kind="bar", ax=ax, color="#c44e52")
ax.set_title("Top 10 States/Provinces")
ax.set_xlabel("Region")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(save_path / "plot_top10_states.png", dpi=150)
plt.close()

# Brewery Type distribution only for USA
us_type_counts = df.loc[
    df.loc[:, "country"] == "United States", "brewery_type"
].value_counts()
fig, ax = plt.subplots(figsize=(8, 5))
us_type_counts.plot(kind="bar", ax=ax, color="#8172b2")
ax.set_title("Distribution of Brewery Types in the USA")
ax.set_xlabel("Brewery Type")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(save_path / "plot_brewery_type_usa.png", dpi=150)
plt.close()
