# 🍺 BreweryMap

> A data science project exploring the Open Brewery DB — featuring geospatial hotspot detection, optimized brewery tour routing, and machine learning classification of brewery types.

---

## 📊 Project Overview

**Problem statement:**
The [Open Brewery DB](https://www.openbrewerydb.org/) contains over 11,000 breweries worldwide with location, type, and contact data. Raw as-is, the dataset has encoding issues, missing coordinates, and inconsistent text, making it unsuitable for direct analysis.

**Goal:**
Build a clean, reproducible data pipeline that transforms the raw brewery data into analysis-ready form, then use it to answer three concrete data science questions:
1. Where are the global brewery hotspots? (geospatial clustering)
2. What is the optimal route for a brewery tour in a given region? (TSP routing)
3. Can the brewery type be predicted from location features alone? (ML classification)

**Methods:**
- Data cleaning: Mojibake repair, apostrophe normalization, deduplication
- Geospatial clustering: DBSCAN with Haversine distance metric
- Route optimization: Travelling Salesman Problem (exact DP + 2-opt heuristic)
- Classification: Logistic Regression, Random Forest, XGBoost, LightGBM with RandomizedSearchCV

---

## 🔍 Research Questions & Results

### 1. Where are the global brewery hotspots?
Using DBSCAN clustering (eps=25 km, min_samples=5) on GPS coordinates with a Haversine distance metric, **222 hotspot clusters** were identified across 9,324 breweries with known coordinates (~80% of the dataset).

Top clusters by brewery count:

| Rank | Region | Breweries in Cluster |
|------|--------|----------------------|
| 1 | Munich / Bavaria, Germany | 929 |
| 2 | Brussels, Belgium | 524 |
| 3 | San Diego, CA, USA | 443 |
| 4 | Seattle, WA, USA | 363 |
| 5 | Baltimore, MD, USA | 320 |

→ See `cluster_output/map_brewery_hotspots.html` for the interactive world map.

---

### 2. What is the optimal brewery tour route?
The TSP solver computes the shortest round-trip route through all breweries in a chosen region. Two solvers are used depending on the number of stops:

- **≤ 12 stops**: exact Dynamic Programming (Held-Karp, guaranteed optimal)
- **> 12 stops**: Local Search heuristic (2-opt, near-optimal)

Distances are calculated as straight-line kilometres using the Haversine formula. The default region is **Berlin, GERMANY** — change `FILTER_VALUE`, `FILTER_COLUMN` and `FILTER_COUNTRY` in `brewery_tour.py` to plan a tour elsewhere.

→ See `tour_output/map_brewery_tour.html` for the interactive route map.

---

### 3. Can brewery type be predicted from location?
Four classifiers were compared using RandomizedSearchCV (20 iterations, 5-fold CV, scoring: Macro-F1):

| Model | Test Accuracy | Test Macro-F1 |
|-------|--------------|--------------|
| Random Forest | 0.51 | 0.29 |
| LightGBM | 0.48 | 0.27 |
| XGBoost | 0.62 | 0.27 |
| Logistic Regression | 0.28 | 0.19 |

**Key finding:** Location alone is a weak predictor of brewery type. Latitude and longitude account for ~76% of feature importance in tree-based models, but performance on minority classes (regional, contract, planning) remains poor. This suggests brewery type depends more on individual business decisions than on geography.

→ See `model_comparison_summary.png` and per-model confusion matrices in `classification_output/`.

---

## ⚙️ Setup

Clone the repository:
```bash
git clone https://github.com/Proggy12/BreweryMap.git
cd BreweryMap
```

Install [uv](https://docs.astral.sh/uv/) (if not already installed) and sync dependencies:
git clone https://github.com/Proggy12/BreweryMap.git
cd BreweryMap
```

Install [uv](https://docs.astral.sh/uv/) (if not already installed) and sync dependencies:
```bash
uv sync
```

---

## ▶️ Usage

Run the scripts in this order:

```bash
# 1. Fetch raw data from the Open Brewery DB API
python api_connection.py

# 2. Repair encoding issues and normalize apostrophes
python data_cleaning.py

# 3. EDA notebook (optional, for data exploration)
EDA.ipynb
figures.py

# 4. Geospatial hotspot clustering
python cluster_hotspots.py

# 5. Brewery tour route planning (edit FILTER_VALUE inside to change region)
python brewery_tour.py

# 6. Brewery type classification
python brewery_type_classification.oy
```

---

## 📦 Data Source

All brewery data is fetched from the [Open Brewery DB](https://www.openbrewerydb.org/) via its public REST API (`https://api.openbrewerydb.org/v1/breweries`). No API key required. The dataset is fetched page by page (200 entries per request) and saved locally as `breweries.csv` and `breweries.json`.

---

## 📝 Notes

- `postal_code` is always loaded as a string to preserve leading zeros (e.g. `"01234"`).
- ~20% of breweries have no GPS coordinates (`latitude`/`longitude` are NaN). These are excluded from all geospatial analyses but retained in the cleaned CSV.

---

## 📄 License

This project is for educational purposes. Brewery data is provided by [Open Brewery DB](https://www.openbrewerydb.org/) under the [MIT License](https://github.com/openbrewerydb/openbrewerydb/blob/master/LICENSE).
```

---

## ▶️ Usage

Run the scripts in this order:

```bash
# 1. Fetch raw data from the Open Brewery DB API
python api_connection.py

# 2. Repair encoding issues and normalize apostrophes
python data_cleaning.py

# 3. EDA notebook (optional, for data exploration)
EDA.ipynb
figures.py

# 4. Geospatial hotspot clustering
python cluster_hotspots.py

# 5. Brewery tour route planning (edit FILTER_VALUE inside to change region)
python brewery_tour.py

# 6. Brewery type classification
python brewery_type_classification.oy
```

---

## 📦 Data Source

All brewery data is fetched from the [Open Brewery DB](https://www.openbrewerydb.org/) via its public REST API (`https://api.openbrewerydb.org/v1/breweries`). No API key required. The dataset is fetched page by page (200 entries per request) and saved locally as `breweries.csv` and `breweries.json`.

---

## 📝 Notes

- `postal_code` is always loaded as a string to preserve leading zeros (e.g. `"01234"`).
- ~20% of breweries have no GPS coordinates (`latitude`/`longitude` are NaN). These are excluded from all geospatial analyses but retained in the cleaned CSV.

---

## 📄 License

This project is for educational purposes. Brewery data is provided by [Open Brewery DB](https://www.openbrewerydb.org/) under the [MIT License](https://github.com/openbrewerydb/openbrewerydb/blob/master/LICENSE).