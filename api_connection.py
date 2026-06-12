import time
import requests
import pandas as pd

API_URL = "https://api.openbrewerydb.org/v1/breweries"
PER_PAGE = 200  # Maximum laut Doku
OUTPUT_CSV = "breweries.csv"
OUTPUT_JSON = "breweries.json"  # optional, kann auskommentiert werden


def fetch_all_breweries():
    """Holt alle Brauereien seitenweise von der API."""
    all_breweries = []
    page = 1

    while True:
        params = {"page": page, "per_page": PER_PAGE}
        response = requests.get(API_URL, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        if not data:  # leere Liste -> letzte Seite erreicht
            break

        all_breweries.extend(data)
        print(
            f"Seite {page}: {len(data)} Brauereien geladen "
            f"(gesamt: {len(all_breweries)})"
        )

        page += 1
        time.sleep(0.2)

    return all_breweries


def main():
    print("Starte Download der Brauerei-Daten ...")
    breweries = fetch_all_breweries()
    print(f"\nFertig! Insgesamt {len(breweries)} Brauereien geladen.")

    df = pd.DataFrame(breweries)

    # Sinnvolle Spaltenreihenfolge für eine übersichtliche Datei
    column_order = [
        "id",
        "name",
        "brewery_type",
        "address_1",
        "address_2",
        "address_3",
        "city",
        "state_province",
        "postal_code",
        "country",
        "latitude",
        "longitude",
        "phone",
        "website_url",
    ]
    # nur Spalten verwenden, die auch tatsächlich vorhanden sind
    column_order = [c for c in column_order if c in df.columns]
    df = df[column_order]

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"CSV gespeichert unter: {OUTPUT_CSV}")

    # Optional: zusätzlich als JSON speichern
    df.to_json(OUTPUT_JSON, orient="records", indent=2, force_ascii=False)
    print(f"JSON gespeichert unter: {OUTPUT_JSON}")


main()
