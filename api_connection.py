import time
import requests
import pandas as pd

API_URL = "https://api.openbrewerydb.org/v1/breweries"  # see https://www.openbrewerydb.org/documentation/
PER_PAGE = 200  # max. entries per page according to API documentation
OUTPUT_CSV = "breweries.csv"  # output CSV file
OUTPUT_JSON = "breweries.json"  # output JSON file


# Fetch all breweries from the API
def fetch_all_breweries():
    """Fetches all breweries page by page from the API."""
    all_breweries = []
    page = 1

    # The API returns data page by page. We iterate until an empty page is returned (i.e. no more data available).
    while True:
        params = {"page": page, "per_page": PER_PAGE}
        response = requests.get(API_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if not data:  # empty list -> last page reached
            break

        # Add all breweries from the current page to the overall list
        all_breweries.extend(data)
        print(
            f"Page {page}: {len(data)} breweries loaded "
            f"(total: {len(all_breweries)})"
        )
        page += 1
        time.sleep(0.2)  # short pause between requests

    return all_breweries


# Main function
def main():
    print("Starting brewery data download ...")
    breweries = fetch_all_breweries()
    print(f"\nDone! {len(breweries)} breweries loaded in total.")

    df = pd.DataFrame(breweries)

    # Sensible column order for a well-structured file
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

    # only use columns that are actually present
    column_order = [c for c in column_order if c in df.columns]
    df = df[column_order]

    # Output to CSV and JSON
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"CSV saved to: {OUTPUT_CSV}")

    df.to_json(OUTPUT_JSON, orient="records", indent=2, force_ascii=False)
    print(f"JSON saved to: {OUTPUT_JSON}")


# Run script
if __name__ == "__main__":
    main()
