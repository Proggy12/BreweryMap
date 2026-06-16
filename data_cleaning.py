"""
Cleaning pipeline for the Open Brewery DB dataset (breweries.csv).

Steps
1.  Load raw data
2.  Fix Mojibake encoding errors in all text columns
3.  Normalize typographic apostrophes in all text columns
4.  Log & save all text-level changes to text_fixes_log.csv
5.  Inspect missing values and duplicates
6.  Drop irrelevant / mostly-empty columns
7.  Remove duplicate rows
8.  Save cleaned data to breweries_clean.csv
"""

import pandas as pd

INPUT_CSV = "breweries.csv"
OUTPUT_CSV = "breweries_clean.csv"
CHANGES_CSV = "text_fixes_log.csv"

# postal_code is kept as string to preserve leading zeros (e.g. "01234")
df = pd.read_csv(INPUT_CSV, dtype={"postal_code": str})

print(f"Loaded {len(df):,} rows x {len(df.columns)} columns from '{INPUT_CSV}'.")
print(f"Columns: {df.columns.tolist()}\n")


# Fix Mojibake and normalize apostrophes
def fix_mojibake(value: object) -> object:
    """
    Repair a single Mojibake-encoded string.

    Typical pattern: a UTF-8 string was decoded as Latin-1, so 'é' (U+00E9)
    became the two-character sequence 'Ã©'.  Re-encoding as Latin-1 and
    decoding as UTF-8 reverses this.

    Returns the original value unchanged if it is not a string or if the
    round-trip fails (e.g. the string already contains characters outside
    the Latin-1 range, meaning it was never mis-decoded).
    """
    if not isinstance(value, str):
        return value
    try:
        return value.encode("latin-1").decode("utf-8")
    except UnicodeEncodeError, UnicodeDecodeError:
        return value


def normalize_apostrophes(value: object) -> object:
    """
    Replace typographic apostrophes (', U+2019) with straight ones (', U+0027).
    """
    if not isinstance(value, str):
        return value
    return value.replace("\u2019", "'")


# Apply both fixes to every text (object-dtype) column and track changes
text_columns = df.select_dtypes(include="object").columns.tolist()

changes = []

for col in text_columns:
    original = df.loc[:, col].copy()

    fixed = original.apply(fix_mojibake)
    fixed = fixed.apply(normalize_apostrophes)

    # A cell is "changed" only when the new value differs AND it is not a
    # NaN-vs-NaN comparison (NaN != NaN evaluates to True in Python).
    changed_mask = (fixed != original) & ~(original.isna() & fixed.isna())

    for idx in df.index[changed_mask]:
        changes.append(
            {
                "row_index": idx,
                "id": df.loc[idx, "id"],
                "column": col,
                "before": original.loc[idx],
                "after": fixed.loc[idx],
            }
        )

    df.loc[:, col] = fixed


# Track text fixes in CHANGES_CSV
if changes:
    changes_df = pd.DataFrame(changes)
    changes_df.to_csv(CHANGES_CSV, index=False, encoding="utf-8")
    print(f"\n-> Change log saved as '{CHANGES_CSV}'")
else:
    print("-> No Mojibake or apostrophe issues found.")


# Drop irrelevant / mostly-empty columns
#   id          – internal API identifier, not necessary for analysis
#   address_2   – >99 % empty, not analytically useful
#   address_3   – >99 % empty, not analytically useful
#   phone       – not necessary for this analysis
#   website_url – not necessary for this analysis
COLS_TO_DROP = ["id", "address_2", "address_3", "phone", "website_url"]
df_clean = df.drop(columns=COLS_TO_DROP)


# Remove duplicate rows
# A row is considered a duplicate when name, address_1, city, and country
# all match.  We keep the first occurrence and discard the rest.
# Using a subset (rather than all columns) catches rows that differ only in
# sparse fields like phone or website_url.
# ---------------------------------------------------------------------------
DEDUP_SUBSET = ["name", "address_1", "city", "country"]

before = len(df_clean)
df_clean = df_clean.drop_duplicates(subset=DEDUP_SUBSET, keep="first")
after = len(df_clean)
print(f"Duplicate rows removed: {before - after:,}  ({before:,} -> {after:,} rows)\n")

remaining_dups = df_clean.duplicated(subset=DEDUP_SUBSET).sum()
print(f"Remaining duplicates after deduplication: {remaining_dups}\n")

# Save cleaned data
df_clean.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print(f"-> Cleaned data saved as '{OUTPUT_CSV}'  ({len(df_clean):,} rows)")
