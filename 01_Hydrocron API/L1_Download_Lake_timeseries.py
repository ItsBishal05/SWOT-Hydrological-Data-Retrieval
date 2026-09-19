import pandas as pd
import requests
from io import StringIO
import os
import re

# -----------------------------
# CONFIGURATION
# -----------------------------
csv_file_path = os.path.join("data", "NP_Lake_Datasets.csv")

feature = "PriorLake"
start_time = "2023-01-01T00:00:00Z"
end_time = "2026-02-01T00:00:00Z"
output = "csv"

fields = "lake_id,lake_name,time_str,wse,quality_f"

# Output directories
base_output_dir = "Lake_Output"
csv_output_dir = os.path.join(base_output_dir, "csv")

# Create folders if they do not exist
os.makedirs(csv_output_dir, exist_ok=True)

# -----------------------------
# HELPER FUNCTION
# -----------------------------
def clean_filename(text):
    """Make a string safe for Windows filenames"""
    text = str(text)
    text = text.strip().replace(" ", "_")
    return re.sub(r"[^\w\-]", "", text)

# -----------------------------
# LOAD LAKE IDs
# -----------------------------
try:
    lake_df = pd.read_csv(csv_file_path)

    if 'lake_id' not in lake_df.columns:
        raise ValueError("CSV must contain a column named 'lake_id'")

    lake_ids = lake_df['lake_id'].dropna().astype(int).unique()
    print(f"📂 Found {len(lake_ids)} unique lake IDs")

except Exception as e:
    print(f"❌ Failed to read lake IDs: {e}")
    exit(1)

# -----------------------------
# PROCESS EACH LAKE
# -----------------------------
for lake_id in lake_ids:

    print(f"\n{'='*60}")
    print(f"🔍 Fetching data for Lake ID: {lake_id}")
    print(f"{'='*60}")

    url = (
        f"https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries?"
        f"feature={feature}&"
        f"feature_id={lake_id}&"
        f"start_time={start_time}&"
        f"end_time={end_time}&"
        f"output={output}&"
        f"fields={fields}"
    )

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        if 'results' not in data or output not in data['results']:
            print("  ⚠️ No valid results in API response.")
            continue

        # Read the CSV data
        df = pd.read_csv(StringIO(data['results'][output]))

        if df.empty:
            print("  ⚠️ Empty dataset returned.")
            continue

        # -----------------------------
        # CLEAN AND FILTER DATA
        # -----------------------------
        # Clean time and WSE
        df['time_str'] = df['time_str'].replace('no_data', pd.NA)
        df['wse'] = pd.to_numeric(df['wse'], errors='coerce')
        df = df.dropna(subset=['time_str', 'wse'])

        if df.empty:
            print("  ⚠️ No valid WSE observations.")
            continue

        # Convert quality_f to numeric
        df['quality_f'] = pd.to_numeric(df['quality_f'], errors='coerce')
        
        # KEEP ONLY quality 0 (Good) and 1 (Suspect)
        # Remove all data with quality 2 (Bad) or any other values
        df = df[df['quality_f'].isin([0, 1])]
        
        if df.empty:
            print("  ⚠️ No data with quality 0 or 1. Skipping this lake.")
            continue

        # Convert time to datetime
        df['time_str'] = pd.to_datetime(df['time_str'])

        # Get lake name
        lake_name = df['lake_name'].dropna().iloc[0] if 'lake_name' in df.columns else "Unknown"
        lake_name_clean = clean_filename(lake_name)

        # -----------------------------
        # SAVE CSV
        # -----------------------------
        output_file = os.path.join(
            csv_output_dir,
            f"lake_{lake_id}_{lake_name_clean}.csv"
        )

        df = df.sort_values("time_str")
        df.to_csv(output_file, index=False)

        print(f"  ✅ Saved {len(df)} records (quality 0 and 1 only)")
        print(f"  📄 File: {output_file}")

    except requests.exceptions.RequestException as e:
        print(f"  ❌ Request failed: {e}")
    except Exception as e:
        print(f"  ❌ Error processing lake {lake_id}: {e}")

print("\n🎉 Finished processing all lakes.")