import pandas as pd
import requests
from io import StringIO
import os
import re

# -----------------------------
# CONFIGURATION
# -----------------------------
# Primary CSV (original source)
csv_file_path = os.path.join("data", "NP_Node_Datasets.csv")

feature = "Node"  # Critical: Use "Node" for river nodes
start_time = "2023-01-01T00:00:00Z"
end_time = "2026-06-26T00:00:00Z"
output = "csv"

# River node fields (added river_name)
fields = "node_id,reach_id,river_name,time_str,wse,width,lat,lon,node_q,sword_version"

# Output directories
base_output_dir = "River_Output"
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
# INPUT SPECIFIC NODE IDS
# -----------------------------
print("="*60)
print("📝 ENTER NODE IDS TO DOWNLOAD")
print("="*60)
print("Options:")
print("  1. Enter specific node IDs (comma-separated)")
print("  2. Enter 'all' to download all nodes from CSV files")
print("  3. Enter 'list' to see available node IDs")
print("-"*60)

user_input = input("Enter node IDs (comma-separated), 'all', or 'list': ").strip()

# Load all available node IDs if needed
all_node_ids = set()

try:
    if os.path.exists(csv_file_path):
        node_df1 = pd.read_csv(csv_file_path)
        if 'node_id' in node_df1.columns:
            ids1 = node_df1['node_id'].dropna().astype(int).unique()
            all_node_ids.update(ids1)
            print(f"✅ Loaded {len(ids1)} node IDs from '{os.path.basename(csv_file_path)}' (column: 'node_id')")
        else:
            print(f"⚠️ Warning: Column 'min_node' not found in {csv_file_path}")
    else:
        print(f"⚠️ Warning: File not found: {csv_file_path}")
except Exception as e:
    print(f"❌ Error reading {csv_file_path}: {e}")

# Process user input
if user_input.lower() == 'all':
    # Use all available node IDs
    node_ids = sorted(list(all_node_ids))
    print(f"\n📊 Using all {len(node_ids)} unique node IDs from CSV files")
    
elif user_input.lower() == 'list':
    # Display available node IDs
    sorted_ids = sorted(list(all_node_ids))
    print(f"\n📋 Available Node IDs ({len(sorted_ids)} total):")
    print("-"*40)
    # Show first 50 IDs if there are many
    if len(sorted_ids) > 50:
        for i, node_id in enumerate(sorted_ids[:50], 1):
            print(f"  {i}. {node_id}")
        print(f"  ... and {len(sorted_ids) - 50} more")
    else:
        for i, node_id in enumerate(sorted_ids, 1):
            print(f"  {i}. {node_id}")
    print("-"*40)
    
    # Ask for input again
    user_input = input("\nEnter node IDs (comma-separated) or 'all': ").strip()
    if user_input.lower() == 'all':
        node_ids = sorted(list(all_node_ids))
    else:
        # Parse comma-separated IDs
        try:
            node_ids = [int(x.strip()) for x in user_input.split(',') if x.strip()]
        except ValueError:
            print("❌ Invalid input. Please enter comma-separated numbers.")
            exit(1)
            
else:
    # Parse comma-separated IDs
    try:
        node_ids = [int(x.strip()) for x in user_input.split(',') if x.strip()]
    except ValueError:
        print("❌ Invalid input. Please enter comma-separated numbers.")
        exit(1)

if not node_ids:
    print("❌ No valid node IDs provided. Exiting.")
    exit(1)

# Validate if the requested node IDs exist in the available data
available_ids = set(all_node_ids)
requested_ids = set(node_ids)
missing_ids = requested_ids - available_ids

if missing_ids:
    print(f"\n⚠️ Warning: The following node IDs were not found in the CSV files:")
    print(f"   {sorted(missing_ids)}")
    proceed = input("Do you want to continue anyway? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Exiting.")
        exit(1)

# Filter only available node IDs (with option to include missing ones)
if not missing_ids or proceed == 'y':
    node_ids_to_process = [nid for nid in node_ids if nid in available_ids or missing_ids]
else:
    node_ids_to_process = [nid for nid in node_ids if nid in available_ids]

print(f"\n📊 Processing {len(node_ids_to_process)} node IDs: {node_ids_to_process[:10]}{'...' if len(node_ids_to_process) > 10 else ''}")

# -----------------------------
# PROCESS EACH NODE
# -----------------------------
successful = 0
failed = 0

for idx, node_id in enumerate(node_ids_to_process, 1):
    print(f"\n{'='*60}")
    print(f"🔍 [{idx}/{len(node_ids_to_process)}] Fetching data for Node ID: {node_id}")
    print(f"{'='*60}")
    
    url = (
        f"https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries?"
        #f"collection_name=SWOT_L2_HR_RiverSP_2.0&"
        f"feature={feature}&"
        f"feature_id={node_id}&"
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
            failed += 1
            continue

        df = pd.read_csv(StringIO(data['results'][output]))

        if df.empty:
            print("  ⚠️ Empty dataset returned.")
            failed += 1
            continue

        # -----------------------------
        # CLEAN DATA
        # -----------------------------
        df['time_str'] = df['time_str'].replace('no_data', pd.NA)
        df['wse'] = pd.to_numeric(df['wse'], errors='coerce')
        df = df.dropna(subset=['time_str', 'wse'])

        if df.empty:
            print("  ⚠️ No valid WSE observations after cleaning.")
            failed += 1
            continue

        df['time_str'] = pd.to_datetime(df['time_str'])

        # Get river_name for filename (first non-null value)
        river_name = df['river_name'].dropna().iloc[0] if 'river_name' in df.columns and not df['river_name'].isna().all() else "Unknown"
        river_name_clean = clean_filename(river_name)

        # -----------------------------
        # SAVE CSV
        # -----------------------------
        output_file = os.path.join(
            csv_output_dir,
            f"node_{node_id}_{river_name_clean}.csv"
        )

        df = df.sort_values("time_str")
        df.to_csv(output_file, index=False)

        print(f"  ✅ Saved {len(df)} records")
        print(f"  📄 File: {output_file}")
        successful += 1

    except requests.exceptions.RequestException as e:
        print(f"  ❌ Request failed: {e}")
        failed += 1
    except Exception as e:
        print(f"  ❌ Error processing node {node_id}: {e}")
        failed += 1

# -----------------------------
# SUMMARY
# -----------------------------
print(f"\n{'='*60}")
print("🎉 PROCESSING COMPLETE")
print(f"{'='*60}")
print(f"✅ Successfully processed: {successful} nodes")
print(f"❌ Failed: {failed} nodes")
print(f"📊 Total attempted: {len(node_ids_to_process)} nodes")
print(f"\n📁 Output directory: {csv_output_dir}")