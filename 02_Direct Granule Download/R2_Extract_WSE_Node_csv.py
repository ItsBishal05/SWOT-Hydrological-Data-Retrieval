import os
import pandas as pd
import geopandas as gpd
import re
import glob
from datetime import datetime

# ==========================
# CONFIGURATION
# ==========================

NODE_CSV_PATH = "data/NP_Node_Datasets.csv"
BASE_ORBIT_DIR = "swot_riversp_data/orbit_organized"
OUTPUT_BASE_DIR = "River_nodes_WSE_Data"

# Create required folders inside the specified output directory
os.makedirs(os.path.join(OUTPUT_BASE_DIR, "Data"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_BASE_DIR, "Plots"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_BASE_DIR, "Summary"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_BASE_DIR, "Trends"), exist_ok=True)

# ==========================
# HELPER FUNCTION: Extract time from filename
# ==========================
def extract_time_from_filename(filename):
    """Extract datetime from SWOT filename pattern: ..._YYYYMMDDTHHMMSS_..."""
    match = re.search(r'_(\d{8}T\d{6})_', filename)
    if match:
        dt_str = match.group(1)
        return datetime.strptime(dt_str, "%Y%m%dT%H%M%S")
    else:
        return None

# ==========================
# HELPER FUNCTION: Clean node ID
# ==========================
def clean_node_id(raw_node):
    """Convert raw min_node_id (possibly float like 123456.0) to clean string (e.g., '123456')"""
    try:
        if pd.isna(raw_node):
            return "NaN"
        return str(int(float(raw_node)))
    except (ValueError, TypeError, OverflowError):
        return str(raw_node).split('.')[0].strip()

# ==========================
# RESUME CONFIGURATION
# ==========================
LAST_PROCESSED_NODE_ID = "45242600080011"  # ⬅️ CHANGE THIS IF NEEDED

# ==========================
# MAIN PROCESSING
# ==========================
print("🚀 Starting WSE Time Series Extraction...")

# Load node inventory
try:
    node_df = pd.read_csv(NODE_CSV_PATH)
except Exception as e:
    print(f"❌ Error loading node CSV: {e}")
    exit(1)

# Required columns
required_cols = ['min_node_id', 'reach_id', 'min_river_name', 'swot_orbit']
if not all(col in node_df.columns for col in required_cols):
    print(f"❌ Missing required columns. Expected: {required_cols}")
    print(f"Available columns: {list(node_df.columns)}")
    exit(1)

total_nodes = len(node_df)
print(f"📊 Found {total_nodes} nodes to process.")

# --------------------------
# DETERMINE START INDEX FOR RESUME
# --------------------------
resume_index = -1
for i, row in node_df.iterrows():
    cleaned = clean_node_id(row['min_node_id'])
    if cleaned == LAST_PROCESSED_NODE_ID:
        resume_index = i

if resume_index >= 0:
    start_index = resume_index + 1
    print(f"⏭️  Resuming after node {LAST_PROCESSED_NODE_ID} (row index {resume_index})")
else:
    print(f"⚠️  Node {LAST_PROCESSED_NODE_ID} not found in inventory. Starting from beginning.")
    start_index = 0

# --------------------------
# PROCESS NODES
# --------------------------
for idx, row in node_df.iloc[start_index:].iterrows():
    # Clean min_node_id
    min_node_id = clean_node_id(row['min_node_id'])
    reach_id = str(row['reach_id'])
    river_name = str(row['min_river_name'])
    swot_orbits_str = str(row['swot_orbit'])

    # Parse orbits (e.g., "23 64" -> [23, 64])
    orbit_list = [o.strip() for o in swot_orbits_str.split() if o.strip()]
    try:
        orbit_list = [f"{int(o):03d}" for o in orbit_list]
    except ValueError:
        print(f"   ⚠️ Invalid orbit IDs in '{swot_orbits_str}' for node {min_node_id}")
        continue

    # Sanitize river name for filename
    safe_river_name = re.sub(r'[<>:"/\\|?*]', '_', river_name)
    output_filename = f"node={min_node_id}_reach={reach_id}_river={safe_river_name}.csv"
    output_path = os.path.join(OUTPUT_BASE_DIR, "Data", output_filename)

    # 🔁 Skip if already processed (extra safety)
    if os.path.exists(output_path):
        print(f"   ⏭️  Skipping (already exists): node {min_node_id}")
        continue

    print(f"\n🔍 Processing Node {min_node_id} | Reach {reach_id} | River '{river_name}'")
    print(f"   Orbits to scan: {orbit_list}")

    wse_data = []

    # Scan each orbit folder
    for orbit in orbit_list:
        orbit_dir = os.path.join(BASE_ORBIT_DIR, f"orbit_{orbit}")
        if not os.path.exists(orbit_dir):
            print(f"   ⚠️ Orbit folder not found: {orbit_dir}")
            continue

        # Find all .zip files
        zip_files = glob.glob(os.path.join(orbit_dir, "*.zip"))
        if not zip_files:
            print(f"   ⚠️ No .zip files in {orbit_dir}")
            continue

        print(f"   ➤ Scanning {len(zip_files)} ZIP files in orbit_{orbit}...")

        for zip_file in zip_files:
            try:
                # Read shapefile directly from ZIP
                zip_uri = f"zip://{zip_file}"
                gdf = gpd.read_file(zip_uri)

                # Normalize column names to lowercase
                gdf.columns = gdf.columns.str.lower()

                if 'node_id' not in gdf.columns:
                    print(f"      ❌ 'node_id' column not found in {os.path.basename(zip_file)}")
                    continue

                # Convert node_id to string for safe comparison
                gdf['node_id'] = gdf['node_id'].astype(str)

                # Filter for exact match
                node_rows = gdf[gdf['node_id'] == min_node_id]

                if len(node_rows) == 0:
                    continue  # No match

                # Extract time from filename
                file_time = extract_time_from_filename(os.path.basename(zip_file))
                if file_time is None:
                    print(f"      ⚠️ Could not parse time from {os.path.basename(zip_file)}")
                    continue

                # Extract valid WSE values
                for _, row_in_shp in node_rows.iterrows():
                    wse_val = row_in_shp.get('wse', None)
                    if wse_val is not None and pd.notna(wse_val):
                        try:
                            wse_data.append({
                                'time': file_time,
                                'wse': float(wse_val)
                            })
                        except (ValueError, TypeError):
                            continue  # Skip invalid WSE

            except Exception as e:
                print(f"      ❌ Error reading {os.path.basename(zip_file)}: {e}")
                continue

    # Save if data found
    if wse_data:
        df_out = pd.DataFrame(wse_data).sort_values('time').reset_index(drop=True)
        df_out.to_csv(output_path, index=False)
        print(f"   ✅ Saved {len(wse_data)} WSE points to: {output_path}")
    else:
        print(f"   ❌ No WSE data found for node {min_node_id}")

print("\n🎉 All remaining nodes processed!")