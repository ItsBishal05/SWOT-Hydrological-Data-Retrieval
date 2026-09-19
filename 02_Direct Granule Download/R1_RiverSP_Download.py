# SWOT_RiverReach_Downloader_Optimized.py
import earthaccess
import os
import getpass
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import traceback
import pandas as pd
import time
import logging
from contextlib import redirect_stdout, redirect_stderr
import io
import shutil
import glob
import re


def load_all_river_data(csv_file):
    """Load all river data from CSV and return a dictionary with reach_id as key"""
    try:
        # Read CSV with all columns as string to preserve formatting
        df = pd.read_csv(csv_file, dtype={'swot_orbit': str, 'reach_id': str})
        
        # Print column names for debugging
        print(f"📋 CSV Columns: {df.columns.tolist()}")
        
        # Check if required columns exist
        required_columns = ['reach_id', 'swot_orbit', 'min_river_name', 'min_x', 'min_y', 'min_node_id']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ Missing columns: {missing_columns}")
            print(f"   Available columns: {df.columns.tolist()}")
            return {}

        river_data_dict = {}

        for _, row in df.iterrows():
            # Get reach_id
            reach_id = str(row['reach_id']).strip()
            if not reach_id or reach_id == 'nan' or reach_id == 'NaN':
                continue
            
            # Get min_node_id
            try:
                min_node_id = int(row['min_node_id'])
            except (ValueError, TypeError):
                print(f"⚠️ Invalid min_node_id for reach {reach_id}, skipping")
                continue
            
            # Get river name (use min_river_name only)
            river_name = str(row['min_river_name']).strip()
            if not river_name or river_name == 'nan' or river_name == 'NaN' or river_name == 'NODATA':
                river_name = f"Reach_{reach_id}"
            
            # Clean river_name for folder creation
            clean_river_name = re.sub(r'[^\w\s-]', '', river_name)
            clean_river_name = re.sub(r'[-\s]+', '_', clean_river_name).strip('_')
            
            # Get coordinates (use min_x, min_y as point location)
            try:
                p_lon = float(row['min_x'])
                p_lat = float(row['min_y'])
                point = (p_lon, p_lat)
            except (ValueError, TypeError):
                print(f"⚠️ Invalid coordinates for reach {reach_id}, skipping")
                continue

            # Process swot_orbit (space-separated values)
            swot_orbit = str(row['swot_orbit']).strip()
            if not swot_orbit or swot_orbit == 'nan' or swot_orbit == 'NaN':
                print(f"⚠️ No orbit data for reach {reach_id}, skipping")
                continue
                
            if ' ' in swot_orbit:
                pass_list = [p.strip() for p in swot_orbit.split(' ') if p.strip()]
            else:
                pass_list = [swot_orbit]

            # Remove duplicates and ensure proper formatting
            pass_list = list(set(pass_list))
            
            # Format orbit numbers to match the expected pattern in granule names
            formatted_pass_list = []
            for orbit in pass_list:
                # Remove any non-digit characters and ensure proper formatting
                orbit_clean = re.sub(r'\D', '', orbit)
                if orbit_clean:
                    # Pad with leading zeros to make 3-digit orbit numbers
                    orbit_clean = orbit_clean.zfill(3)
                    formatted_pass_list.append(orbit_clean)

            # Store data
            river_data_dict[reach_id] = {
                'river_name': river_name,
                'clean_river_name': clean_river_name,
                'point': point,
                'pass_list': formatted_pass_list,
                'reach_id': reach_id,
                'min_node_id': min_node_id,
                'min_x': p_lon,
                'min_y': p_lat
            }

        print(f"✅ Successfully loaded {len(river_data_dict)} river reaches")
        return river_data_dict

    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        traceback.print_exc()
        return {}


def get_all_unique_orbits(river_data_dict):
    """Get all unique orbit numbers from all rivers"""
    unique_orbits = set()
    for reach_id, data in river_data_dict.items():
        unique_orbits.update(data['pass_list'])
    return sorted(list(unique_orbits))


def download_orbits_batch(unique_orbits, time_scale, central_download_dir,
                          product_variant="_Node_"):
    """
    Download files for given orbits using the exact Earthdata Search pattern.

    Parameters
    ----------
    unique_orbits : list[str]
        Zero-padded 3-digit orbit numbers (e.g. ["064", "217", "564"]).
    time_scale : str
        "YYYY-MM-DD to YYYY-MM-DD".
    central_download_dir : str
        Where to place downloaded files.
    product_variant : str
        "_Node_" (per-node, has node_id) or "_Reach_" (per-reach, has reach_id).
        Defaults to "_Node_" so the WSE extractor can find node_id.
    """
    try:
        start_date, end_date = time_scale.split(' to ')
        start_date = datetime.strptime(start_date.strip(), '%Y-%m-%d')
        end_date = datetime.strptime(end_date.strip(), '%Y-%m-%d')

        print(f"🔍 Searching for data for {len(unique_orbits)} unique orbits...")
        print(f"   Product variant: {product_variant.strip('_')}")
        print(f"   Orbit list: {', '.join(unique_orbits)}")

        all_results = []

        # 🇳🇵 Define bounding box for Nepal
        bbox_nepal = (79.72063, 25.02673, 89.09155, 32.6108)  # (lon_min, lat_min, lon_max, lat_max)

        # Pre-compile regex: matches "_{Variant}_{CYCLE}_{ORBIT}_" and captures ORBIT only.
        # Example: ..._Node_025_217_AS_...  ->  captures "217" (orbit), skips "025" (cycle).
        variant_clean = product_variant.strip('_')          # "Node" or "Reach"
        orbit_re = re.compile(rf'_{variant_clean}_\d{{3}}_(\d{{3}})_')

        for i, orbit_str in enumerate(unique_orbits, 1):
            print(f"   ➤ Searching orbit {orbit_str} ({i}/{len(unique_orbits)})...")

            search_params = {
                'short_name': 'SWOT_L2_HR_RiverSP_2.0',
                'temporal': (start_date, end_date),
                'bounding_box': bbox_nepal,
                'version': '2.0'
            }

            try:
                # Search for all granules in the bounding box (all orbits, all variants)
                results = earthaccess.search_data(**search_params)
                if not results:
                    print(f"     ⚠️ No granules found in Nepal region for orbit {orbit_str}")
                    continue

                print(f"     ✅ Found {len(results)} total granules in Nepal region")

                # Keep only granules whose ORBIT field (not cycle) equals orbit_str
                # AND whose product variant matches (Node/Reach).
                orbit_results = []
                for granule in results:
                    granule_name = granule['umm']['GranuleUR']
                    m = orbit_re.search(granule_name)
                    if m and m.group(1) == orbit_str:
                        orbit_results.append(granule)

                if orbit_results:
                    print(f"       ✅ Found {len(orbit_results)} {variant_clean} granules for orbit {orbit_str}")
                    all_results.extend(orbit_results)
                else:
                    print(f"       ⚠️ No matching {variant_clean} granules found for orbit {orbit_str}")

            except Exception as e:
                print(f"     ❌ Error searching orbit {orbit_str}: {e}")
                continue

            time.sleep(0.5)  # Be kind to CMR

        if not all_results:
            print(f"❌ No '{variant_clean}' data found for any of the specified orbits")
            return False, []

        # De-duplicate: the same granule may match multiple orbit searches if
        # orbit numbers repeat across cycles — keep one copy per GranuleUR.
        seen_urs = set()
        deduped = []
        for g in all_results:
            ur = g['umm']['GranuleUR']
            if ur not in seen_urs:
                seen_urs.add(ur)
                deduped.append(g)
        all_results = deduped

        total_files = len(all_results)
        print(f"\n✅ Found {total_files} unique '{variant_clean}' files across {len(unique_orbits)} orbits")
        print(f"📥 Downloading all files to central directory...")

        os.makedirs(central_download_dir, exist_ok=True)

        logging.getLogger("earthaccess").setLevel(logging.CRITICAL)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            files = earthaccess.download(all_results, central_download_dir)
            success_count = len(files) if files else 0

        logging.getLogger("earthaccess").setLevel(logging.INFO)
        print(f"✅ Downloaded {success_count}/{total_files} files")
        return success_count > 0, files if files else []

    except Exception as e:
        print(f"❌ Error during batch download: {e}")
        traceback.print_exc()
        return False, []


def organize_files_by_orbit(central_download_dir, output_base_dir):
    """
    Organize downloaded files into orbit-specific folders
    """
    print("\n📁 Organizing files by orbit ID...")
    
    # Create orbit_organized directory
    orbit_organized_dir = os.path.join(output_base_dir, "orbit_organized")
    os.makedirs(orbit_organized_dir, exist_ok=True)
    
    # Get all downloaded files
    downloaded_files = glob.glob(os.path.join(central_download_dir, "*.nc"))
    downloaded_files.extend(glob.glob(os.path.join(central_download_dir, "*.h5")))
    downloaded_files.extend(glob.glob(os.path.join(central_download_dir, "*.zip")))
    
    if not downloaded_files:
        print("❌ No downloaded files found to organize")
        return {}
    
    # Dictionary to track which files go to which orbit
    orbit_files = {}
    orbit_file_count = {}
    
    for file_path in downloaded_files:
        filename = os.path.basename(file_path)
        # Extract orbit number from filename (pattern: _Node_{CYCLE}_{ORBIT}_AS_)
        orbit_match = re.search(r'_Node_\d{3}_(\d{3})_', filename)
        if orbit_match:
            orbit = orbit_match.group(1)
            
            # Create orbit folder if it doesn't exist
            orbit_folder = os.path.join(orbit_organized_dir, f"orbit_{orbit}")
            os.makedirs(orbit_folder, exist_ok=True)
            
            # Create hard link or copy to orbit folder
            dest_path = os.path.join(orbit_folder, filename)
            if not os.path.exists(dest_path):
                try:
                    # Try hard link first (saves space)
                    os.link(file_path, dest_path)
                except (OSError, PermissionError):
                    # Fallback to copy
                    shutil.copy2(file_path, dest_path)
            
            # Track file
            if orbit not in orbit_files:
                orbit_files[orbit] = []
                orbit_file_count[orbit] = 0
            orbit_files[orbit].append(dest_path)
            orbit_file_count[orbit] += 1
    
    print(f"✅ Organized {len(downloaded_files)} files into {len(orbit_files)} orbit folders")
    
    # Print summary of orbit folders
    print("\n📊 Orbit folder summary:")
    for orbit in sorted(orbit_files.keys()):
        print(f"   Orbit {orbit}: {orbit_file_count[orbit]} files")
    
    return orbit_files


def create_orbit_manifest(orbit_files, output_base_dir):
    """Create a manifest file for orbit-organized files"""
    if not orbit_files:
        print("⚠️ No orbit files to create manifest")
        return
    
    manifest_records = []
    for orbit, file_paths in orbit_files.items():
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            manifest_records.append({
                'Orbit': orbit,
                'Filename': filename,
                'File_Path': file_path
            })
    
    manifest_df = pd.DataFrame(manifest_records)
    manifest_path = os.path.join(output_base_dir, "orbit_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)
    print(f"\n✅ Orbit manifest saved to: {manifest_path}")
    print(f"   Total records: {len(manifest_df)}")


def check_netrc_authentication():
    """Check if .netrc file authentication is successful"""
    netrc_path = os.path.join(os.path.expanduser("~"), ".netrc")
    try:
        if not os.path.exists(netrc_path):
            print("ℹ️ No .netrc file found, will try manual login")
            return False
        
        os.environ['NETRC'] = netrc_path
        results = earthaccess.search_data(short_name='SWOT_L2_HR_RiverSP_2.0', count=1)
        print("✅ .netrc authentication successful")
        return True
    except Exception as e:
        print(f"❌ .netrc authentication failed: {e}")
        return False


def manual_earthdata_login():
    """Fallback to manual authentication if .netrc fails"""
    print("\n🔐 .netrc authentication failed. Falling back to manual login.")
    username = input("Enter NASA Earthdata Username: ").strip()
    password = getpass.getpass("Enter NASA Earthdata Password: ").strip()
    if not username or not password:
        print("❌ Username and password cannot be empty")
        return False
    try:
        auth = earthaccess.login(strategy="interactive", persist=True)
        if auth:
            print("✅ Manual login successful")
            return True
        else:
            print("❌ Manual login failed - credentials rejected")
            return False
    except Exception as e:
        print(f"❌ Manual login error: {e}")
        return False


def create_netrc_file():
    """Helper function to create .netrc file"""
    print("\n📝 Let's create a .netrc file for you!")
    username = input("Enter your NASA Earthdata username (email): ").strip()
    password = getpass.getpass("Enter your NASA Earthdata password: ").strip()
    
    if not username or not password:
        print("❌ Username and password cannot be empty")
        return False
    
    home_dir = os.path.expanduser("~")
    netrc_path = os.path.join(home_dir, ".netrc")
    
    try:
        with open(netrc_path, 'w') as f:
            f.write(f"machine urs.earthdata.nasa.gov\n")
            f.write(f"login {username}\n")
            f.write(f"password {password}\n")
        
        print(f"✅ .netrc file created at {netrc_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to create .netrc file: {e}")
        return False


def main():
    """Main function: optimized download for all rivers"""
    print("🌍 NASA Earthdata SWOT RiverSP Downloader (Optimized)")
    print("=" * 70)
    print("📋 Strategy: Batch download by orbits + organize by orbit ID")
    print("=" * 70)

    # Check if .netrc exists
    home_dir = os.path.expanduser("~")
    netrc_path = os.path.join(home_dir, ".netrc")
    
    if not os.path.exists(netrc_path):
        print("⚠️ No .netrc file found.")
        create = input("Would you like to create one now? (y/n): ").strip().lower()
        if create == 'y':
            if not create_netrc_file():
                print("❌ Failed to create .netrc file.")
                return
        else:
            print("⚠️ Continuing without .netrc file. You'll need to log in manually.")

    # Authentication
    print("\n🔐 Attempting authentication...")
    
    if os.path.exists(netrc_path):
        try:
            os.environ['NETRC'] = netrc_path
            auth = earthaccess.login(strategy="netrc")
            if auth:
                print("✅ Authenticated using .netrc file")
            else:
                print("❌ .netrc authentication failed, trying manual login...")
                if not manual_earthdata_login():
                    print("❌ Authentication failed. Exiting.")
                    return
        except Exception as e:
            print(f"❌ .netrc authentication error: {e}")
            if not manual_earthdata_login():
                print("❌ Authentication failed. Exiting.")
                return
    else:
        print("ℹ️ No .netrc file found, proceeding with manual login...")
        if not manual_earthdata_login():
            print("❌ Authentication failed. Exiting.")
            return

    # CSV file path - UPDATE THIS TO YOUR ACTUAL PATH
    csv_file = "data/NP_Node_Datasets.csv"
    # Time range for SWOT RiverSP data
    time_scale = "2022-04-01 to 2026-08-10"

    # Directory paths
    output_base_dir = "swot_riversp_data"
    central_download_dir = os.path.join(output_base_dir, "central_downloads")

    # Check CSV exists
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        return

    # PHASE 1: Load all river data
    print("\n📊 Loading river data from CSV...")
    river_data_dict = load_all_river_data(csv_file)

    if not river_data_dict:
        print("❌ No valid river data found in CSV")
        return

    print(f"✅ Loaded data for {len(river_data_dict)} river reaches")
    
    # Print first few entries for verification
    print("\n📋 Sample of loaded data:")
    for i, (reach_id, data) in enumerate(list(river_data_dict.items())[:3]):
        print(f"   {i+1}. Reach {reach_id}: {data['river_name']} (Node: {data['min_node_id']}) - Orbits: {', '.join(data['pass_list'])}")

    # Get all unique orbits
    unique_orbits = get_all_unique_orbits(river_data_dict)
    print(f"\n📊 Found {len(unique_orbits)} unique orbit numbers: {', '.join(unique_orbits)}")

    # PHASE 2: Download all orbit files in batch to central directory
    print("\n" + "=" * 50)
    print("📥 PHASE 1: Downloading unique granules to central directory")
    print("=" * 50)
    
    success, downloaded_files = download_orbits_batch(
        unique_orbits, time_scale, central_download_dir
    )

    if not success:
        print("❌ Batch download failed. Exiting.")
        return

    # PHASE 3: Organize files by orbit ID
    print("\n" + "=" * 50)
    print("📁 PHASE 2: Organizing files by orbit ID")
    print("=" * 50)
    
    orbit_files = organize_files_by_orbit(central_download_dir, output_base_dir)
    
    # Create orbit manifest
    create_orbit_manifest(orbit_files, output_base_dir)

    # Final Summary
    print("\n" + "=" * 70)
    print(f"🎉 OPTIMIZED PROCESSING COMPLETE")
    print("=" * 70)
    print(f"📊 FINAL SUMMARY:")
    print(f"✅ Total river reaches processed: {len(river_data_dict)}")
    print(f"✅ Unique orbits downloaded: {len(unique_orbits)}")
    print(f"✅ Total granules downloaded: {len(downloaded_files)}")
    print(f"✅ Orbit folders created: {len(orbit_files)}")
    print(f"📁 Central downloads: {central_download_dir}")
    print(f"📁 Orbit-organized data: {os.path.join(output_base_dir, 'orbit_organized')}")
    print(f"📋 Manifest: {os.path.join(output_base_dir, 'orbit_manifest.csv')}")
    print("=" * 70)


if __name__ == "__main__":
    # Set up logging to suppress most messages
    logging.basicConfig(level=logging.CRITICAL)
    main()