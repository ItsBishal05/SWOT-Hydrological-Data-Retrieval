# SWOT_LakeSP_Downloader_Optimized.py
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
import shutil
import urllib3
from collections import defaultdict

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def setup_earthdata_auth():
    """Setup Earthdata authentication with SSL handling"""
    home_dir = os.path.expanduser("~")
    netrc_path = os.path.join(home_dir, ".netrc")
    
    if os.path.exists(netrc_path):
        print(f"✅ Found .netrc at {netrc_path}")
        try:
            os.environ['NETRC'] = netrc_path
            earthaccess.login(strategy="netrc")
            print("✅ Authentication successful with .netrc")
            return True
        except Exception as e:
            print(f"⚠️ .netrc authentication failed: {e}")
    
    # Manual login
    print("\n🔐 Please enter your NASA Earthdata credentials")
    username = input("Enter NASA Earthdata Username: ").strip()
    password = getpass.getpass("Enter NASA Earthdata Password: ").strip()
    
    if not username or not password:
        print("❌ Username and password cannot be empty")
        return False
    
    try:
        os.environ['EARTHDATA_USERNAME'] = username
        os.environ['EARTHDATA_PASSWORD'] = password
        earthaccess.login(username=username, password=password)
        print("✅ Manual login successful!")
        return True
    except Exception as e:
        print(f"❌ Manual login error: {e}")
        return False


def load_all_lake_data(csv_file):
    """Load all lake data from CSV and organize by unique pass numbers"""
    try:
        df = pd.read_csv(csv_file, dtype={'pass_full': str, 'pass_part': str})
        df['SN'] = pd.to_numeric(df['SN'], errors='coerce').fillna(0).astype(int)
        
        # Dictionary to store lake info organized by unique pass numbers
        pass_to_lakes = defaultdict(list)
        all_lake_info = {}
        
        for _, row in df.iterrows():
            sn = int(row['SN'])
            lake_name = row['lake_name']
            p_lon = float(row['p_lon'])
            p_lat = float(row['p_lat'])
            point = (p_lon, p_lat)
            
            # Process pass_full
            pass_full = str(row['pass_full']).strip()
            pass_list = []
            
            if ';' in pass_full:
                pass_list.extend([p.strip().zfill(3) for p in pass_full.split(';') if p.strip()])
            elif pass_full and pass_full != 'nan':
                pass_list.append(pass_full.zfill(3))
            
            # Process pass_part
            if 'pass_part' in row.index:
                pass_part = str(row['pass_part']).strip()
                if pass_part and pass_part != 'nan' and pass_part != '' and pass_part != 'None':
                    if ';' in pass_part:
                        pass_list.extend([p.strip().zfill(3) for p in pass_part.split(';') if p.strip()])
                    else:
                        pass_list.append(pass_part.zfill(3))
            
            # Remove duplicates and filter invalid
            pass_list = list(set([p for p in pass_list if p.isdigit() and len(p) == 3]))
            
            if not pass_list:
                print(f"⚠️ SN {sn} ({lake_name}): No valid pass numbers found")
                continue
            
            # Store lake info
            lake_info = {
                'sn': sn,
                'lake_name': lake_name,
                'point': point,
                'pass_list': pass_list
            }
            all_lake_info[sn] = lake_info
            
            # Map passes to lakes
            for pass_num in pass_list:
                pass_to_lakes[pass_num].append(sn)
        
        print(f"\n📊 Loaded {len(all_lake_info)} lakes with {len(pass_to_lakes)} unique pass numbers")
        return all_lake_info, pass_to_lakes
        
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        traceback.print_exc()
        return {}, {}


def search_and_download_unique_granules(pass_to_lakes, time_scale, output_base_dir):
    """Search and download unique granules once for all passes"""
    try:
        start_date, end_date = time_scale.split(' to ')
        start_date = datetime.strptime(start_date.strip(), '%Y-%m-%d')
        end_date = datetime.strptime(end_date.strip(), '%Y-%m-%d')
        
        # Dictionary to store downloaded granule paths by pass number
        pass_to_granules = defaultdict(list)
        
        print(f"\n{'='*60}")
        print(f"🔍 PHASE 1: Searching for unique granules for {len(pass_to_lakes)} unique passes")
        print(f"{'='*60}")
        
        # Create a central download directory
        central_download_dir = os.path.join(output_base_dir, "central_downloads")
        os.makedirs(central_download_dir, exist_ok=True)
        
        total_passes = len(pass_to_lakes)
        processed = 0
        
        for pass_num, lake_sns in pass_to_lakes.items():
            processed += 1
            print(f"\n📌 [{processed}/{total_passes}] Processing Pass {pass_num} (needed for {len(lake_sns)} lakes)")
            
            # Get one lake location for this pass (use the first lake in the list)
            sample_lake_sn = lake_sns[0]
            sample_lake_info = all_lake_info.get(sample_lake_sn)
            if not sample_lake_info:
                print(f"⚠️ No lake info found for SN {sample_lake_sn}")
                continue
            
            location = sample_lake_info['point']
            lake_name = sample_lake_info['lake_name']
            
            print(f"  Using location from {lake_name} (SN {sample_lake_sn}) for search")
            
            # Search for granules
            try:
                search_params = {
                    'short_name': 'SWOT_L2_HR_LakeSP_2.0',
                    'temporal': (start_date, end_date),
                    'point': location
                }
                
                print(f"  🔍 Searching for data...")
                results = earthaccess.search_data(**search_params)
                
                if not results:
                    print(f"  ❌ No data found for pass {pass_num}")
                    continue
                
                # Filter for Prior + specific pass
                filtered_results = []
                for r in results:
                    granule_ur = r['umm']['GranuleUR']
                    if "_Prior_" in granule_ur and f"_{pass_num}_AS_" in granule_ur:
                        filtered_results.append(r)
                
                if not filtered_results:
                    print(f"  ❌ No Prior granules found for pass {pass_num}")
                    continue
                
                print(f"  ✅ Found {len(filtered_results)} matching granules for pass {pass_num}")
                
                # Download granules if not already downloaded
                downloaded_paths = []
                
                for granule in filtered_results:
                    try:
                        url = granule.data_links()[0]
                        filename = url.split("/")[-1]
                        filepath = os.path.join(central_download_dir, filename)
                        
                        # Check if already downloaded
                        if os.path.exists(filepath):
                            print(f"  ℹ️ {filename} already exists, skipping download")
                            downloaded_paths.append(filepath)
                            continue
                        
                        # Download with fallback
                        print(f"  📥 Downloading: {filename}")
                        downloaded = download_granule_with_fallback(granule, central_download_dir)
                        
                        if downloaded:
                            downloaded_paths.append(downloaded)
                            print(f"  ✅ Downloaded: {os.path.basename(downloaded)}")
                        else:
                            print(f"  ❌ Failed to download: {filename}")
                            
                    except Exception as e:
                        print(f"  ❌ Error downloading granule: {e}")
                        continue
                
                if downloaded_paths:
                    pass_to_granules[pass_num] = downloaded_paths
                    
                # Delay to avoid overwhelming server
                time.sleep(2)
                
            except Exception as e:
                print(f"  ❌ Error processing pass {pass_num}: {e}")
                continue
        
        return pass_to_granules
        
    except Exception as e:
        print(f"❌ Error in search phase: {e}")
        traceback.print_exc()
        return {}


def download_granule_with_fallback(granule, output_dir, max_retries=3):
    """Download a single granule with fallback methods"""
    try:
        # Try earthaccess download first
        try:
            files = earthaccess.download([granule], output_dir)
            if files:
                return files[0]  # Return the downloaded file path
        except Exception as e:
            print(f"  ⚠️ earthaccess download failed: {e}")
        
        # Fallback to requests download
        print("  🔄 Using fallback download method...")
        username = os.environ.get('EARTHDATA_USERNAME')
        password = os.environ.get('EARTHDATA_PASSWORD')
        
        if not username or not password:
            # Try to read from .netrc
            netrc_path = os.path.join(os.path.expanduser("~"), ".netrc")
            if os.path.exists(netrc_path):
                try:
                    with open(netrc_path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            if 'login' in line:
                                username = line.split()[1]
                            elif 'password' in line:
                                password = line.split()[1]
                except:
                    pass
        
        if not username or not password:
            print("  ❌ No credentials found")
            return None
        
        # Download using requests
        session = requests.Session()
        retries = Retry(total=max_retries, backoff_factor=1, status_forcelist=[502, 503, 504, 401, 403])
        session.mount("https://", HTTPAdapter(max_retries=retries))
        session.verify = False  # Disable SSL verification
        
        url = granule.data_links()[0]
        filename = url.split("/")[-1]
        filepath = os.path.join(output_dir, filename)
        
        with session.get(url, stream=True, timeout=120, auth=(username, password)) as response:
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\r    Progress: {percent:.1f}%", end='')
        
        print()  # New line after progress
        return filepath if os.path.exists(filepath) else None
        
    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        return None


def distribute_files_to_lakes(all_lake_info, pass_to_granules, output_base_dir):
    """Distribute downloaded granules to each lake's folder using symlinks or copies"""
    print(f"\n{'='*60}")
    print(f"📦 PHASE 2: Distributing granules to lake folders")
    print(f"{'='*60}")
    
    # Create a manifest file to track which files go where
    manifest_records = []
    
    for sn, lake_info in all_lake_info.items():
        lake_name = lake_info['lake_name']
        pass_list = lake_info['pass_list']
        
        # Create lake-specific directory
        lake_dir = os.path.join(output_base_dir, lake_name)
        os.makedirs(lake_dir, exist_ok=True)
        
        print(f"\n📌 SN {sn}: {lake_name}")
        print(f"  Passes needed: {pass_list}")
        
        # Track how many files are linked
        linked_count = 0
        
        for pass_num in pass_list:
            if pass_num in pass_to_granules:
                granule_files = pass_to_granules[pass_num]
                print(f"  Pass {pass_num}: {len(granule_files)} granule(s)")
                
                for source_path in granule_files:
                    if os.path.exists(source_path):
                        filename = os.path.basename(source_path)
                        dest_path = os.path.join(lake_dir, filename)
                        
                        # Create a hard link (or copy if hard link not supported)
                        try:
                            # Try to create hard link (saves space)
                            if not os.path.exists(dest_path):
                                os.link(source_path, dest_path)
                                print(f"    ✅ Linked: {filename}")
                                linked_count += 1
                            else:
                                print(f"    ℹ️ {filename} already exists")
                        except (OSError, PermissionError):
                            # If hard link fails, create a symbolic link or copy
                            try:
                                # Try symbolic link first
                                if not os.path.exists(dest_path):
                                    os.symlink(source_path, dest_path)
                                    print(f"    ✅ Symlinked: {filename}")
                                    linked_count += 1
                            except (OSError, PermissionError):
                                # Fallback to copy
                                if not os.path.exists(dest_path):
                                    shutil.copy2(source_path, dest_path)
                                    print(f"    ✅ Copied: {filename}")
                                    linked_count += 1
                        
                        # Record in manifest
                        manifest_records.append({
                            'SN': sn,
                            'Lake': lake_name,
                            'Pass': pass_num,
                            'Filename': filename,
                            'Source': source_path,
                            'Destination': dest_path
                        })
            else:
                print(f"  ⚠️ Pass {pass_num}: No granules found")
        
        print(f"  📊 Total files available for this lake: {linked_count}")
    
    # Save manifest to CSV
    if manifest_records:
        manifest_df = pd.DataFrame(manifest_records)
        manifest_path = os.path.join(output_base_dir, "download_manifest.csv")
        manifest_df.to_csv(manifest_path, index=False)
        print(f"\n✅ Manifest saved to: {manifest_path}")
    
    return manifest_records


def main():
    """Main function: optimized download for SN 1 to 44"""
    print("🌍 NASA Earthdata SWOT LakeSP Prior Downloader (OPTIMIZED - Single Download)")
    print("=" * 60)
    print("⚠️ SSL verification is temporarily disabled for NASA servers")
    print("🚀 This version downloads each unique granule only once!")
    print("=" * 60)

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
    if not setup_earthdata_auth():
        print("❌ Authentication failed. Exiting.")
        return

    # CSV file path
    csv_file = "data/NP_Lake_Datasets.csv"
    time_scale = "2022-12-16 to 2026-08-10"

    # Check CSV exists
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        return

    output_base_dir = "swot_lakesp_data"
    
    # PHASE 1: Load all lake data
    print("\n📖 Loading lake data from CSV...")
    global all_lake_info  # Make it accessible in search function
    all_lake_info, pass_to_lakes = load_all_lake_data(csv_file)
    
    if not all_lake_info:
        print("❌ No lake data loaded. Exiting.")
        return
    
    # PHASE 2: Search and download unique granules
    pass_to_granules = search_and_download_unique_granules(
        pass_to_lakes, time_scale, output_base_dir
    )
    
    if not pass_to_granules:
        print("❌ No granules downloaded. Exiting.")
        return
    
    # PHASE 3: Distribute files to individual lake folders
    manifest = distribute_files_to_lakes(
        all_lake_info, pass_to_granules, output_base_dir
    )
    
    # Final Summary
    print("\n" + "=" * 60)
    print(f"📊 SUMMARY")
    print("=" * 60)
    print(f"✅ Total lakes processed: {len(all_lake_info)}")
    print(f"✅ Unique passes downloaded: {len(pass_to_granules)}")
    print(f"✅ Total granules downloaded: {sum(len(files) for files in pass_to_granules.values())}")
    print(f"✅ Manifest records created: {len(manifest)}")
    print(f"\n📁 Central downloads: {os.path.join(output_base_dir, 'central_downloads')}")
    print(f"📁 Individual lake folders: {output_base_dir}")
    print("=" * 60)


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


if __name__ == "__main__":
    main()