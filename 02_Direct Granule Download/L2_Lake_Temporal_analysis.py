import geopandas as gpd
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import zipfile
import numpy as np
from scipy.stats import linregress
import logging
from pathlib import Path
import math
import pymannkendall as mk

# Optional imports for combined trends
try:
    import pwlf
    PWLF_AVAILABLE = True
except ImportError:
    PWLF_AVAILABLE = False
    logging.warning("pwlf not installed. Piecewise trend skipped. Use: pip install pwlf")

try:
    from statsmodels.tsa.seasonal import STL
    STL_AVAILABLE = True
except ImportError:
    STL_AVAILABLE = False
    logging.warning("statsmodels not installed. STL trend skipped. Use: pip install statsmodels")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========= USER INPUT =========
input_directory = "swot_lakesp_data"
output_directory = "Lake_WSE_Data"
PRIMARY_ATTRIBUTE = "wse"           # water surface elevation
PRIMARY_LABEL = "Water Surface Elevation (m)"
PRIMARY_UNIT = "m"
FILTER_POSITIVE_ONLY = True
CASE_SENSITIVE_SEARCH = False
REMOVE_WSE_OUTLIERS = True
MIN_DATA_POINTS = 3                 # Minimum points required for analysis
OUTLIER_METHOD = "iqr"             # Options: "iqr", "zscore", "modified_zscore"
TREND_CONFIDENCE_LEVEL = 0.05       # P-value threshold for significance
MIN_FOR_STL = 12
MIN_FOR_PIECEWISE = 4
# ==============================


class LakeAnalyzer:
    def __init__(self, config):
        self.config = config
        self.setup_directories()
        self.summary_data = []
    
    def setup_directories(self):
        """Create output directories"""
        # Input directory
        self.input_dir = Path(self.config['input_directory'])
        
        # Output directory structure
        self.output_dir = Path(self.config['output_directory'])
        self.plots_dir = self.output_dir / "Plots"
        self.data_dir = self.output_dir / "Data" 
        self.trends_dir = self.output_dir / "Trends"
        self.summary_dir = self.output_dir / "Summary"
        
        # Plot subdirectories
        self.linear_dir = self.plots_dir / "Linear_Trend"
        self.combined_dir = self.plots_dir / "Combined_Trends"

        # Create all directories
        for dir_path in [self.input_dir, self.output_dir, self.plots_dir, self.data_dir, 
                        self.trends_dir, self.summary_dir, self.linear_dir, self.combined_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)
    
    def extract_datetime_from_filename(self, filename):
        """Enhanced datetime extraction"""
        datetime_patterns = [
            ("%Y%m%dT%H%M%S", 15),
            ("%Y%m%d", 8),
            ("%Y-%m-%d", 10),
            ("%Y_%m_%d", 10),
            ("%d%m%Y", 8)
        ]
        
        for part in filename.replace("-", "_").split("_"):
            for pattern, length in datetime_patterns:
                try:
                    if len(part) == length:
                        if "T" in pattern and "T" in part:
                            return datetime.strptime(part, pattern)
                        elif "T" not in pattern and part.isdigit():
                            return datetime.strptime(part, pattern)
                        elif "-" in pattern:
                            return datetime.strptime(part, pattern)
                except ValueError:
                    continue
        return None

    def find_lake_in_gdf(self, gdf, lake_name):
        """Find lake by name with case-insensitive matching"""
        name_columns = [col for col in gdf.columns if 'name' in col.lower()]
        if not name_columns:
            logger.warning(f"No name columns found in GeoDataFrame")
            return gpd.GeoDataFrame()
        
        name_col = name_columns[0]
        gdf_clean = gdf.dropna(subset=[name_col])
        if gdf_clean.empty:
            return gpd.GeoDataFrame()
        
        if self.config['case_sensitive']:
            mask = gdf_clean[name_col].str.strip() == lake_name
        else:
            mask = gdf_clean[name_col].str.strip().str.upper() == lake_name.upper()
        
        gdf_filtered = gdf_clean[mask]
        if not gdf_filtered.empty:
            return gdf_filtered
        
        try:
            if self.config['case_sensitive']:
                mask = gdf_clean[name_col].str.contains(lake_name, na=False, regex=False)
            else:
                mask = gdf_clean[name_col].str.contains(lake_name, case=False, na=False, regex=False)
            return gdf_clean[mask]
        except Exception as e:
            logger.warning(f"Error in partial matching: {e}")
            return gpd.GeoDataFrame()

    def flag_outliers(self, df, colname, method="iqr"):
        """Multiple outlier detection methods"""
        if method == "iqr":
            Q1 = df[colname].quantile(0.25)
            Q3 = df[colname].quantile(0.75)
            IQR = Q3 - Q1
            lower_near = Q1 - 1.5 * IQR
            upper_near = Q3 + 1.5 * IQR
            lower_far = Q1 - 3 * IQR
            upper_far = Q3 + 3 * IQR
            
            near_outliers = ((df[colname] < lower_near) & (df[colname] >= lower_far)) | \
                            ((df[colname] > upper_near) & (df[colname] <= upper_far))
            far_outliers = (df[colname] < lower_far) | (df[colname] > upper_far)
            
            df["is_near_outlier_wse"] = near_outliers
            df["is_far_outlier_wse"] = far_outliers
            return far_outliers
        
        elif method == "zscore":
            z_scores = np.abs((df[colname] - df[colname].mean()) / df[colname].std())
            return z_scores > 3
        
        elif method == "modified_zscore":
            median = df[colname].median()
            mad = np.median(np.abs(df[colname] - median))
            modified_z_scores = 0.6745 * (df[colname] - median) / mad
            return np.abs(modified_z_scores) > 3.5
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

    def compute_trend(self, x, y):
        """Enhanced trend analysis with OLS"""
        if len(x) < 2:
            return None
        try:
            x_num = np.array([d.toordinal() for d in x])
            result = linregress(x_num, y)
            return result
        except Exception as e:
            logger.error(f"Error in trend computation: {e}")
            return None

    def compute_mann_kendall(self, y):
        """Compute Mann-Kendall test and Sen's slope"""
        if len(y) < 3:
            return None
        try:
            result = mk.original_test(y)
            return result
        except Exception as e:
            logger.error(f"Error in Mann-Kendall computation: {e}")
            return None

    def create_linear_plot(self, original_df, df, lake_name, trend_result, mk_result):
        """Create the original-style linear plot with stats box and legend"""
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        non_outliers = df[~df["is_near_outlier_wse"]]
        near_outliers = df[df["is_near_outlier_wse"]]
        
        # Combine for y-limit calculation
        plotted_df = pd.concat([non_outliers, near_outliers])
        
        # Main data plot (non-outliers)
        ax1.plot(non_outliers["datetime"], non_outliers[self.config['primary_attribute']], 
                 marker="o", color="blue", linewidth=2, markersize=6,
                 label=f"{self.config['primary_label']}", zorder=5)
        
        # Plot near outliers as red X
        if not near_outliers.empty:
            ax1.scatter(near_outliers["datetime"], near_outliers[self.config['primary_attribute']],
                        color="red", marker="x", s=100, linewidth=3,
                        label=f"{self.config['primary_attribute'].upper()} Outliers", zorder=6)
        
        # Connect near outliers in sequence
        if not near_outliers.empty:
            df_sorted = df.sort_values("datetime")
            is_non_outlier = ~df_sorted["is_near_outlier_wse"]
            non_outlier_indices = np.where(is_non_outlier)[0]
            
            if len(non_outlier_indices) >= 2:
                for i in range(len(non_outlier_indices) - 1):
                    start_idx = non_outlier_indices[i]
                    end_idx = non_outlier_indices[i + 1]
                    segment = df_sorted.iloc[start_idx:end_idx + 1]
                    outlier_segment = segment[segment["is_near_outlier_wse"]]
                    
                    if not outlier_segment.empty:
                        first_non_outlier = segment.iloc[0]
                        first_outlier = outlier_segment.iloc[0]
                        ax1.plot([first_non_outlier["datetime"], first_outlier["datetime"]], 
                                 [first_non_outlier[self.config['primary_attribute']], 
                                  first_outlier[self.config['primary_attribute']]], 
                                 color="red", linestyle="--", linewidth=1, zorder=4)
                        
                        for j in range(len(outlier_segment) - 1):
                            curr = outlier_segment.iloc[j]
                            next_ = outlier_segment.iloc[j + 1]
                            ax1.plot([curr["datetime"], next_["datetime"]], 
                                     [curr[self.config['primary_attribute']], 
                                      next_[self.config['primary_attribute']]], 
                                     color="red", linestyle="--", linewidth=1, zorder=4)
                        
                        last_outlier = outlier_segment.iloc[-1]
                        second_non_outlier = segment.iloc[-1]
                        ax1.plot([last_outlier["datetime"], second_non_outlier["datetime"]], 
                                 [last_outlier[self.config['primary_attribute']], 
                                  second_non_outlier[self.config['primary_attribute']]], 
                                 color="red", linestyle="--", linewidth=1, zorder=4)
        
        # OLS Trend line
        if trend_result and len(non_outliers) > 1:
            x_vals = np.array([d.toordinal() for d in non_outliers["datetime"]])
            y_fit = trend_result.intercept + trend_result.slope * x_vals
            ax1.plot(non_outliers["datetime"], y_fit, 
                     color="navy", linestyle="--", linewidth=2,
                     label="OLS Trend", zorder=3)
        
        # Sen's Slope line (if available)
        if mk_result and len(non_outliers) > 1:
            # Sen's slope line centered on median
            x_vals = np.array([d.toordinal() for d in non_outliers["datetime"]])
            x_centered = x_vals - np.median(x_vals)
            sen_intercept = np.median(non_outliers[self.config['primary_attribute']].values)
            y_sen = sen_intercept + mk_result.slope * x_centered
            ax1.plot(non_outliers["datetime"], y_sen, 
                     color="red", linestyle=":", linewidth=2,
                     label="Sen's Slope", zorder=3)
        
        # Formatting
        ax1.set_xlabel("Date", fontsize=12, fontweight='bold')
        ax1.set_ylabel(self.config['primary_label'], color="blue", fontsize=12, fontweight='bold')
        ax1.tick_params(axis='x', labelsize=10, rotation=45)
        ax1.tick_params(axis='y', labelsize=10, labelcolor="blue")
        
        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_fontweight('bold')
        
        # === Ensure lowest point is visible ===
        y_min = plotted_df[self.config['primary_attribute']].min()
        y_max = plotted_df[self.config['primary_attribute']].max()

        bottom_buffer = 0.10
        top_buffer = 0.03

        y_min_pad = y_min - bottom_buffer
        y_max_pad = y_max + top_buffer

        ax1.set_ylim(y_min_pad, y_max_pad)

        # Dynamic y-ticks
        y_range = y_max - y_min
        base_interval = y_range / 4
        adjusted_interval = math.floor(base_interval * 2) / 2
        major_interval = math.floor(adjusted_interval) + 0.5 if adjusted_interval % 1 != 0 else adjusted_interval
        major_interval = max(major_interval, 0.5)

        y_min_round = math.floor(y_min_pad / major_interval) * major_interval
        y_max_round = math.ceil(y_max_pad / major_interval) * major_interval
        major_ticks = np.arange(y_min_round, y_max_round + major_interval, major_interval)
        if len(major_ticks) > 6:
            major_ticks = np.linspace(y_min_round, y_max_round, 6, endpoint=True)
        ax1.set_yticks(major_ticks)
        ax1.grid(True, which='major', alpha=0.7, linewidth=1.2)

        minor_interval = major_interval / 5
        minor_ticks = np.arange(y_min_round, y_max_round + minor_interval, minor_interval)
        ax1.set_yticks(minor_ticks, minor=True)
        ax1.grid(True, which='minor', alpha=0.4, linewidth=0.8, linestyle='--')
        
        # Statistics box
        stats_text = self.create_stats_text(original_df, non_outliers, trend_result, mk_result)
        ax1.text(0.02, 0.02, stats_text, transform=ax1.transAxes,
                 verticalalignment='bottom', horizontalalignment='left',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                 fontsize=10)
        
        # Legend
        ax1.legend(loc="lower right", fontsize=11)
        
        plt.title(f"Temporal Analysis: {lake_name}", fontsize=16, fontweight="bold", pad=20)
        plt.tight_layout()
        
        return fig

    def create_combined_plot(self, df, lake_name):
        """Create combined trend plot with all models overlaid"""
        fig, ax = plt.subplots(figsize=(12, 7))
        non_outliers = df[~df["is_near_outlier_wse"]].sort_values("datetime")
        near_outliers = df[df["is_near_outlier_wse"]]
        
        # Combine for y-limit calculation
        plotted_df = pd.concat([non_outliers, near_outliers])
        
        if len(non_outliers) < 2:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha='center', va='center', fontsize=14)
            ax.set_title(f"Combined Trends: {lake_name}")
            plt.tight_layout()
            return fig

        # Plot data
        ax.plot(non_outliers["datetime"], non_outliers[self.config['primary_attribute']], 
                marker="o", color="blue", linewidth=2, markersize=5, label="WSE", zorder=5)
        if not near_outliers.empty:
            ax.scatter(near_outliers["datetime"], near_outliers[self.config['primary_attribute']],
                       color="red", marker="x", s=60, linewidth=2, label="Outliers", zorder=6)
        
        # Get all trends
        trend_results = self.get_trend_predictions(non_outliers)
        
        # Plot all trends
        for res in trend_results.values():
            ax.plot(non_outliers["datetime"], res['y_pred'],
                    color=res['color'], linestyle=res['style'], linewidth=2,
                    label=res['label'], zorder=3)
        
        # Formatting
        ax.set_title(f"Combined Trends: {lake_name}", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Date", fontsize=12, fontweight='bold')
        ax.set_ylabel(self.config['primary_label'], color="blue", fontsize=12, fontweight='bold')
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        ax.tick_params(axis='x', labelsize=10, rotation=45)
        ax.tick_params(axis='y', labelsize=10, labelcolor="blue")
        ax.grid(True, alpha=0.6)
        ax.legend(loc='upper left', fontsize=9, ncol=2)

        # === Ensure lowest point is visible ===
        y_min = plotted_df[self.config['primary_attribute']].min()
        y_max = plotted_df[self.config['primary_attribute']].max()

        bottom_buffer = 0.10
        top_buffer = 0.03

        y_min_pad = y_min - bottom_buffer
        y_max_pad = y_max + top_buffer

        ax.set_ylim(y_min_pad, y_max_pad)

        # Dynamic y-ticks
        y_range = y_max - y_min
        base_interval = y_range / 4
        adjusted_interval = math.floor(base_interval * 2) / 2
        major_interval = math.floor(adjusted_interval) + 0.5 if adjusted_interval % 1 != 0 else adjusted_interval
        major_interval = max(major_interval, 0.5)

        y_min_round = math.floor(y_min_pad / major_interval) * major_interval
        y_max_round = math.ceil(y_max_pad / major_interval) * major_interval
        major_ticks = np.arange(y_min_round, y_max_round + major_interval, major_interval)
        if len(major_ticks) > 6:
            major_ticks = np.linspace(y_min_round, y_max_round, 6, endpoint=True)
        ax.set_yticks(major_ticks)
        ax.grid(True, which='major', alpha=0.7, linewidth=1.2)

        minor_interval = major_interval / 5
        minor_ticks = np.arange(y_min_round, y_max_round + minor_interval, minor_interval)
        ax.set_yticks(minor_ticks, minor=True)
        ax.grid(True, which='minor', alpha=0.4, linewidth=0.8, linestyle='--')

        plt.tight_layout()
        return fig

    def get_trend_predictions(self, non_outliers):
        """Compute predictions and R² for all trend types (for combined plot)"""
        results = {}
        x_days = np.array([d.toordinal() for d in non_outliers["datetime"]])
        y_vals = non_outliers[self.config['primary_attribute']].values

        # 1. Linear (OLS)
        linear_trend = self.compute_trend(non_outliers["datetime"], y_vals)
        if linear_trend:
            y_pred = linear_trend.intercept + linear_trend.slope * x_days
            r2 = linear_trend.rvalue ** 2
            results['Linear'] = {
                'y_pred': y_pred,
                'r2': r2,
                'label': f"OLS (R²={r2:.3f})",
                'color': 'navy',
                'style': '--'
            }

        # 2. Sen's Slope (if Mann-Kendall available)
        mk_result = self.compute_mann_kendall(y_vals)
        if mk_result:
            x_centered = x_days - np.median(x_days)
            sen_intercept = np.median(y_vals)
            y_pred = sen_intercept + mk_result.slope * x_centered
            r2 = np.corrcoef(y_vals, y_pred)[0,1]**2
            results['Sen\'s Slope'] = {
                'y_pred': y_pred,
                'r2': r2,
                'label': f"Sen\'s (R²={r2:.3f})",
                'color': 'red',
                'style': ':'
            }

        # 3. Polynomial (Quadratic)
        if len(non_outliers) >= 3:
            coeffs = np.polyfit(x_days, y_vals, deg=2)
            poly = np.poly1d(coeffs)
            y_pred = poly(x_days)
            r2 = np.corrcoef(y_vals, y_pred)[0,1]**2
            results['Polynomial'] = {
                'y_pred': y_pred,
                'r2': r2,
                'label': f"Quadratic (R²={r2:.3f})",
                'color': 'green',
                'style': '-.'
            }

        # 4. Moving Average
        window = min(3, len(non_outliers) // 2 or 1)
        ma = non_outliers[self.config['primary_attribute']].rolling(window=window, center=True).mean()
        if not ma.isna().all():
            residuals = (y_vals - ma) ** 2
            r2 = 1 - np.sum(residuals) / np.sum((y_vals - y_vals.mean()) ** 2)
            results['Moving Average'] = {
                'y_pred': ma.values,
                'r2': r2,
                'label': f"MA (w={window})",
                'color': 'purple',
                'style': '-'
            }

        # 5. Piecewise
        if PWLF_AVAILABLE and len(non_outliers) >= MIN_FOR_PIECEWISE:
            try:
                my_pwlf = pwlf.PiecewiseLinFit(x_days, y_vals)
                breaks = my_pwlf.fit(2)
                y_pred = my_pwlf.predict(x_days)
                r2 = np.corrcoef(y_vals, y_pred)[0,1]**2
                results['Piecewise'] = {
                    'y_pred': y_pred,
                    'r2': r2,
                    'label': f"Piecewise (R²={r2:.3f})",
                    'color': 'orange',
                    'style': ':'
                }
            except Exception as e:
                logger.debug(f"Piecewise fit failed: {e}")

        # 6. STL Trend
        if STL_AVAILABLE and len(non_outliers) >= MIN_FOR_STL:
            try:
                df_temp = non_outliers.set_index("datetime")[self.config['primary_attribute']].sort_index()
                df_monthly = df_temp.resample('ME').mean().dropna()
                if len(df_monthly) >= MIN_FOR_STL:
                    stl = STL(df_monthly, seasonal=13, period=12).fit()
                    stl_daily = np.interp(
                        np.searchsorted(stl.trend.index.astype(int), x_days),
                        np.arange(len(stl.trend)), 
                        stl.trend.values
                    )
                    r2 = np.corrcoef(y_vals, stl_daily)[0,1]**2
                    results['STL'] = {
                        'y_pred': stl_daily,
                        'r2': r2,
                        'label': f"STL (R²={r2:.3f})",
                        'color': 'brown',
                        'style': '-'
                    }
            except Exception as e:
                logger.debug(f"STL failed: {e}")

        return results

    def create_stats_text(self, df, non_outliers, trend_result, mk_result):
        """Create comprehensive statistics text with OLS and Mann-Kendall results"""
        near_count = len(df[df['is_near_outlier_wse']]) if 'is_near_outlier_wse' in df else 0
        far_count = len(df[df['is_far_outlier_wse']]) if 'is_far_outlier_wse' in df else 0
        
        stats = [
            f"Total points: {len(df)}",
            f"Valid points: {len(non_outliers)}",
            f"Near Outliers: {near_count}",
            f"Far Outliers: {far_count}",
            f"Date range: {df['datetime'].min().strftime('%Y-%m-%d')} to {df['datetime'].max().strftime('%Y-%m-%d')}",
            f"WSE range: {non_outliers[self.config['primary_attribute']].min():.2f} – {non_outliers[self.config['primary_attribute']].max():.2f} {self.config['primary_unit']}"
        ]
        
        # OLS results
        if trend_result and len(non_outliers) > 1:
            annual_change = trend_result.slope * 365
            significance = "Significant" if trend_result.pvalue < self.config['confidence_level'] else "Not Significant"
            stats.extend([
                "",
                "OLS Regression:",
                f"  Trend: {annual_change:.3f} {self.config['primary_unit']}/yr",
                f"  R² = {trend_result.rvalue**2:.3f}",
                f"  p-value = {trend_result.pvalue:.4f} ({significance})"
            ])
        
        # Mann-Kendall results
        if mk_result:
            mk_trend = mk_result.trend
            mk_p = mk_result.p
            mk_significant = "Significant" if mk_p < self.config['confidence_level'] else "Not Significant"
            # Convert Sen's slope to annual change
            if len(non_outliers) > 1:
                span_days = (non_outliers['datetime'].max() - non_outliers['datetime'].min()).days
                if span_days > 0:
                    sen_annual = mk_result.slope * (len(non_outliers) / span_days) * 365
                else:
                    sen_annual = np.nan
            else:
                sen_annual = np.nan
            
            stats.extend([
                "",
                "Mann-Kendall Test:",
                f"  Trend: {mk_trend}",
                f"  Sen's Slope: {sen_annual:.3f} {self.config['primary_unit']}/yr" if not np.isnan(sen_annual) else "  Sen's Slope: N/A",
                f"  p-value = {mk_p:.4f} ({mk_significant})",
                f"  Tau = {mk_result.Tau:.3f}"
            ])
        
        return "\n".join(stats)

    def process_lake_folder(self, lake_folder_path):
        """Process each lake folder"""
        lake_name = lake_folder_path.name
        records = []
        
        logger.info(f"Processing lake: {lake_name}")
        
        for file_path in lake_folder_path.iterdir():
            if file_path.suffix.lower() == '.shp':
                shp_paths = [str(file_path)]
            elif file_path.suffix.lower() == '.zip':
                shp_paths = self.extract_shp_from_zip(file_path)
            else:
                continue
            
            for shp_path in shp_paths:
                try:
                    gdf = gpd.read_file(shp_path)
                    gdf_filtered = self.find_lake_in_gdf(gdf, lake_name)
                    
                    if gdf_filtered.empty or self.config['primary_attribute'] not in gdf_filtered.columns:
                        continue
                    
                    dt = self.extract_datetime_from_filename(file_path.name)
                    if dt is None:
                        continue
                    
                    row = gdf_filtered.iloc[0]
                    value = row[self.config['primary_attribute']]
                    
                    if pd.isna(value) or (self.config['filter_positive'] and value <= 0):
                        continue
                        
                    records.append([dt, value])
                    
                except Exception as e:
                    logger.error(f"Error processing {shp_path}: {e}")
        
        if len(records) < self.config['min_data_points']:
            logger.warning(f"Insufficient data for {lake_name} ({len(records)} points)")
            self.summary_data.append({
                'Lake': lake_name, 'Status': 'Insufficient Data', 'Total_Points': len(records)
            })
            return
        
        # Process data
        df = self.process_lake_data(records, lake_name)
        if df is not None:
            self.save_results(df, lake_name)

    def extract_shp_from_zip(self, zip_path):
        """Extract shapefile paths from zip"""
        shp_paths = []
        try:
            with zipfile.ZipFile(zip_path) as z:
                shp_names = [f for f in z.namelist() if f.endswith(".shp")]
                for shp_name in shp_names:
                    shp_paths.append(f"zip://{zip_path}!{shp_name}")
        except Exception as e:
            logger.error(f"Error extracting from zip {zip_path}: {e}")
        return shp_paths

    def process_lake_data(self, records, lake_name):
        """Process and analyze lake data with OLS and Mann-Kendall"""
        df = pd.DataFrame(records, columns=['datetime', self.config['primary_attribute']])
        df = df.sort_values('datetime').drop_duplicates().reset_index(drop=True)
        
        if df.empty:
            return None
        
        if self.config['remove_outliers']:
            self.flag_outliers(df, self.config['primary_attribute'], self.config['outlier_method'])
        else:
            df["is_near_outlier_wse"] = False
            df["is_far_outlier_wse"] = False
        
        clean_name = lake_name.replace(' ', '_').replace(';', '_')
        data_path = self.data_dir / f"{clean_name}_WSE_data.csv"
        df_save = df.drop('trend_result', axis=1) if 'trend_result' in df.columns else df
        df_save.to_csv(data_path, index=False)
        
        original_df = df.copy()
        df = df[~df["is_far_outlier_wse"]]
        non_outliers = df[~df["is_near_outlier_wse"]]
        
        # Compute OLS trend
        trend_result = self.compute_trend(non_outliers["datetime"], non_outliers[self.config['primary_attribute']])
        
        # Compute Mann-Kendall test
        mk_result = self.compute_mann_kendall(non_outliers[self.config['primary_attribute']].values)
        
        # === Save Linear Plot (with OLS and Sen's slope) ===
        fig_linear = self.create_linear_plot(original_df, df, lake_name, trend_result, mk_result)
        plot_path_linear = self.linear_dir / f"{clean_name}_WSE_Analysis.png"
        fig_linear.savefig(plot_path_linear, dpi=300, bbox_inches='tight')
        plt.close(fig_linear)
        
        # === Save Combined Trends Plot ===
        fig_combined = self.create_combined_plot(df, lake_name)
        plot_path_combined = self.combined_dir / f"{clean_name}_Combined_Trends.png"
        fig_combined.savefig(plot_path_combined, dpi=300, bbox_inches='tight')
        plt.close(fig_combined)
        
        # Calculate Sen's slope annual change
        sen_annual = None
        if mk_result and len(non_outliers) > 1:
            span_days = (non_outliers['datetime'].max() - non_outliers['datetime'].min()).days
            if span_days > 0:
                sen_annual = mk_result.slope * (len(non_outliers) / span_days) * 365
        
        # Add to summary with OLS and Mann-Kendall results
        summary_entry = {
            'Lake': lake_name,
            'Status': 'Analyzed',
            'Total_Points': len(original_df),
            'Valid_Points': len(non_outliers),
            'Start_Date': df['datetime'].min().strftime('%Y-%m-%d') if len(df) > 0 else None,
            'End_Date': df['datetime'].max().strftime('%Y-%m-%d') if len(df) > 0 else None,
            # OLS results
            'OLS_Trend_m_per_year': round(trend_result.slope * 365, 6) if trend_result else None,
            'OLS_R_squared': round(trend_result.rvalue**2, 6) if trend_result else None,
            'OLS_P_Value': round(trend_result.pvalue, 8) if trend_result else None,
            'OLS_Significant': trend_result.pvalue < self.config['confidence_level'] if trend_result else None,
            # Mann-Kendall results
            'MK_Trend': mk_result.trend if mk_result else None,
            'MK_P_Value': round(mk_result.p, 8) if mk_result else None,
            'MK_Significant': mk_result.p < self.config['confidence_level'] if mk_result else None,
            'MK_Tau': round(mk_result.Tau, 4) if mk_result else None,
            'Sen_Slope_m_per_year': round(sen_annual, 6) if sen_annual is not None else None
        }
        self.summary_data.append(summary_entry)
        
        return df

    def save_results(self, df, lake_name):
        """Save trend analysis results with OLS and Mann-Kendall"""
        clean_name = lake_name.replace(' ', '_').replace(';', '_')
        trend_path = self.trends_dir / f"{clean_name}_WSE_trends.csv"
        
        non_outliers = df[~df["is_near_outlier_wse"]]
        
        # OLS results
        trend_result = self.compute_trend(non_outliers["datetime"], non_outliers[self.config['primary_attribute']])
        
        # Mann-Kendall results
        mk_result = self.compute_mann_kendall(non_outliers[self.config['primary_attribute']].values)
        
        # Calculate Sen's slope annual change
        sen_annual = None
        if mk_result and len(non_outliers) > 1:
            span_days = (non_outliers['datetime'].max() - non_outliers['datetime'].min()).days
            if span_days > 0:
                sen_annual = mk_result.slope * (len(non_outliers) / span_days) * 365
        
        with open(trend_path, 'w') as f:
            # Header
            f.write("Method,Slope_per_day,Annual_change_m_per_year,Intercept,R2,P_value,Significance,Tau\n")
            
            # OLS row
            if trend_result:
                annual_change = trend_result.slope * 365
                significance = "Significant" if trend_result.pvalue < self.config['confidence_level'] else "Not_Significant"
                f.write(f"OLS,{trend_result.slope:.8f},{annual_change:.6f},{trend_result.intercept:.3f},"
                       f"{trend_result.rvalue**2:.6f},{trend_result.pvalue:.6f},{significance},\n")
            
            # Mann-Kendall / Sen's row
            if mk_result:
                significance = "Significant" if mk_result.p < self.config['confidence_level'] else "Not_Significant"
                f.write(f"MannKendall,{mk_result.slope:.8f},{sen_annual:.6f},NA,NA,{mk_result.p:.6f},{significance},{mk_result.Tau:.4f}\n")
        
        logger.info(f"Trend results saved for {lake_name}")

    def generate_summary_report(self):
        """Generate main summary and highlight top 5 rising/falling lakes"""
        if not self.summary_data:
            logger.warning("No data to generate summary report.")
            return

        # === 1. Main Summary ===
        summary_df = pd.DataFrame(self.summary_data)
        summary_path = self.summary_dir / "lakes_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Main summary report saved to {summary_path}")

        # Filter valid lakes with trend data
        valid_df = summary_df[
            (summary_df['Status'] == 'Analyzed') & 
            (summary_df['OLS_Trend_m_per_year'].notna())
        ].copy()

        if valid_df.empty:
            logger.warning("No valid trend data for top lakes analysis.")
            return

        # Sort for top rising and falling (using OLS)
        top_rising = valid_df.nlargest(5, 'OLS_Trend_m_per_year')
        top_falling = valid_df.nsmallest(5, 'OLS_Trend_m_per_year')

        # Create top trends DataFrame
        top_rising['Rank'] = [f"Rising #{i+1}" for i in range(len(top_rising))]
        top_falling['Rank'] = [f"Falling #{i+1}" for i in range(len(top_falling))]

        top_combined = pd.concat([top_rising, top_falling], ignore_index=True)
        top_combined = top_combined[[
            'Rank', 'Lake', 'Total_Points', 'Valid_Points',
            'Start_Date', 'End_Date', 
            'OLS_Trend_m_per_year', 'OLS_R_squared', 'OLS_P_Value', 'OLS_Significant',
            'MK_Trend', 'MK_P_Value', 'MK_Significant', 'Sen_Slope_m_per_year'
        ]]

        # Rename columns for clarity
        top_combined.rename(columns={
            'OLS_Trend_m_per_year': 'OLS_Trend_mm_per_year',
            'Sen_Slope_m_per_year': 'Sen_Slope_mm_per_year'
        }, inplace=True)

        # Convert trends to mm/yr for readability
        top_combined['OLS_Trend_mm_per_year'] = (top_combined['OLS_Trend_mm_per_year'] * 1000).round(3)
        if 'Sen_Slope_mm_per_year' in top_combined.columns:
            top_combined['Sen_Slope_mm_per_year'] = (top_combined['Sen_Slope_mm_per_year'] * 1000).round(3)

        # Save top trends
        top_path = self.summary_dir / "top_trends_summary.csv"
        top_combined.to_csv(top_path, index=False)
        logger.info(f"Top trends summary saved to {top_path}")

        # Print to console
        print("\n" + "="*60)
        print("🏆 TOP 5 RISING LAKES (OLS Trend)")
        print("="*60)
        for _, row in top_rising.iterrows():
            mk_status = f"MK: {row['MK_Trend']}" if pd.notna(row.get('MK_Trend')) else ""
            print(f"{row['Lake'][:35]:<35} {row['OLS_Trend_m_per_year']:>+8.3f} m/yr  {mk_status}")

        print("\n" + "="*60)
        print("📉 TOP 5 FALLING LAKES (OLS Trend)")
        print("="*60)
        for _, row in top_falling.iterrows():
            mk_status = f"MK: {row['MK_Trend']}" if pd.notna(row.get('MK_Trend')) else ""
            print(f"{row['Lake'][:35]:<35} {row['OLS_Trend_m_per_year']:>+8.3f} m/yr  {mk_status}")

        # Also show Sen's slope results
        print("\n" + "="*60)
        print("📊 METHOD COMPARISON (Top 5 by Sen's Slope)")
        print("="*60)
        sen_valid = valid_df[valid_df['Sen_Slope_m_per_year'].notna()].copy()
        if not sen_valid.empty:
            sen_rising = sen_valid.nlargest(5, 'Sen_Slope_m_per_year')
            for _, row in sen_rising.iterrows():
                print(f"{row['Lake'][:35]:<35} Sen: {row['Sen_Slope_m_per_year']:>+8.3f} m/yr  OLS: {row['OLS_Trend_m_per_year']:>+8.3f} m/yr")

    def run_analysis(self):
        """Main analysis runner"""
        input_path = Path(self.config['input_directory'])
        if not input_path.exists():
            logger.error(f"Input directory does not exist: {input_path}")
            return
        
        processed_count = 0
        for folder_path in input_path.iterdir():
            if folder_path.is_dir():
                try:
                    self.process_lake_folder(folder_path)
                    processed_count += 1
                except Exception as e:
                    logger.error(f"Error processing {folder_path.name}: {e}")
                    self.summary_data.append({
                        'Lake': folder_path.name, 'Status': 'Error'
                    })

        self.generate_summary_report()
        logger.info(f"Analysis complete. Processed {processed_count} lakes.")


# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    config = {
        'input_directory': input_directory,
        'output_directory': output_directory,
        'primary_attribute': PRIMARY_ATTRIBUTE,
        'primary_label': PRIMARY_LABEL,
        'primary_unit': PRIMARY_UNIT,
        'filter_positive': FILTER_POSITIVE_ONLY,
        'case_sensitive': CASE_SENSITIVE_SEARCH,
        'remove_outliers': REMOVE_WSE_OUTLIERS,
        'min_data_points': MIN_DATA_POINTS,
        'outlier_method': OUTLIER_METHOD,
        'confidence_level': TREND_CONFIDENCE_LEVEL
    }
    
    analyzer = LakeAnalyzer(config)
    analyzer.run_analysis()