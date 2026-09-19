import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from pathlib import Path
from scipy.stats import linregress
import math

# Optional imports
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
input_directory = "River_nodes_WSE_Data/Data"
output_directory = "River_nodes_WSE_Data"

PRIMARY_ATTRIBUTE = "wse"
PRIMARY_LABEL = "Water Surface Elevation (m)"
PRIMARY_UNIT = "m"
FILTER_POSITIVE_ONLY = True
REMOVE_WSE_OUTLIERS = True
MIN_DATA_POINTS = 3
OUTLIER_METHOD = "iqr"
TREND_CONFIDENCE_LEVEL = 0.05
MIN_FOR_STL = 12
MIN_FOR_PIECEWISE = 4
# ==============================


class RiverReachAnalyzer:
    def __init__(self, config):
        self.config = config
        self.setup_directories()
        self.summary_data = []
    
    def setup_directories(self):
        self.input_dir = Path(self.config['input_directory'])
        self.output_dir = Path(self.config['output_directory'])
        self.plots_dir = self.output_dir / "Plots"
        self.data_dir = self.output_dir / "Data"
        self.trends_dir = self.output_dir / "Trends"
        self.summary_dir = self.output_dir / "Summary"
        self.linear_dir = self.plots_dir / "Linear_Trend"
        self.combined_dir = self.plots_dir / "Combined_Trends"

        for dir_path in [self.input_dir, self.output_dir, self.plots_dir, self.data_dir,
                        self.trends_dir, self.summary_dir, self.linear_dir, self.combined_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)
    
    def parse_datetime(self, time_str):
        try:
            return pd.to_datetime(time_str, format='%m/%d/%Y %H:%M')
        except Exception:
            try:
                return pd.to_datetime(time_str)
            except Exception:
                return None

    def flag_outliers(self, df, colname, method="iqr"):
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
        if len(x) < 2:
            return None
        try:
            x_num = np.array([d.toordinal() for d in x])
            result = linregress(x_num, y)
            return result
        except Exception as e:
            logger.error(f"Error in trend computation: {e}")
            return None

    def _safe_r2(self, y_true, y_pred):
        """Robust R² that avoids division by zero"""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.size == 0 or y_pred.size == 0:
            return 0.0
        var_true = np.var(y_true)
        var_pred = np.var(y_pred)
        if var_true == 0 and var_pred == 0:
            return 1.0  # Both constant and equal
        if var_true == 0 or var_pred == 0:
            return 0.0  # No variability → no correlation
        corr_matrix = np.corrcoef(y_true, y_pred)
        corr = corr_matrix[0, 1]
        return float(corr ** 2) if not np.isnan(corr) else 0.0

    def create_linear_plot(self, original_df, df, identifier, trend_result):
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        non_outliers = df[~df["is_near_outlier_wse"]]
        near_outliers = df[df["is_near_outlier_wse"]]
        plotted_df = pd.concat([non_outliers, near_outliers])
        
        ax1.plot(non_outliers["datetime"], non_outliers[self.config['primary_attribute']], 
                 marker="o", color="blue", linewidth=2, markersize=6,
                 label=self.config['primary_label'], zorder=5)
        
        if not near_outliers.empty:
            ax1.scatter(near_outliers["datetime"], near_outliers[self.config['primary_attribute']],
                        color="red", marker="x", s=100, linewidth=3,
                        label=f"{self.config['primary_attribute'].upper()} Outliers", zorder=6)
        
        if trend_result and len(non_outliers) > 1:
            x_vals = np.array([d.toordinal() for d in non_outliers["datetime"]])
            y_fit = trend_result.intercept + trend_result.slope * x_vals
            ax1.plot(non_outliers["datetime"], y_fit, 
                     color="navy", linestyle="--", linewidth=2,
                     label="WSE Trend", zorder=3)
        
        ax1.set_xlabel("Date", fontsize=12, fontweight='bold')
        ax1.set_ylabel(self.config['primary_label'], color="blue", fontsize=12, fontweight='bold')
        ax1.tick_params(axis='x', labelsize=10, rotation=45)
        ax1.tick_params(axis='y', labelsize=10, labelcolor="blue")
        for label in ax1.get_xticklabels() + ax1.get_yticklabels():
            label.set_fontweight('bold')
        
        y_min = plotted_df[self.config['primary_attribute']].min()
        y_max = plotted_df[self.config['primary_attribute']].max()
        bottom_buffer = 0.10
        top_buffer = 0.03
        y_min_pad = y_min - bottom_buffer
        y_max_pad = y_max + top_buffer
        ax1.set_ylim(y_min_pad, y_max_pad)

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
        
        stats_text = self.create_stats_text(original_df, non_outliers, trend_result)
        ax1.text(0.02, 0.02, stats_text, transform=ax1.transAxes,
                 verticalalignment='bottom', horizontalalignment='left',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                 fontsize=10)
        
        ax1.legend(loc="lower right", fontsize=11)
        plt.title(f"Temporal Analysis: {identifier}", fontsize=16, fontweight="bold", pad=20)
        plt.tight_layout()
        return fig

    def create_combined_plot(self, df, identifier):
        fig, ax = plt.subplots(figsize=(12, 7))
        non_outliers = df[~df["is_near_outlier_wse"]].sort_values("datetime")
        near_outliers = df[df["is_near_outlier_wse"]]
        plotted_df = pd.concat([non_outliers, near_outliers])
        
        if len(non_outliers) < 2:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha='center', va='center', fontsize=14)
            ax.set_title(f"Combined Trends: {identifier}")
            plt.tight_layout()
            return fig

        ax.plot(non_outliers["datetime"], non_outliers[self.config['primary_attribute']], 
                marker="o", color="blue", linewidth=2, markersize=5, label="WSE", zorder=5)
        if not near_outliers.empty:
            ax.scatter(near_outliers["datetime"], near_outliers[self.config['primary_attribute']],
                       color="red", marker="x", s=60, linewidth=2, label="Outliers", zorder=6)
        
        trend_results = self.get_trend_predictions(non_outliers)
        for res in trend_results.values():
            ax.plot(non_outliers["datetime"], res['y_pred'],
                    color=res['color'], linestyle=res['style'], linewidth=2,
                    label=res['label'], zorder=3)
        
        ax.set_title(f"Combined Trends: {identifier}", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Date", fontsize=12, fontweight='bold')
        ax.set_ylabel(self.config['primary_label'], color="blue", fontsize=12, fontweight='bold')
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        ax.tick_params(axis='x', labelsize=10, rotation=45)
        ax.tick_params(axis='y', labelsize=10, labelcolor="blue")
        ax.grid(True, alpha=0.6)
        ax.legend(loc='upper left', fontsize=9, ncol=2)

        y_min = plotted_df[self.config['primary_attribute']].min()
        y_max = plotted_df[self.config['primary_attribute']].max()
        bottom_buffer = 0.10
        top_buffer = 0.03
        y_min_pad = y_min - bottom_buffer
        y_max_pad = y_max + top_buffer
        ax.set_ylim(y_min_pad, y_max_pad)

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
        results = {}
        x_days = np.array([d.toordinal() for d in non_outliers["datetime"]])
        y_vals = non_outliers[self.config['primary_attribute']].values

        # Linear
        linear_trend = self.compute_trend(non_outliers["datetime"], y_vals)
        if linear_trend:
            y_pred = linear_trend.intercept + linear_trend.slope * x_days
            r2 = linear_trend.rvalue ** 2
            results['Linear'] = {
                'y_pred': y_pred,
                'r2': r2,
                'label': f"Linear (R²={r2:.3f})",
                'color': 'navy',
                'style': '--'
            }

        # Polynomial (Quadratic)
        if len(non_outliers) >= 3:
            try:
                coeffs = np.polyfit(x_days, y_vals, deg=2)
                poly = np.poly1d(coeffs)
                y_pred = poly(x_days)
                r2 = self._safe_r2(y_vals, y_pred)
                results['Polynomial'] = {
                    'y_pred': y_pred,
                    'r2': r2,
                    'label': f"Quadratic (R²={r2:.3f})",
                    'color': 'green',
                    'style': '-.'
                }
            except Exception as e:
                logger.debug(f"Polynomial fit failed: {e}")

        # Moving Average
        window = min(3, len(non_outliers) // 2 or 1)
        ma = non_outliers[self.config['primary_attribute']].rolling(window=window, center=True).mean()
        if not ma.isna().all():
            ma_clean = ma.dropna()
            y_clean = non_outliers.loc[ma_clean.index, self.config['primary_attribute']]
            r2 = self._safe_r2(y_clean, ma_clean)
            results['Moving Average'] = {
                'y_pred': ma.values,
                'r2': r2,
                'label': f"MA (w={window})",
                'color': 'purple',
                'style': '-'
            }

        # Piecewise
        if PWLF_AVAILABLE and len(non_outliers) >= MIN_FOR_PIECEWISE:
            try:
                my_pwlf = pwlf.PiecewiseLinFit(x_days, y_vals)
                breaks = my_pwlf.fit(2)
                y_pred = my_pwlf.predict(x_days)
                r2 = self._safe_r2(y_vals, y_pred)
                results['Piecewise'] = {
                    'y_pred': y_pred,
                    'r2': r2,
                    'label': f"Piecewise (R²={r2:.3f})",
                    'color': 'orange',
                    'style': ':'
                }
            except Exception as e:
                logger.debug(f"Piecewise fit failed: {e}")

        # STL Trend
        if STL_AVAILABLE and len(non_outliers) >= MIN_FOR_STL:
            try:
                df_temp = non_outliers.set_index("datetime")[self.config['primary_attribute']].sort_index()
                df_monthly = df_temp.resample('ME').mean().dropna()
                if len(df_monthly) >= MIN_FOR_STL:
                    stl = STL(df_monthly, seasonal=13, period=12).fit()
                    # Interpolate STL trend to daily
                    dates_daily = non_outliers["datetime"].sort_values()
                    stl_daily = np.interp(
                        [d.timestamp() for d in dates_daily],
                        [d.timestamp() for d in stl.trend.index],
                        stl.trend.values
                    )
                    r2 = self._safe_r2(y_vals, stl_daily)
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

    def create_stats_text(self, df, non_outliers, trend_result):
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
        
        if trend_result and len(non_outliers) > 1:
            annual_change = trend_result.slope * 365
            significance = "Significant" if trend_result.pvalue < self.config['confidence_level'] else "Not Significant"
            stats.extend([
                f"Annual trend: {annual_change:.3f} {self.config['primary_unit']}/yr",
                f"R² = {trend_result.rvalue**2:.3f}",
                f"p-value = {trend_result.pvalue:.4f} ({significance})"
            ])
        
        return "\n".join(stats)

    def process_csv_file(self, file_path):
        filename = file_path.stem
        logger.info(f"Processing file: {filename}")

        try:
            df_raw = pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            self.summary_data.append({'Reach_ID': filename, 'Status': 'Read Error'})
            return

        if 'time' not in df_raw.columns or 'wse' not in df_raw.columns:
            logger.warning(f"'time' or 'wse' column missing in {file_path}")
            self.summary_data.append({'Reach_ID': filename, 'Status': 'Missing Columns'})
            return

        df_raw['datetime'] = df_raw['time'].apply(self.parse_datetime)
        df_raw = df_raw.dropna(subset=['datetime'])
        df_raw = df_raw[['datetime', 'wse']].sort_values('datetime').drop_duplicates().reset_index(drop=True)

        # Remove invalid (-1e12) and optionally non-positive
        df_clean = df_raw[df_raw['wse'] != -1.0e12].copy()
        if self.config['filter_positive']:
            df_clean = df_clean[df_clean['wse'] > 0].copy()

        if len(df_clean) < self.config['min_data_points']:
            logger.warning(f"Insufficient valid data for {filename} ({len(df_clean)} points)")
            self.summary_data.append({
                'Reach_ID': filename,
                'Status': 'Insufficient Data',
                'Total_Points': len(df_raw),
                'Valid_Points': len(df_clean)
            })
            return

        if self.config['remove_outliers']:
            self.flag_outliers(df_clean, 'wse', self.config['outlier_method'])
        else:
            df_clean["is_near_outlier_wse"] = False
            df_clean["is_far_outlier_wse"] = False

        data_path = self.data_dir / f"{filename}_WSE_data.csv"
        df_clean.to_csv(data_path, index=False)

        df_for_analysis = df_clean[~df_clean["is_far_outlier_wse"]].copy()
        non_outliers = df_for_analysis[~df_for_analysis["is_near_outlier_wse"]]

        trend_result = self.compute_trend(non_outliers["datetime"], non_outliers["wse"])

        fig_linear = self.create_linear_plot(df_clean, df_for_analysis, filename, trend_result)
        plot_path_linear = self.linear_dir / f"{filename}_WSE_Analysis.png"
        fig_linear.savefig(plot_path_linear, dpi=300, bbox_inches='tight')
        plt.close(fig_linear)

        fig_combined = self.create_combined_plot(df_for_analysis, filename)
        plot_path_combined = self.combined_dir / f"{filename}_Combined_Trends.png"
        fig_combined.savefig(plot_path_combined, dpi=300, bbox_inches='tight')
        plt.close(fig_combined)

        trend_path = self.trends_dir / f"{filename}_WSE_trends.csv"
        with open(trend_path, 'w') as f:
            f.write("Attribute,Slope_per_day,Annual_change,Intercept,R2,P_value,Significance\n")
            if trend_result:
                annual_change = trend_result.slope * 365
                significance = "Significant" if trend_result.pvalue < self.config['confidence_level'] else "Not_Significant"
                f.write(f"WSE,{trend_result.slope:.8f},{annual_change:.6f},{trend_result.intercept:.3f},"
                       f"{trend_result.rvalue**2:.6f},{trend_result.pvalue:.6f},{significance}\n")

        self.summary_data.append({
            'Reach_ID': filename,
            'Status': 'Analyzed',
            'Total_Points': len(df_clean),
            'Valid_Points': len(non_outliers),
            'Start_Date': df_for_analysis['datetime'].min().strftime('%Y-%m-%d') if len(df_for_analysis) > 0 else None,
            'End_Date': df_for_analysis['datetime'].max().strftime('%Y-%m-%d') if len(df_for_analysis) > 0 else None,
            'Annual_Trend_m_per_year': round(trend_result.slope * 365, 6) if trend_result else None,
            'R_squared': round(trend_result.rvalue**2, 6) if trend_result else None,
            'P_Value': round(trend_result.pvalue, 8) if trend_result else None,
            'Significant': trend_result.pvalue < self.config['confidence_level'] if trend_result else None
        })

    def generate_summary_report(self):
        if not self.summary_data:
            logger.warning("No data to generate summary report.")
            return

        summary_df = pd.DataFrame(self.summary_data)
        summary_path = self.summary_dir / "reaches_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Main summary report saved to {summary_path}")

        valid_df = summary_df[
            (summary_df['Status'] == 'Analyzed') & 
            (summary_df['Annual_Trend_m_per_year'].notna())
        ].copy()

        if valid_df.empty:
            logger.warning("No valid trend data for top reaches analysis.")
            return

        top_rising = valid_df.nlargest(5, 'Annual_Trend_m_per_year')
        top_falling = valid_df.nsmallest(5, 'Annual_Trend_m_per_year')

        top_rising['Rank'] = [f"Rising #{i+1}" for i in range(len(top_rising))]
        top_falling['Rank'] = [f"Falling #{i+1}" for i in range(len(top_falling))]
        top_combined = pd.concat([top_rising, top_falling], ignore_index=True)

        top_combined.rename(columns={
            'Annual_Trend_m_per_year': 'Trend_mm_per_year',
            'Significant': 'Stat_Significant'
        }, inplace=True)
        top_combined['Trend_mm_per_year'] = (top_combined['Trend_mm_per_year'] * 1000).round(3)

        top_combined = top_combined[[
            'Rank', 'Reach_ID', 'Total_Points', 'Valid_Points',
            'Start_Date', 'End_Date', 'Trend_mm_per_year',
            'R_squared', 'P_Value', 'Stat_Significant'
        ]]

        top_path = self.summary_dir / "top_trends_summary.csv"
        top_combined.to_csv(top_path, index=False)
        logger.info(f"Top trends summary saved to {top_path}")

        print("\n" + "="*60)
        print("🏆 TOP 5 RISING RIVER REACHES (m/year)")
        print("="*60)
        for _, row in top_rising.iterrows():
            print(f"{row['Reach_ID']:<40} {row['Annual_Trend_m_per_year']:>+8.3f} m/yr (R²={row['R_squared']:.3f})")

        print("\n" + "="*60)
        print("📉 TOP 5 FALLING RIVER REACHES (m/year)")
        print("="*60)
        for _, row in top_falling.iterrows():
            print(f"{row['Reach_ID']:<40} {row['Annual_Trend_m_per_year']:>+8.3f} m/yr (R²={row['R_squared']:.3f})")

    def run_analysis(self):
        input_path = Path(self.config['input_directory'])
        if not input_path.exists():
            logger.error(f"Input directory does not exist: {input_path}")
            return

        csv_files = list(input_path.glob("*.csv"))
        if not csv_files:
            logger.error(f"No CSV files found in {input_path}")
            return

        logger.info(f"Found {len(csv_files)} CSV files. Starting analysis...")
        for file_path in csv_files:
            try:
                self.process_csv_file(file_path)
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {e}")
                self.summary_data.append({'Reach_ID': file_path.stem, 'Status': 'Processing Error'})

        self.generate_summary_report()
        logger.info("Analysis complete.")


# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    config = {
        'input_directory': input_directory,
        'output_directory': output_directory,
        'primary_attribute': PRIMARY_ATTRIBUTE,
        'primary_label': PRIMARY_LABEL,
        'primary_unit': PRIMARY_UNIT,
        'filter_positive': FILTER_POSITIVE_ONLY,
        'remove_outliers': REMOVE_WSE_OUTLIERS,
        'min_data_points': MIN_DATA_POINTS,
        'outlier_method': OUTLIER_METHOD,
        'confidence_level': TREND_CONFIDENCE_LEVEL
    }
    
    analyzer = RiverReachAnalyzer(config)
    analyzer.run_analysis()