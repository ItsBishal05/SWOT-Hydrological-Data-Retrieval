import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import math
from pathlib import Path
from scipy.stats import linregress
from scipy.stats import kendalltau
from scipy.stats import norm
from scipy.stats import pearsonr

# ================= USER SETTINGS =================
CSV_DIR = os.path.join("River_Output", "csv")
PLOT_DIR = os.path.join("River_Output", "plots")

# Create main plot directory if it doesn't exist
os.makedirs(PLOT_DIR, exist_ok=True)

PRIMARY_ATTRIBUTE = "wse"
PRIMARY_LABEL = "Water Surface Elevation (m)"
PRIMARY_UNIT = "m"

OUTLIER_METHOD = "iqr"      
MIN_POINTS = 3
CONFIDENCE_LEVEL = 0.05
NULL_VALUE = -999999999999  # Define the null value to filter out

# Filter for node_q values (only 0 and 1 will be kept)
NODE_Q_FILTER = [0, 1]

# -------------------------------------------------
# MODIFIED MANN-KENDALL TEST WITH AUTOCORRELATION
# -------------------------------------------------
def modified_mann_kendall_test(times, values):
    """
    Perform Modified Mann-Kendall trend test with autocorrelation correction
    (Hamed and Rao, 1998 method)
    
    Returns: tau, p_value, trend_direction, trend_magnitude, corrected_variance
    """
    # Convert to numpy arrays
    times_array = np.array(times)
    values_array = np.array(values)
    n = len(values_array)
    
    # Step 1: Calculate Kendall's tau
    tau, p_value_original = kendalltau(times_array, values_array)
    
    # Step 2: Calculate Sen's slope (trend magnitude)
    slopes = []
    for i in range(n-1):
        for j in range(i+1, n):
            if times_array[j] != times_array[i]:
                slope = (values_array[j] - values_array[i]) / (times_array[j] - times_array[i])
                slopes.append(slope)
    
    sen_slope = np.median(slopes) if slopes else 0
    
    # Step 3: Remove trend from the data to get residuals
    # Use Sen's slope to detrend the data
    if sen_slope != 0:
        # Detrend using Sen's slope
        residuals = values_array - sen_slope * (times_array - times_array[0])
    else:
        residuals = values_array - np.mean(values_array)
    
    # Step 4: Calculate autocorrelation coefficients for lags 1 to (n/4)
    max_lag = min(n // 4, 20)  # Use up to n/4 lags or 20, whichever is smaller
    autocorr_coeffs = []
    significant_lags = []
    
    for lag in range(1, max_lag + 1):
        if len(residuals) > lag:
            # Calculate autocorrelation for this lag
            r_lag = pearsonr(residuals[:-lag], residuals[lag:])[0]
            autocorr_coeffs.append(r_lag)
            
            # Check if autocorrelation is significant at 95% confidence level
            # Approximate standard error: 1/sqrt(n)
            se_auto = 1.0 / np.sqrt(n)
            if abs(r_lag) > 1.96 * se_auto:
                significant_lags.append(lag)
    
    # Step 5: Calculate variance correction factor
    n_eff = n / (1 + 2 * sum(autocorr_coeffs[:len(significant_lags)]))
    
    # Step 6: Calculate corrected variance
    # Original variance for Mann-Kendall
    var_s_original = (n * (n - 1) * (2 * n + 5)) / 18
    
    # Apply correction factor
    if len(significant_lags) > 0:
        correction_factor = n / n_eff
        var_s_corrected = var_s_original * correction_factor
    else:
        var_s_corrected = var_s_original
    
    # Step 7: Calculate test statistic
    # S statistic (Mann-Kendall)
    S = 0
    for i in range(n-1):
        for j in range(i+1, n):
            S += np.sign(values_array[j] - values_array[i])
    
    # Standard deviation of S
    std_s = np.sqrt(var_s_corrected)
    
    # Z statistic
    if S > 0:
        Z = (S - 1) / std_s
    elif S < 0:
        Z = (S + 1) / std_s
    else:
        Z = 0
    
    # Two-tailed p-value
    p_value = 2 * (1 - norm.cdf(abs(Z)))
    
    # Determine trend direction
    if p_value < CONFIDENCE_LEVEL:
        if sen_slope > 0:
            trend_direction = "Increasing"
        elif sen_slope < 0:
            trend_direction = "Decreasing"
        else:
            trend_direction = "No trend"
    else:
        trend_direction = "No significant trend"
    
    return tau, p_value, trend_direction, sen_slope, var_s_corrected, n_eff, len(significant_lags)

# -------------------------------------------------
# OUTLIER DETECTION
# -------------------------------------------------
def iqr_outlier_classification(df, col, k_near=0.75, k_far=1):
    """
    Classify outliers into near and far based on IQR
    - Near outliers: k_near*IQR < deviation <= k_far*IQR
    - Far outliers: deviation > k_far*IQR
    """
    # Filter out null values before calculating quartiles
    valid_data = df[df[col] != NULL_VALUE][col]
    
    if len(valid_data) == 0:
        df["is_near_outlier"] = False
        df["is_far_outlier"] = False
        return df
    
    Q1 = valid_data.quantile(0.25)
    Q3 = valid_data.quantile(0.75)
    IQR = Q3 - Q1

    lower_far = Q1 - k_far * IQR
    upper_far = Q3 + k_far * IQR
    lower_near = Q1 - k_near * IQR
    upper_near = Q3 + k_near * IQR

    # Initialize columns
    df["is_near_outlier"] = False
    df["is_far_outlier"] = False

    # Far outliers (excluding null values)
    far_outlier_mask = ((df[col] < lower_far) | (df[col] > upper_far)) & (df[col] != NULL_VALUE)
    df.loc[far_outlier_mask, "is_far_outlier"] = True

    # Near outliers (between k_near*IQR and k_far*IQR, excluding null values)
    near_outlier_mask = (
        ((df[col] < lower_near) & (df[col] >= lower_far)) |
        ((df[col] > upper_near) & (df[col] <= upper_far))
    ) & (df[col] != NULL_VALUE)
    
    df.loc[near_outlier_mask, "is_near_outlier"] = True

    return df

# -------------------------------------------------
# LINEAR TREND
# -------------------------------------------------
def compute_trend(dates, values):
    x = np.array([d.toordinal() for d in dates])
    return linregress(x, values)

# -------------------------------------------------
# MAIN PLOT FUNCTION
# -------------------------------------------------
def plot_node(df, node_name, river_name, output_path):
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Filter for node_q values 0 and 1
    if "node_q" in df.columns:
        initial_count = len(df)
        df = df[df["node_q"].isin(NODE_Q_FILTER)].copy()
        node_q_removed = initial_count - len(df)
        print(f"   Filtered by node_q: kept {len(df)} points (removed {node_q_removed} points with node_q not in {NODE_Q_FILTER})")
    else:
        print(f"   Warning: 'node_q' column not found. Using all data.")
        node_q_removed = 0
    
    # Filter out null values from the beginning
    initial_count = len(df)
    df = df[df[PRIMARY_ATTRIBUTE] != NULL_VALUE].copy()
    
    if len(df) == 0:
        print(f"⚠️ Skipping {node_name} ({river_name}): All WSE values are null ({NULL_VALUE})")
        return
    
    null_removed = initial_count - len(df)
    
    # Sort by time
    df = df.sort_values("time_str").reset_index(drop=True)
    
    # Apply outlier classification (will skip null values automatically)
    df = iqr_outlier_classification(df, PRIMARY_ATTRIBUTE, k_near=1.0, k_far=1.5)

    # Remove far outliers completely
    df_no_far = df[~df["is_far_outlier"]].copy()
    
    # Reset index after filtering
    df_no_far = df_no_far.reset_index(drop=True)
    
    # Data for analysis (all points except far outliers - includes near outliers)
    analysis_data = df_no_far.copy()
    
    # Non-outliers only (for plotting as solid line)
    non_outliers = df_no_far[~df_no_far["is_near_outlier"]].copy()

    if len(analysis_data) < MIN_POINTS:
        print(f"⚠️ Skipping {node_name} ({river_name}): insufficient data after outlier removal ({len(analysis_data)} points)")
        return

    # Linear regression on all valid points (including near outliers)
    linear_trend = compute_trend(analysis_data["time_str"], analysis_data[PRIMARY_ATTRIBUTE])
    
    # Modified Mann-Kendall test on all valid points (including near outliers)
    times_numeric = np.array([d.toordinal() for d in analysis_data["time_str"]])
    values_numeric = np.array(analysis_data[PRIMARY_ATTRIBUTE])
    
    mk_tau, mk_pvalue, mk_direction, mk_slope, mk_corrected_var, mk_eff_n, mk_sig_lags = modified_mann_kendall_test(
        times_numeric, values_numeric
    )

    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot non-outliers with solid line
    if len(non_outliers) > 0:
        ax.plot(
            non_outliers["time_str"],
            non_outliers[PRIMARY_ATTRIBUTE],
            marker="o",
            linewidth=2,
            markersize=6,
            label=PRIMARY_LABEL,
            color="blue"
        )

    # Plot near outliers as red x markers
    near = df_no_far[df_no_far["is_near_outlier"]]
    if not near.empty:
        ax.scatter(
            near["time_str"],
            near[PRIMARY_ATTRIBUTE],
            color="red",
            marker="x",
            s=80,
            label="Near Outliers"
        )

    # Linear trend line (using all valid data including near outliers)
    if len(analysis_data) > 1:
        x_ord = np.array([d.toordinal() for d in analysis_data["time_str"]])
        y_fit = linear_trend.intercept + linear_trend.slope * x_ord
        ax.plot(
            analysis_data["time_str"],
            y_fit,
            linestyle="--",
            linewidth=2,
            color="green",
            label="Linear Trend"
        )

    # === AXIS FIX ===
    if len(df_no_far) > 0:
        y_min = df_no_far[PRIMARY_ATTRIBUTE].min()
        y_max = df_no_far[PRIMARY_ATTRIBUTE].max()
        y_range = y_max - y_min
        
        if y_range > 0:
            ax.set_ylim(y_min - 0.10, y_max + 0.03)
            
            # Dynamic y-ticks
            interval = max(round(y_range / 4, 2), 0.1)
            ticks = np.arange(
                math.floor((y_min - 0.1) / interval) * interval,
                math.ceil((y_max + 0.03) / interval) * interval + interval,
                interval
            )
            ax.set_yticks(ticks)
    
    ax.grid(True, which="major", alpha=0.7)

    # Labels
    ax.set_xlabel("Date", fontweight="bold")
    ax.set_ylabel(PRIMARY_LABEL, fontweight="bold")
    ax.set_title(f"Temporal WSE Analysis: Node {node_name} - {river_name}", fontweight="bold", fontsize=15)

    # Stats box with both linear and Modified Mann-Kendall trends
    linear_annual_change = linear_trend.slope * 365 if hasattr(linear_trend, 'slope') else 0
    mk_annual_change = mk_slope * 365
    
    linear_significance = "Significant" if linear_trend.pvalue < CONFIDENCE_LEVEL else "Not Significant"
    mk_significance = "Significant" if mk_pvalue < CONFIDENCE_LEVEL else "Not Significant"

    # Get node_q values for display
    node_q_values = df_no_far["node_q"].unique().tolist() if "node_q" in df_no_far.columns else []

    stats_text = (
        f"Node ID: {node_name}\n"
        f"River: {river_name}\n"
        f"Total points: {initial_count + null_removed + node_q_removed}\n"
        f"Node_q filter: {node_q_values}\n"
        f"Null values removed: {null_removed}\n"
        f"Valid points: {len(analysis_data)}\n"
        f"Far outliers removed: {len(df[df['is_far_outlier']])}\n"
        f"Near outliers: {len(near)}\n"
        f"WSE range: {analysis_data[PRIMARY_ATTRIBUTE].min():.2f} – "
        f"{analysis_data[PRIMARY_ATTRIBUTE].max():.2f} {PRIMARY_UNIT}\n"
        f"\n--- Linear Regression ---\n"
        f"Annual trend: {linear_annual_change:.3f} {PRIMARY_UNIT}/yr\n"
        f"R² = {linear_trend.rvalue**2:.3f}\n"
        f"p-value = {linear_trend.pvalue:.4f} ({linear_significance})\n"
        f"\n--- Modified Mann-Kendall ---\n"
        f"Direction: {mk_direction}\n"
        f"Kendall's τ = {mk_tau:.3f}\n"
        f"Sen's slope: {mk_annual_change:.3f} {PRIMARY_UNIT}/yr\n"
        f"p-value = {mk_pvalue:.4f} ({mk_significance})\n"
        f"Effective n = {mk_eff_n:.1f}\n"
        f"Significant lags = {mk_sig_lags}"
    )

    ax.text(
        0.02, 0.02, stats_text,
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.85),
        fontsize=9,
        verticalalignment='bottom'
    )

    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

# -------------------------------------------------
# LOOP OVER ALL NODE CSVs
# -------------------------------------------------
csv_files = sorted(Path(CSV_DIR).glob("*.csv"))

for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)
        
        # Check if required columns exist
        required_cols = ["node_id", "river_name", "time_str", "wse"]
        if not all(col in df.columns for col in required_cols):
            print(f"⚠️ Skipping {csv_file.name}: Missing required columns")
            continue
            
        df["time_str"] = pd.to_datetime(df["time_str"])
        
        # Get node information
        node_id = str(df["node_id"].iloc[0])
        river_name = str(df["river_name"].iloc[0]) if "river_name" in df.columns else "Unknown"
        
        # Clean names for file system
        clean_river_name = river_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        
        # Create output file path directly in PLOT_DIR
        out_file = Path(PLOT_DIR) / f"node_{node_id}_{clean_river_name}_WSE.png"
        
        print(f"📊 Plotting Node {node_id} ({river_name}) - {len(df)} points")
        plot_node(df, node_id, river_name, out_file)
        
    except Exception as e:
        print(f"❌ Error processing {csv_file.name}: {str(e)}")
        continue

print(f"\n✅ All node plots generated in: {PLOT_DIR}")