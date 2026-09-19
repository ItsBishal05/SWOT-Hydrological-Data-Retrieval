import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import math
from pathlib import Path
from scipy.stats import linregress
from scipy.stats import kendalltau
from scipy.stats import norm

# ================= USER SETTINGS =================
CSV_DIR = os.path.join("Lake_Output", "csv")
PLOT_DIR = os.path.join("Lake_Output", "plots")

# Create folders if they do not exist
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

PRIMARY_ATTRIBUTE = "wse"
PRIMARY_LABEL = "Water Surface Elevation (m)"
PRIMARY_UNIT = "m"

OUTLIER_METHOD = "iqr"      
MIN_POINTS = 3
CONFIDENCE_LEVEL = 0.05

# ==================================================
# MODIFIED MANN-KENDALL TEST (Hamed & Rao, 1998)
# ==================================================

def mann_kendall_modified(times, values):
    """
    Modified Mann-Kendall test (Hamed & Rao, 1998)
    Accounts for autocorrelation in the data
    """
    n = len(values)
    
    # Calculate Kendall's tau
    tau, p_value = kendalltau(times, values)
    
    # Calculate variance with autocorrelation correction
    # Step 1: Calculate the variance of S (standard MK)
    var_s = n * (n - 1) * (2*n + 5) / 18
    
    # Step 2: Calculate autocorrelation at different lags
    max_lag = min(n - 1, 30)  # Limit to 30 lags or n-1
    autocorr = []
    for lag in range(1, max_lag + 1):
        if len(values) > lag:
            # Calculate autocorrelation at this lag
            corr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
            if not np.isnan(corr):
                autocorr.append(corr)
    
    # Step 3: Calculate effective sample size correction
    if autocorr:
        # Variance correction factor
        n_eff = n / (1 + 2 * sum(autocorr))
        var_s_corrected = var_s * (n_eff / n)
    else:
        var_s_corrected = var_s
    
    # Calculate Z statistic
    if tau > 0:
        z = (tau - 1/n) / np.sqrt(var_s_corrected / (n*(n-1)/2))
    elif tau < 0:
        z = (tau + 1/n) / np.sqrt(var_s_corrected / (n*(n-1)/2))
    else:
        z = 0
    
    # Calculate p-value
    p_value_corrected = 2 * (1 - norm.cdf(abs(z)))
    
    # Calculate Sen's slope (median of all pairwise slopes)
    slopes = []
    for i in range(n-1):
        for j in range(i+1, n):
            if times[j] != times[i]:
                slope = (values[j] - values[i]) / (times[j] - times[i])
                slopes.append(slope)
    
    sen_slope = np.median(slopes) if slopes else 0
    
    return tau, p_value_corrected, sen_slope, var_s_corrected

# ==================================================
# OUTLIER DETECTION
# ==================================================

def iqr_outlier_classification(df, col, k_near=0.75, k_far=1):
    """
    Classify outliers into near and far based on IQR
    - Near outliers: k_near*IQR < deviation <= k_far*IQR
    - Far outliers: deviation > k_far*IQR
    """
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_far = Q1 - k_far * IQR
    upper_far = Q3 + k_far * IQR
    lower_near = Q1 - k_near * IQR
    upper_near = Q3 + k_near * IQR

    # Initialize columns
    df["is_near_outlier"] = False
    df["is_far_outlier"] = False

    # Far outliers
    df.loc[(df[col] < lower_far) | (df[col] > upper_far), "is_far_outlier"] = True

    # Near outliers (between k_near*IQR and k_far*IQR)
    df.loc[
        ((df[col] < lower_near) & (df[col] >= lower_far)) |
        ((df[col] > upper_near) & (df[col] <= upper_far)),
        "is_near_outlier"
    ] = True

    return df

# ==================================================
# LINEAR TREND
# ==================================================

def compute_trend(dates, values):
    x = np.array([d.toordinal() for d in dates])
    return linregress(x, values)

# ==================================================
# MAIN PLOT FUNCTION
# ==================================================

def plot_lake(df, lake_name, output_path):
    # Make a copy to avoid modifying the original
    df = df.copy()
    df = df.sort_values("time_str").reset_index(drop=True)
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
        print(f"⚠️ Skipping {lake_name}: insufficient data after outlier removal ({len(analysis_data)} points)")
        return

    # Linear regression on all valid points (including near outliers)
    linear_trend = compute_trend(analysis_data["time_str"], analysis_data[PRIMARY_ATTRIBUTE])
    
    # ========================================
    # MODIFIED MANN-KENDALL TEST
    # ========================================
    times_numeric = np.array([d.toordinal() for d in analysis_data["time_str"]])
    values_numeric = np.array(analysis_data[PRIMARY_ATTRIBUTE])
    
    # Perform Modified Mann-Kendall test
    mk_tau, mk_pvalue, mk_slope, var_corrected = mann_kendall_modified(
        times_numeric, values_numeric
    )
    
    # Determine trend direction
    if mk_pvalue < CONFIDENCE_LEVEL:
        if mk_slope > 0:
            mk_direction = "Increasing (Significant)"
        elif mk_slope < 0:
            mk_direction = "Decreasing (Significant)"
        else:
            mk_direction = "No trend"
    else:
        mk_direction = "No significant trend"

    # Create figure
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

    # === AXIS SETTINGS ===
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
    ax.set_title(f"Temporal WSE Analysis: {lake_name}\n(Modified Mann-Kendall Test)", fontweight="bold", fontsize=15)

    # Stats box with both linear and modified Mann-Kendall trends
    linear_annual_change = linear_trend.slope * 365 if hasattr(linear_trend, 'slope') else 0
    mk_annual_change = mk_slope * 365
    
    linear_significance = "Significant" if linear_trend.pvalue < CONFIDENCE_LEVEL else "Not Significant"
    mk_significance = "Significant" if mk_pvalue < CONFIDENCE_LEVEL else "Not Significant"

    stats_text = (
        f"Total points: {len(df)}\n"
        f"Valid points: {len(analysis_data)}\n"
        f"Far outliers removed: {len(df[df['is_far_outlier']])}\n"
        f"Near outliers: {len(near)}\n"
        f"WSE range: {analysis_data[PRIMARY_ATTRIBUTE].min():.2f} – "
        f"{analysis_data[PRIMARY_ATTRIBUTE].max():.2f} {PRIMARY_UNIT}\n"
        f"\n--- Linear Regression ---\n"
        f"Annual trend: {linear_annual_change:.3f} {PRIMARY_UNIT}/yr\n"
        f"R² = {linear_trend.rvalue**2:.3f}\n"
        f"p-value = {linear_trend.pvalue:.4f} ({linear_significance})\n"
        f"\n--- Modified Mann-Kendall (Hamed & Rao, 1998) ---\n"
        f"Direction: {mk_direction}\n"
        f"Kendall's τ = {mk_tau:.3f}\n"
        f"Sen's slope: {mk_annual_change:.3f} {PRIMARY_UNIT}/yr\n"
        f"p-value = {mk_pvalue:.4f} ({mk_significance})\n"
        f"Variance corrected: {var_corrected:.3f}"
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

# ==================================================
# LOOP OVER ALL LAKE CSVs
# ==================================================

print("📊 Using MODIFIED Mann-Kendall Test (Hamed & Rao, 1998)")
print("="*60)

csv_files = sorted(Path(CSV_DIR).glob("*.csv"))

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    df["time_str"] = pd.to_datetime(df["time_str"])

    lake_id = df["lake_id"].iloc[0]
    lake_name = df["lake_name"].iloc[0]
    clean_name = lake_name.replace(" ", "_")

    out_file = Path(PLOT_DIR) / f"lake_{lake_id}_{clean_name}_WSE.png"

    print(f"📊 Plotting {lake_name} ({len(df)} points)")
    plot_lake(df, lake_name, out_file)

print("\n✅ All lake plots generated with Modified Mann-Kendall Test!")