"""Standalone LiDAR and energy analysis script for the JGR package.

This script converts the `restructured_lidar_energy_analysis.ipynb` workflow
into a reusable command-line entrypoint. All inputs default to files located in
the local `database/` directory.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mplconfig").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from scipy.stats import wasserstein_distance, weibull_min, wilcoxon

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    CARTOPY_AVAILABLE = True
except Exception:
    CARTOPY_AVAILABLE = False

try:
    import geopandas as gpd
    from shapely.geometry import Polygon

    GEOPANDAS_AVAILABLE = True
except Exception:
    GEOPANDAS_AVAILABLE = False

plt.rcParams.update({"font.size": 13})
sns.set_style("whitegrid")

POINT_LAT = -4.816689310172148
POINT_LON = -37.04502065180374
POINT_NAME = "Porto-Ilha"

LIDAR_TIME_COLUMN = "Time and Date"
LIDAR_WIND_COLUMN = "Horizontal Wind Speed (m/s) at 100m"

GRID_LAT_MIN, GRID_LAT_MAX, GRID_LAT_STEP = -4.75, -4.25, 0.25
GRID_LON_MIN, GRID_LON_MAX, GRID_LON_STEP = -36.40, -36.00, 0.25

BOX_LAT_MIN, BOX_LAT_MAX = -4.75, -4.25
BOX_LON_MIN, BOX_LON_MAX = -36.40, -36.00
BOX_RESOLUTION_KM = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the LiDAR validation and energy analysis from database files."
    )
    parser.add_argument(
        "--database-dir",
        default="database",
        help="Directory containing the notebook input files.",
    )
    parser.add_argument(
        "--wrfswan-file",
        default="wrfswan_100m.nc",
        help="Coupled WRF-SWAN 100 m wind dataset filename inside --database-dir.",
    )
    parser.add_argument(
        "--wrfstand-file",
        default="wrfstand_100m.nc",
        help="Stand-alone WRF 100 m wind dataset filename inside --database-dir.",
    )
    parser.add_argument(
        "--wrfswan-energy-file",
        default="energy_10MW_wrfswan.nc",
        help="Coupled energy dataset filename inside --database-dir.",
    )
    parser.add_argument(
        "--wrfstand-energy-file",
        default="energy_10MW_wrfstand.nc",
        help="Stand-alone energy dataset filename inside --database-dir.",
    )
    parser.add_argument(
        "--lidar-file",
        default="Porto_Ilha_2.csv",
        help="LiDAR CSV filename inside --database-dir.",
    )
    parser.add_argument(
        "--figures-dir",
        default="figures",
        help="Directory where figures will be written when --save-figures is enabled.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where summary tables will be written.",
    )
    parser.add_argument(
        "--time-shift-hours",
        type=int,
        default=-3,
        help="Hour offset applied to the wind-model timestamps before comparison.",
    )
    parser.add_argument(
        "--save-figures",
        action="store_true",
        help="Save generated figures to --figures-dir.",
    )
    parser.add_argument(
        "--save-tables",
        action="store_true",
        help="Save summary tables to --output-dir.",
    )
    parser.add_argument(
        "--export-bounding-box-shapefile",
        action="store_true",
        help="Export the offshore bounding box shapefile to --output-dir if GeoPandas is available.",
    )
    return parser.parse_args()


def assert_required_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "The following required files were not found:\n- " + "\n- ".join(missing)
        )


def safe_drop_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    return df.drop(columns=list(columns), errors="ignore")


def build_common_grid() -> xr.Dataset:
    lat = np.arange(GRID_LAT_MIN, GRID_LAT_MAX, GRID_LAT_STEP)
    lon = np.arange(GRID_LON_MIN, GRID_LON_MAX, GRID_LON_STEP)
    return xr.Dataset({"lat": (["lat"], lat), "lon": (["lon"], lon)})


def load_model_datasets(
    wrfswan_file: Path,
    wrfstand_file: Path,
    wrfswan_energy_file: Path,
    wrfstand_energy_file: Path,
) -> dict[str, xr.Dataset]:
    assert_required_files(
        [wrfswan_file, wrfstand_file, wrfswan_energy_file, wrfstand_energy_file]
    )
    return {
        "wrfswan": xr.open_dataset(wrfswan_file)[["wspd", "wdir"]],
        "wrfstand": xr.open_dataset(wrfstand_file)[["wspd", "wdir"]],
        "wrfswan_energy": xr.open_dataset(wrfswan_energy_file),
        "wrfstand_energy": xr.open_dataset(wrfstand_energy_file),
    }


def load_lidar_hourly(lidar_file: Path) -> pd.DataFrame:
    assert_required_files([lidar_file])

    lidar_df = pd.read_csv(lidar_file, sep=";")
    if LIDAR_TIME_COLUMN not in lidar_df.columns:
        raise KeyError(f"Column '{LIDAR_TIME_COLUMN}' not found in LiDAR file.")
    if LIDAR_WIND_COLUMN not in lidar_df.columns:
        raise KeyError(f"Column '{LIDAR_WIND_COLUMN}' not found in LiDAR file.")

    lidar_df["time"] = pd.to_datetime(lidar_df[LIDAR_TIME_COLUMN])
    lidar_df = lidar_df.set_index("time").sort_index()
    return lidar_df[[LIDAR_WIND_COLUMN]].resample("1h").mean().dropna()


def dataset_point_series(
    dataset: xr.Dataset,
    variable: str,
    time_shift_hours: int,
    lat: float = POINT_LAT,
    lon: float = POINT_LON,
) -> pd.DataFrame:
    series_df = (
        dataset[variable]
        .sel(y=lat, x=lon, method="nearest")
        .to_dataframe()
        .pipe(safe_drop_columns, ["lev", "y", "x", "spatial_ref"])
    )
    series_df.index = pd.to_datetime(series_df.index) + pd.Timedelta(hours=time_shift_hours)
    return series_df.sort_index()


def merge_validation_timeseries(
    wrfswan_df: pd.DataFrame,
    wrfstand_df: pd.DataFrame,
    lidar_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = wrfswan_df.merge(
        wrfstand_df,
        left_index=True,
        right_index=True,
        suffixes=("_wrfswan", "_wrfstand"),
    )
    merged = merged.merge(
        lidar_df[[LIDAR_WIND_COLUMN]],
        left_index=True,
        right_index=True,
    )
    merged.index.name = "time"
    return merged.dropna()


def compute_validation_metrics(observed: np.ndarray, modeled: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    modeled = np.asarray(modeled, dtype=float)

    metrics = {
        "EMD": float(wasserstein_distance(observed, modeled)),
        "BIAS": float(np.mean(modeled - observed)),
        "BIAS [%]": float(100.0 * np.mean(observed - modeled) / np.mean(observed)),
        "RMSE": float(np.sqrt(np.mean((observed - modeled) ** 2))),
        "CORR": float(np.corrcoef(observed, modeled)[0, 1]),
    }

    k_obs, _, c_obs = weibull_min.fit(observed, floc=0)
    k_mod, _, c_mod = weibull_min.fit(modeled, floc=0)
    metrics["ΔC"] = float(c_obs - c_mod)
    metrics["ΔK"] = float(k_obs - k_mod)
    return metrics


def build_metrics_table(validation_df: pd.DataFrame) -> pd.DataFrame:
    obs = validation_df[LIDAR_WIND_COLUMN].values
    wrfswan_metrics = compute_validation_metrics(obs, validation_df["wspd_wrfswan"].values)
    wrfstand_metrics = compute_validation_metrics(obs, validation_df["wspd_wrfstand"].values)
    return pd.DataFrame({"WRF-SWAN": wrfswan_metrics, "WRF": wrfstand_metrics}).round(2)


def compute_energy_difference_fields(
    wrfswan_energy_ds: xr.Dataset,
    wrfstand_energy_ds: xr.Dataset,
    power_var: str = "power_15m",
) -> tuple[xr.DataArray, xr.DataArray]:
    wrfswan_integrated = wrfswan_energy_ds[power_var].sum(dim="time") / 1000.0
    wrfstand_integrated = wrfstand_energy_ds[power_var].sum(dim="time") / 1000.0
    energy_difference = wrfswan_integrated - wrfstand_integrated
    relative_difference = xr.where(wrfswan_integrated != 0, energy_difference / wrfswan_integrated, np.nan)
    return energy_difference, relative_difference


def make_offshore_grid_dataframe() -> pd.DataFrame:
    lat_step = BOX_RESOLUTION_KM / 111.0
    lon_step = BOX_RESOLUTION_KM / (
        111.0 * np.cos(np.radians((BOX_LAT_MIN + BOX_LAT_MAX) / 2.0))
    )

    latitudes = np.arange(BOX_LAT_MIN, BOX_LAT_MAX + lat_step, lat_step)
    longitudes = np.arange(BOX_LON_MIN, BOX_LON_MAX + lon_step, lon_step)
    grid_points = [(lat, lon) for lat in latitudes for lon in longitudes]
    return pd.DataFrame(grid_points, columns=["Latitude", "Longitude"])


def make_bounding_box_from_grid(
    grid_df: pd.DataFrame,
    sample_stop: int = 155,
    sample_step: int = 4,
    lat_offset: float = -0.2,
) -> tuple[list[float], list[float]]:
    longitudes = grid_df["Longitude"][:sample_stop:sample_step]
    latitudes = grid_df["Latitude"][:sample_stop:sample_step] + lat_offset

    min_lon, max_lon = longitudes.min(), longitudes.max()
    min_lat, max_lat = latitudes.min(), latitudes.max()
    box_lon = [min_lon, max_lon, max_lon, min_lon, min_lon]
    box_lat = [min_lat, min_lat, max_lat, max_lat, min_lat]
    return box_lon, box_lat


def compute_rre_with_mae(
    model_data: np.ndarray,
    observed_data: np.ndarray,
    bin_width: int = 1,
    min_bin: int = 4,
    max_bin: int = 12,
) -> pd.DataFrame:
    bins = np.arange(min_bin, max_bin + bin_width, bin_width)
    labels = [f"{bins[i]}-{bins[i + 1]} m/s" for i in range(len(bins) - 1)]

    df = pd.DataFrame({"Observed": observed_data, "Model": model_data})
    df["Bin"] = pd.cut(df["Observed"], bins=bins, labels=labels, include_lowest=True)

    abs_error = (df["Model"] - df["Observed"]).abs()
    mae_by_bin = abs_error.groupby(df["Bin"], observed=False).mean()
    mean_observed_by_bin = df.groupby("Bin", observed=False)["Observed"].mean()
    rre_by_bin = mae_by_bin / mean_observed_by_bin

    return pd.DataFrame(
        {"MAE": mae_by_bin, "Mean Observed": mean_observed_by_bin, "RRE": rre_by_bin}
    )


def save_figure(fig: plt.Figure, figures_dir: Path, filename: str, save_figures: bool, dpi: int = 300) -> None:
    if save_figures:
        fig.savefig(figures_dir / filename, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def build_energy_summary(
    wrfswan_energy_ds: xr.Dataset,
    wrfstand_energy_ds: xr.Dataset,
    energy_difference: xr.DataArray,
    power_var: str = "power_15m",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "WRF-SWAN total energy [MWh]": [
                float((wrfswan_energy_ds[power_var].sum(dim="time") / 1000.0).sum().values)
            ],
            "WRF total energy [MWh]": [
                float((wrfstand_energy_ds[power_var].sum(dim="time") / 1000.0).sum().values)
            ],
            "Difference [MWh]": [float(energy_difference.sum().values)],
        }
    )


def create_validation_plots(
    validation_df: pd.DataFrame,
    metrics_table: pd.DataFrame,
    figures_dir: Path,
    save_figures: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    validation_df[["wspd_wrfswan", "wspd_wrfstand", LIDAR_WIND_COLUMN]].iloc[168:].plot(ax=ax)
    ax.set_title(f"Wind speed time series at {POINT_NAME}")
    ax.set_ylabel("Wind speed [m s$^{{-1}}$]")
    ax.set_xlabel("Time")
    ax.legend(["WRF-SWAN", "WRF", "LiDAR 100 m"])
    save_figure(fig, figures_dir, "timeseries_validation_porto_ilha.png", save_figures)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(validation_df["wspd_wrfswan"].values, label="WRF-SWAN", fill=True, ax=ax)
    sns.kdeplot(validation_df["wspd_wrfstand"].values, label="WRF", fill=True, ax=ax)
    sns.kdeplot(validation_df[LIDAR_WIND_COLUMN].values, label="LiDAR", ax=ax)
    ax.set_title(f"Wind-speed distribution at {POINT_NAME}")
    ax.set_xlabel("Wind speed [m s$^{{-1}}$]")
    ax.legend()
    save_figure(fig, figures_dir, "kde_wind_speed_porto_ilha.png", save_figures)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(
        validation_df["wspd_wrfswan"].values - validation_df[LIDAR_WIND_COLUMN].values,
        label="WRF-SWAN",
        stat="density",
        kde=True,
        bins=30,
        alpha=0.4,
        ax=ax,
    )
    sns.histplot(
        validation_df["wspd_wrfstand"].values - validation_df[LIDAR_WIND_COLUMN].values,
        label="WRF",
        stat="density",
        kde=True,
        bins=30,
        alpha=0.4,
        ax=ax,
    )
    ax.axvline(0, color="k", linestyle="--", linewidth=1.2)
    ax.set_title(POINT_NAME)
    ax.set_xlabel("U$_{mod}$ - U$_{obs}$ [m s$^{-1}$]")
    ax.legend()
    save_figure(fig, figures_dir, "error_distribution_porto_ilha.png", save_figures)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        metrics_table,
        cmap="RdBu_r",
        annot=True,
        fmt=".2f",
        ax=ax,
        cbar_kws={"label": "Metric value"},
    )
    ax.set_title("Wind-speed validation metrics")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    save_figure(fig, figures_dir, "metrics_heatmap_porto_ilha.png", save_figures)


def create_rre_plot(
    validation_df: pd.DataFrame,
    figures_dir: Path,
    save_figures: bool,
) -> pd.DataFrame:
    rre_by_bin_wrfswan = compute_rre_with_mae(
        validation_df["wspd_wrfswan"].values,
        validation_df[LIDAR_WIND_COLUMN].values,
    )
    rre_by_bin_wrfstand = compute_rre_with_mae(
        validation_df["wspd_wrfstand"].values,
        validation_df[LIDAR_WIND_COLUMN].values,
    )
    rre_between_models = compute_rre_with_mae(
        validation_df["wspd_wrfswan"].values,
        validation_df["wspd_wrfstand"].values,
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(rre_by_bin_wrfswan.index, rre_by_bin_wrfswan["RRE"], "-o", label="RRE WRF-SWAN")
    ax.plot(rre_by_bin_wrfstand.index, rre_by_bin_wrfstand["RRE"], "-*", label="RRE WRF")
    ax.plot(rre_between_models.index, rre_between_models["RRE"], ":s", label="RRE WRF-SWAN vs WRF")
    ax.set_xticks(range(len(rre_by_bin_wrfswan.index)))
    ax.set_xticklabels(rre_by_bin_wrfswan.index, rotation=45, ha="right")
    ax.set_ylabel("Relative reduced error")
    ax.set_xlabel("Observed wind-speed bins")
    ax.set_title("RRE by wind-speed bin")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, figures_dir, "rre_by_bin_porto_ilha.png", save_figures)

    combined = pd.concat(
        [
            rre_by_bin_wrfswan.add_prefix("wrfswan_"),
            rre_by_bin_wrfstand.add_prefix("wrfstand_"),
            rre_between_models.add_prefix("wrfswan_vs_wrf_"),
        ],
        axis=1,
    )
    return combined


def create_energy_map(
    wrfswan_energy_ds: xr.Dataset,
    wrfstand_energy_ds: xr.Dataset,
    relative_energy_difference: xr.DataArray,
    figures_dir: Path,
    save_figures: bool,
) -> bool:
    if not CARTOPY_AVAILABLE:
        return False

    grid_df = make_offshore_grid_dataframe()
    box_lon, box_lat = make_bounding_box_from_grid(grid_df)

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[5, 0.3])

    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    ax3 = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())
    cbar_ax1 = fig.add_subplot(gs[1, 0:2])
    cbar_ax2 = fig.add_subplot(gs[1, 2])

    lon = wrfswan_energy_ds.x
    lat = wrfswan_energy_ds.y
    land_feature = cfeature.LAND.with_scale("50m")

    def add_gridlines(ax: plt.Axes) -> None:
        gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5, alpha=0.7)
        gl.right_labels = False
        gl.top_labels = False
        gl.xlabel_style = {"size": 12}
        gl.ylabel_style = {"size": 12}

    plot1 = ax1.contourf(
        lon,
        lat,
        wrfswan_energy_ds["power_15m"].sum(dim="time") / 1000.0,
        transform=ccrs.PlateCarree(),
        cmap="Spectral_r",
        vmin=0,
        vmax=10500,
    )
    plot2 = ax2.contourf(
        lon,
        lat,
        wrfstand_energy_ds["power_15m"].sum(dim="time") / 1000.0,
        transform=ccrs.PlateCarree(),
        cmap="Spectral_r",
        vmin=0,
        vmax=10500,
    )

    valid_min = float(np.nanmin(relative_energy_difference.values))
    valid_max = float(np.nanmax(relative_energy_difference.values))
    abs_max = max(abs(valid_min), abs(valid_max))

    plot3 = ax3.contourf(
        lon,
        lat,
        100.0 * relative_energy_difference.values,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r",
        vmin=-abs_max * 100.0,
        vmax=abs_max * 100.0,
    )

    for ax, title in zip([ax1, ax2, ax3], ["WRF-SWAN", "WRF", "(WRF-SWAN) - WRF"]):
        ax.plot(box_lon, box_lat, color="k", linestyle="--", linewidth=2.0)
        ax.scatter(POINT_LON, POINT_LAT, color="k")
        ax.text(
            POINT_LON + 0.3,
            POINT_LAT + 0.05,
            POINT_NAME,
            fontsize=12,
            verticalalignment="bottom",
            horizontalalignment="right",
            color="black",
            weight="bold",
        )
        center_lon = (min(box_lon) + max(box_lon)) / 2.0
        center_lat = (min(box_lat) + max(box_lat)) / 2.0
        ax.text(center_lon, center_lat + 0.05, "OWF", fontsize=12, ha="center", weight="bold")
        ax.set_title(title)
        ax.add_feature(land_feature, facecolor="gray", zorder=10)
        ax.add_feature(cfeature.BORDERS, linestyle="-", zorder=11)
        ax.add_feature(cfeature.STATES, linestyle="-", alpha=0.5, zorder=12)
        add_gridlines(ax)

    cb1 = plt.colorbar(plot2, cax=cbar_ax1, orientation="horizontal")
    cb1.set_label("Energy [MWh]")
    cb2 = plt.colorbar(plot3, cax=cbar_ax2, orientation="horizontal")
    cb2.set_label("Δ Energy [%]")

    fig.tight_layout()
    save_figure(fig, figures_dir, "energy_maps_porto_ilha.png", save_figures)
    _ = plot1
    return True


def export_bounding_box_shapefile(output_dir: Path) -> bool:
    if not GEOPANDAS_AVAILABLE:
        return False

    grid_df = make_offshore_grid_dataframe()
    box_lon, box_lat = make_bounding_box_from_grid(grid_df)
    polygon = Polygon(list(zip(box_lon, box_lat)))
    gdf = gpd.GeoDataFrame({"geometry": [polygon]}, crs="EPSG:4326")
    gdf.to_file(output_dir / "bounding_box.shp")
    return True


def main() -> None:
    args = parse_args()

    database_dir = Path(args.database_dir)
    figures_dir = Path(args.figures_dir)
    output_dir = Path(args.output_dir)

    figures_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_model_datasets(
        database_dir / args.wrfswan_file,
        database_dir / args.wrfstand_file,
        database_dir / args.wrfswan_energy_file,
        database_dir / args.wrfstand_energy_file,
    )
    common_grid = build_common_grid()

    dset_wrfswan = datasets["wrfswan"]
    dset_wrfstand = datasets["wrfstand"]
    dset_wrfswan_energy = datasets["wrfswan_energy"]
    dset_wrfstand_energy = datasets["wrfstand_energy"]

    _ = dset_wrfswan.interp(y=common_grid.lat, x=common_grid.lon)
    _ = dset_wrfstand.interp(y=common_grid.lat, x=common_grid.lon)
    _ = dset_wrfswan_energy.interp(y=common_grid.lat, x=common_grid.lon)
    _ = dset_wrfstand_energy.interp(y=common_grid.lat, x=common_grid.lon)

    df_lidar_hourly = load_lidar_hourly(database_dir / args.lidar_file)
    df_wrfswan_wind = dataset_point_series(
        dset_wrfswan,
        "wspd",
        time_shift_hours=args.time_shift_hours,
    )
    df_wrfstand_wind = dataset_point_series(
        dset_wrfstand,
        "wspd",
        time_shift_hours=args.time_shift_hours,
    )
    df_wrfswan_power = dataset_point_series(
        dset_wrfswan_energy,
        "power_15m",
        time_shift_hours=0,
    )
    df_wrfstand_power = dataset_point_series(
        dset_wrfstand_energy,
        "power_15m",
        time_shift_hours=0,
    )

    validation_df = merge_validation_timeseries(
        df_wrfswan_wind,
        df_wrfstand_wind,
        df_lidar_hourly,
    )
    power_df = df_wrfswan_power.merge(
        df_wrfstand_power,
        left_index=True,
        right_index=True,
        suffixes=("_wrfswan", "_wrfstand"),
    )

    metrics_table = build_metrics_table(validation_df)
    energy_difference, relative_energy_difference = compute_energy_difference_fields(
        dset_wrfswan_energy,
        dset_wrfstand_energy,
        power_var="power_15m",
    )
    energy_summary = build_energy_summary(
        dset_wrfswan_energy,
        dset_wrfstand_energy,
        energy_difference,
        power_var="power_15m",
    )

    before_treatment = dset_wrfstand_energy["power_15m"].sum(dim="time").values.flatten()
    after_treatment = dset_wrfswan_energy["power_15m"].sum(dim="time").values.flatten()
    mask = np.isfinite(before_treatment) & np.isfinite(after_treatment)
    statistic, p_value = wilcoxon(before_treatment[mask], after_treatment[mask])

    create_validation_plots(validation_df, metrics_table, figures_dir, args.save_figures)
    rre_summary = create_rre_plot(validation_df, figures_dir, args.save_figures)
    energy_map_created = create_energy_map(
        dset_wrfswan_energy,
        dset_wrfstand_energy,
        relative_energy_difference,
        figures_dir,
        args.save_figures,
    )

    shapefile_created = False
    if args.export_bounding_box_shapefile:
        shapefile_created = export_bounding_box_shapefile(output_dir)

    if args.save_tables:
        metrics_table.to_csv(output_dir / "validation_metrics_porto_ilha.csv")
        validation_df.to_csv(output_dir / "validation_timeseries_porto_ilha.csv")
        power_df.to_csv(output_dir / "power_timeseries_porto_ilha.csv")
        energy_summary.to_csv(output_dir / "energy_summary_porto_ilha.csv", index=False)
        rre_summary.to_csv(output_dir / "rre_summary_porto_ilha.csv")
        pd.DataFrame(
            {
                "wilcoxon_statistic": [float(statistic)],
                "p_value": [float(p_value)],
            }
        ).to_csv(output_dir / "wilcoxon_summary_porto_ilha.csv", index=False)

    print(f"Merged validation samples: {len(validation_df)}")
    print("\nValidation metrics:")
    print(metrics_table.to_string())
    print("\nEnergy summary [MWh]:")
    print(energy_summary.to_string(index=False))
    print(f"\nWilcoxon signed-rank statistic: {statistic:.3f}")
    print(f"Wilcoxon p-value: {p_value:.6g}")
    print(
        "Paired field result: "
        + (
            "reject the null hypothesis."
            if p_value < 0.05
            else "fail to reject the null hypothesis."
        )
    )
    print(f"Figures saved: {args.save_figures}")
    print(f"Energy map created: {energy_map_created}")
    print(f"Bounding box shapefile created: {shapefile_created}")


if __name__ == "__main__":
    main()
