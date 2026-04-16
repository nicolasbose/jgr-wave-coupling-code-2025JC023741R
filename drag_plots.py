"""Standalone drag-coefficient plots for the JGR submission package.

This script separates the drag analysis from the original
`Drag_and_wave_stress.ipynb` notebook. It reproduces the main drag figure
using only the database files and keeps optional wave-age and wind-wave
alignment figures available when the SWAN post-process file contains the
required variables.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

G = 9.81


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create standalone drag-coefficient plots from the database files."
    )
    parser.add_argument(
        "--wrfswan-sfc",
        default="database/wrfout_wrfswan_JFM2024.nc",
        help="Coupled surface WRF output used for UST, Wspd10, and Wdir.",
    )
    parser.add_argument(
        "--wrfstand-sfc",
        default="database/wrfout_wrfstand_JFM2024.nc",
        help="Stand-alone surface WRF output used for UST, Wspd10, and Wdir.",
    )
    parser.add_argument(
        "--wrfswan-100",
        default="database/wrfout_wrfswan_100_JFM2024.nc",
        help="Coupled 100 m WRF output used for Wspd100.",
    )
    parser.add_argument(
        "--wrfstand-100",
        default="database/wrfout_wrfstand_100_JFM2024.nc",
        help="Stand-alone 100 m WRF output used for Wspd100.",
    )
    parser.add_argument(
        "--spec",
        default="database/spec_1d_tau_with_tau_chen.nc",
        help="SWAN post-process file used for optional wave-age and angle figures.",
    )
    parser.add_argument(
        "--output-dir",
        default="figures",
        help="Directory where the figures will be saved.",
    )
    parser.add_argument(
        "--wave-direction-var",
        default="wave_dir",
        help="Wave-direction variable in the SWAN post-process file.",
    )
    parser.add_argument(
        "--lat-start",
        type=float,
        default=-4.75,
        help="Start latitude for the notebook common grid.",
    )
    parser.add_argument(
        "--lat-stop",
        type=float,
        default=-4.25,
        help="Stop latitude for the notebook common grid.",
    )
    parser.add_argument(
        "--lat-step",
        type=float,
        default=0.25,
        help="Latitude spacing for the notebook common grid.",
    )
    parser.add_argument(
        "--lon-start",
        type=float,
        default=-36.4,
        help="Start longitude for the notebook common grid.",
    )
    parser.add_argument(
        "--lon-stop",
        type=float,
        default=-36.0,
        help="Stop longitude for the notebook common grid.",
    )
    parser.add_argument(
        "--lon-step",
        type=float,
        default=0.25,
        help="Longitude spacing for the notebook common grid.",
    )
    return parser.parse_args()


def estimate_wave_age(u: np.ndarray, tm01: np.ndarray) -> np.ndarray:
    cp = (G * tm01) / (2.0 * np.pi)
    return u / cp


def separete_variable_wind_bins(
    wspd: np.ndarray,
    main_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wind_bins = np.arange(1, 13, 0.5)
    bin_centers = (wind_bins[:-1] + wind_bins[1:]) / 2.0
    means: list[float] = []
    stds: list[float] = []

    for left, right in zip(wind_bins[:-1], wind_bins[1:]):
        mask = (wspd >= left) & (wspd < right)
        values = main_data[mask]
        if values.size:
            means.append(float(np.nanmean(values)))
            stds.append(float(np.nanstd(values)))
        else:
            means.append(np.nan)
            stds.append(np.nan)

    return bin_centers, np.asarray(means), np.asarray(stds)


def compute_wind_wave_alignment(
    wind_direction: np.ndarray,
    wave_direction: np.ndarray,
) -> np.ndarray:
    theta1 = np.deg2rad(wind_direction)
    theta2 = np.deg2rad(wave_direction)
    u1, v1 = np.cos(theta1), np.sin(theta1)
    u2, v2 = np.cos(theta2), np.sin(theta2)
    cos_similarity = (u1 * u2) + (v1 * v2)
    cos_similarity = np.clip(cos_similarity, -1.0, 1.0)
    return np.degrees(np.arccos(cos_similarity))


def bin_wind_wave_alignment(angle_wind_wave: np.ndarray) -> np.ndarray:
    """Map raw alignment angles using the notebook's bin edges and labels."""
    bins = np.array([0.0, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360.0])
    bin_labels = np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 150.0, 120.0, 90.0, 60.0, 30.0, 0.0])
    clipped = np.mod(angle_wind_wave, 360.0)
    bin_indices = np.digitize(clipped, bins, right=True)
    return np.array([bin_labels[index - 1] for index in bin_indices], dtype=float)


def reorder_by_alignment_bins(
    angle_raw: np.ndarray,
    binned_directions: np.ndarray,
    *arrays: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Reproduce the notebook's bin-by-bin concatenation order."""
    ranges = [
        (0.0, 30.0, False),
        (30.0, 60.0, False),
        (60.0, 90.0, False),
        (90.0, 120.0, False),
        (120.0, 150.0, False),
        (150.0, 180.0, True),
    ]
    grouped: list[list[np.ndarray]] = [[] for _ in range(len(arrays) + 1)]
    for lower, upper, include_upper in ranges:
        if include_upper:
            mask = (binned_directions >= lower) & (binned_directions <= upper)
        else:
            mask = (binned_directions >= lower) & (binned_directions < upper)
        grouped[0].append(angle_raw[mask])
        for idx, values in enumerate(arrays, start=1):
            grouped[idx].append(values[mask])
    return tuple(np.concatenate(parts) for parts in grouped)


def resolve_wave_direction_var(
    spec: xr.Dataset,
    requested_name: str,
) -> str | None:
    candidates = [
        requested_name,
        "wave_dir",
        "mwd",
        "mean_wave_direction",
        "wind_direction",
    ]
    for name in candidates:
        if name in spec:
            return name
    return None


def prepare_common_datasets(
    wrfswan_sfc_path: str | Path,
    wrfstand_sfc_path: str | Path,
    wrfswan_100_path: str | Path,
    wrfstand_100_path: str | Path,
    spec_path: str | Path,
    lat_start: float,
    lat_stop: float,
    lat_step: float,
    lon_start: float,
    lon_stop: float,
    lon_step: float,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset]:
    wrfswan_sfc = xr.open_dataset(wrfswan_sfc_path)
    wrfstand_sfc = xr.open_dataset(wrfstand_sfc_path)
    wrfswan_100 = xr.open_dataset(wrfswan_100_path)
    wrfstand_100 = xr.open_dataset(wrfstand_100_path)
    spec = xr.open_dataset(spec_path)

    lat = np.arange(lat_start, lat_stop, lat_step)
    lon = np.arange(lon_start, lon_stop, lon_step)
    common_grid = xr.Dataset({"lat": (["lat"], lat), "lon": (["lon"], lon)})

    spec_interp = spec.sel(lat=common_grid.lat, lon=common_grid.lon)
    ds_wrfswan = wrfswan_sfc.interp(y=common_grid.lat, x=common_grid.lon)
    ds_wrfstand = wrfstand_sfc.interp(y=common_grid.lat, x=common_grid.lon)
    ds_wrfswan_100 = wrfswan_100.interp(y=common_grid.lat, x=common_grid.lon)
    ds_wrfstand_100 = wrfstand_100.interp(y=common_grid.lat, x=common_grid.lon)

    common_times = np.intersect1d(ds_wrfswan.time.values, ds_wrfstand.time.values)
    common_times = np.intersect1d(common_times, ds_wrfswan_100.time.values)
    common_times = np.intersect1d(common_times, ds_wrfstand_100.time.values)
    common_times = np.intersect1d(common_times, spec_interp.time.values)

    return (
        ds_wrfswan.sel(time=common_times),
        ds_wrfstand.sel(time=common_times),
        ds_wrfswan_100.sel(time=common_times),
        ds_wrfstand_100.sel(time=common_times),
        spec_interp.sel(time=common_times),
    )


def build_drag_arrays(
    wrfswan_sfc_path: str | Path,
    wrfstand_sfc_path: str | Path,
    wrfswan_100_path: str | Path,
    wrfstand_100_path: str | Path,
    spec_path: str | Path,
    lat_start: float,
    lat_stop: float,
    lat_step: float,
    lon_start: float,
    lon_stop: float,
    lon_step: float,
    wave_direction_var: str,
) -> dict[str, np.ndarray | None]:
    (
        ds_wrfswan,
        ds_wrfstand,
        ds_wrfswan_100,
        ds_wrfstand_100,
        spec_interp,
    ) = prepare_common_datasets(
        wrfswan_sfc_path=wrfswan_sfc_path,
        wrfstand_sfc_path=wrfstand_sfc_path,
        wrfswan_100_path=wrfswan_100_path,
        wrfstand_100_path=wrfstand_100_path,
        spec_path=spec_path,
        lat_start=lat_start,
        lat_stop=lat_stop,
        lat_step=lat_step,
        lon_start=lon_start,
        lon_stop=lon_stop,
        lon_step=lon_step,
    )

    ust_wrfswan = ds_wrfswan.UST.values.flatten()
    ust_wrfstand = ds_wrfstand.UST.values.flatten()
    wspd10_wrfswan = ds_wrfswan.wspd.values.flatten()
    wspd10_wrfstand = ds_wrfstand.wspd.values.flatten()
    wspd100_wrfswan = ds_wrfswan_100.wspd.values.flatten()
    wspd100_wrfstand = ds_wrfstand_100.wspd.values.flatten()

    cd_wrfswan = (ust_wrfswan**2) / (wspd10_wrfswan**2)
    cd_wrfstand = (ust_wrfstand**2) / (wspd10_wrfstand**2)
    d_u10 = wspd10_wrfswan - wspd10_wrfstand
    d_wspd100 = wspd100_wrfswan - wspd100_wrfstand

    valid = (
        np.isfinite(wspd10_wrfswan)
        & np.isfinite(wspd10_wrfstand)
        & np.isfinite(wspd100_wrfswan)
        & np.isfinite(wspd100_wrfstand)
        & np.isfinite(cd_wrfswan)
        & np.isfinite(cd_wrfstand)
    )

    wave_age = None
    if "tm01" in spec_interp:
        wave_age_raw = estimate_wave_age(ust_wrfswan, spec_interp.tm01.values.flatten())
        valid = valid & np.isfinite(wave_age_raw)
        wave_age = wave_age_raw

    angle_wind_wave = None
    angle_wind_wave_binned = None
    resolved_wave_direction_var = resolve_wave_direction_var(spec_interp, wave_direction_var)
    if resolved_wave_direction_var is not None:
        wave_direction = spec_interp[resolved_wave_direction_var].values.flatten()
        wind_direction = ds_wrfswan.wdir.values.flatten()
        valid = valid & np.isfinite(wind_direction) & np.isfinite(wave_direction)
        angle_wind_wave_raw = compute_wind_wave_alignment(wind_direction, wave_direction)
        valid = valid & np.isfinite(angle_wind_wave_raw)
        angle_wind_wave = angle_wind_wave_raw
        angle_wind_wave_binned = bin_wind_wave_alignment(angle_wind_wave_raw)

    data = {
        "wspd10_wrfswan": wspd10_wrfswan[valid],
        "wspd10_wrfstand": wspd10_wrfstand[valid],
        "wspd100_wrfswan": wspd100_wrfswan[valid],
        "wspd100_wrfstand": wspd100_wrfstand[valid],
        "cd_wrfswan": cd_wrfswan[valid],
        "cd_wrfstand": cd_wrfstand[valid],
        "d_u10": d_u10[valid],
        "d_wspd100": d_wspd100[valid],
        "wave_age": wave_age[valid] if wave_age is not None else None,
        "angle_wind_wave": angle_wind_wave[valid] if angle_wind_wave is not None else None,
        "angle_wind_wave_binned": (
            angle_wind_wave_binned[valid] if angle_wind_wave_binned is not None else None
        ),
    }
    data["wave_direction_var_used"] = resolved_wave_direction_var

    (
        data["bin_centers_wrfswan"],
        data["mean_wrfswan"],
        data["std_dev_wrfswan"],
    ) = separete_variable_wind_bins(data["wspd10_wrfswan"], data["cd_wrfswan"] * 1000.0)
    (
        data["bin_centers_wrfstand"],
        data["mean_wrfstand"],
        data["std_dev_wrfstand"],
    ) = separete_variable_wind_bins(data["wspd10_wrfstand"], data["cd_wrfstand"] * 1000.0)

    return data


def make_main_drag_figure(
    data: dict[str, np.ndarray | None],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mask = (
        (data["wspd10_wrfswan"] >= 1.0)
        & (data["wspd10_wrfswan"] <= 13.0)
        & (data["wspd10_wrfstand"] >= 1.0)
        & (data["wspd10_wrfstand"] <= 13.0)
    )
    d_u10 = data["d_u10"][mask]
    lim = float(np.nanpercentile(np.abs(d_u10), 98))
    vmax = max(0.5, lim)
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.cm.coolwarm

    plt.rcParams.update({"font.size": 16})
    fig, ax = plt.subplots(1, 2, figsize=(25, 8), sharex=True, sharey=True)

    ax[0].scatter(
        data["wspd10_wrfswan"][mask],
        data["cd_wrfswan"][mask] * 1000.0,
        c=d_u10,
        cmap=cmap,
        norm=norm,
        s=18,
        linewidths=0,
    )
    ax[0].errorbar(
        data["bin_centers_wrfswan"],
        data["mean_wrfswan"],
        yerr=data["std_dev_wrfswan"],
        color="k",
        fmt="o--",
        capsize=5,
        label="Mean ± SD",
    )
    ax[0].set_title("Coupled: WRF-SWAN", pad=10)
    ax[0].set_xlabel(r"Wspd$_{10}$ [m s$^{-1}$]")
    ax[0].set_ylabel(r"1000 × C$_{D}$")
    ax[0].grid(True)
    ax[0].legend(loc="upper left")

    ax[1].scatter(
        data["wspd10_wrfstand"][mask],
        data["cd_wrfstand"][mask] * 1000.0,
        c=d_u10,
        cmap=cmap,
        norm=norm,
        s=18,
        linewidths=0,
    )
    ax[1].errorbar(
        data["bin_centers_wrfstand"],
        data["mean_wrfstand"],
        yerr=data["std_dev_wrfstand"],
        color="k",
        fmt="o--",
        capsize=5,
        label="Mean ± SD",
    )
    ax[1].set_title("Stand-alone: WRF", pad=10)
    ax[1].set_xlabel(r"Wspd$_{10}$ [m s$^{-1}$]")
    ax[1].set_ylabel(r"1000 × C$_{D}$")
    ax[1].grid(True)
    ax[1].legend(loc="upper left")

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        location="right",
        fraction=0.05,
        pad=0.02,
    )
    cbar.set_label(
        r"$\Delta$ U$_{10}$ = Wspd$_{10}$(WRF-SWAN) − Wspd$_{10}$(WRF) [m s$^{-1}$]"
    )

    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return output_path


def make_wave_age_drag_figure(
    data: dict[str, np.ndarray | None],
    output_path: str | Path,
) -> Path | None:
    if data["wave_age"] is None:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mask = (
        (data["wspd10_wrfswan"] >= 1.0)
        & (data["wspd10_wrfswan"] <= 13.0)
        & (data["wspd10_wrfstand"] >= 1.0)
        & (data["wspd10_wrfstand"] <= 13.0)
    )
    wave_age = data["wave_age"][mask]
    vmin = float(np.nanmin(wave_age))
    vmax = float(np.nanmax(wave_age))
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.025, vmax=vmax)
    if data["angle_wind_wave"] is not None and data["angle_wind_wave_binned"] is not None:
        (
            ordered_angle,
            ordered_wspd10,
            ordered_cd,
            ordered_wave_age,
        ) = reorder_by_alignment_bins(
            data["angle_wind_wave"][mask],
            data["angle_wind_wave_binned"][mask],
            data["wspd10_wrfswan"][mask],
            data["cd_wrfswan"][mask],
            wave_age,
        )
    else:
        ordered_angle = None
        ordered_wspd10 = data["wspd10_wrfswan"][mask]
        ordered_cd = data["cd_wrfswan"][mask]
        ordered_wave_age = wave_age

    plt.rcParams.update({"font.size": 16})
    if data["angle_wind_wave"] is not None:
        fig, ax = plt.subplots(1, 2, figsize=(25, 8), sharex=True, sharey=True)
        axes = np.asarray(ax)
    else:
        fig, ax = plt.subplots(figsize=(12, 7))
        axes = np.asarray([ax])

    axes[0].scatter(
        ordered_wspd10,
        ordered_cd * 1000.0,
        c=ordered_wave_age,
        cmap="RdYlBu_r",
        norm=norm,
        s=22,
        linewidths=0,
        alpha=0.9,
    )
    axes[0].errorbar(
        data["bin_centers_wrfswan"],
        data["mean_wrfswan"],
        yerr=data["std_dev_wrfswan"],
        color="k",
        fmt="o--",
        capsize=5,
        label="Mean ± Std Dev",
    )
    axes[0].set_title("Coupled: WRF-SWAN", pad=8)
    axes[0].set_xlabel(r"Wspd$_{10}$ [m s$^{-1}$]")
    axes[0].set_ylabel(r"1000 × C$_{D}$")
    axes[0].grid(True, alpha=0.6)
    axes[0].legend(loc="upper left")
    axes[0].set_xlim(0.5, 12.6)
    axes[0].set_ylim(0.68, 2.76)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="RdYlBu_r"),
        ax=axes[0],
        location="right",
        fraction=0.05,
        pad=0.02,
    )
    cbar.set_label(r"$u_{*}/C_{p}$")

    if data["angle_wind_wave"] is not None:
        norm_angle = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=90.0, vmax=180.0)
        axes[1].scatter(
            ordered_wspd10,
            ordered_cd * 1000.0,
            c=ordered_angle,
            cmap="coolwarm",
            norm=norm_angle,
            s=22,
            linewidths=0,
            alpha=0.9,
        )
        axes[1].errorbar(
            data["bin_centers_wrfswan"],
            data["mean_wrfswan"],
            yerr=data["std_dev_wrfswan"],
            color="k",
            fmt="o--",
            capsize=5,
            label="Mean ± Std Dev",
        )
        axes[1].set_title("Coupled: WRF-SWAN", pad=8)
        axes[1].set_xlabel(r"Wspd$_{10}$ [m s$^{-1}$]")
        axes[1].set_ylabel(r"1000 × C$_{D}$")
        axes[1].grid(True, alpha=0.6)
        axes[1].legend(loc="upper left")
        axes[1].set_xlim(0.5, 12.6)
        axes[1].set_ylim(0.68, 2.76)

        cbar2 = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm_angle, cmap="coolwarm"),
            ax=axes[1],
            location="right",
            fraction=0.05,
            pad=0.02,
        )
        cbar2.set_label("Wind-Wave alignment [°]")

    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return output_path


def make_wave_age_angle_drag_figure(
    data: dict[str, np.ndarray | None],
    output_path: str | Path,
) -> Path | None:
    if data["wave_age"] is None or data["angle_wind_wave"] is None:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mask = (
        (data["wspd10_wrfswan"] >= 1.0)
        & (data["wspd10_wrfswan"] <= 13.0)
    )
    if data["angle_wind_wave_binned"] is not None:
        (
            ordered_angle,
            ordered_wspd10,
            ordered_cd,
            ordered_wave_age,
        ) = reorder_by_alignment_bins(
            data["angle_wind_wave"][mask],
            data["angle_wind_wave_binned"][mask],
            data["wspd10_wrfswan"][mask],
            data["cd_wrfswan"][mask],
            data["wave_age"][mask],
        )
    else:
        ordered_angle = data["angle_wind_wave"][mask]
        ordered_wspd10 = data["wspd10_wrfswan"][mask]
        ordered_cd = data["cd_wrfswan"][mask]
        ordered_wave_age = data["wave_age"][mask]

    plt.rcParams.update({"font.size": 16})
    fig, ax = plt.subplots(1, 2, figsize=(25, 8), sharex=True, sharey=True)

    norm_wave_age = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=0.025, vmax=0.05)
    ax[0].scatter(
        ordered_wspd10,
        ordered_cd * 1000.0,
        c=ordered_wave_age,
        cmap="RdYlBu_r",
        norm=norm_wave_age,
        s=22,
        linewidths=0,
        alpha=0.9,
    )
    ax[0].errorbar(
        data["bin_centers_wrfswan"],
        data["mean_wrfswan"],
        yerr=data["std_dev_wrfswan"],
        color="k",
        fmt="o--",
        capsize=5,
        label="Mean ± SD",
    )
    ax[0].set_title("Coupled: WRF-SWAN", pad=10)
    ax[0].set_xlabel(r"Wspd$_{10}$ [m s$^{-1}$]")
    ax[0].set_ylabel(r"1000 × C$_{D}$")
    ax[0].grid(True, alpha=0.6)
    ax[0].legend(loc="upper left")
    ax[0].set_xlim(0.5, 12.6)
    ax[0].set_ylim(0.68, 2.76)

    norm_angle = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=90.0, vmax=180.0)
    ax[1].scatter(
        ordered_wspd10,
        ordered_cd * 1000.0,
        c=ordered_angle,
        cmap="coolwarm",
        norm=norm_angle,
        s=22,
        linewidths=0,
        alpha=0.9,
    )
    ax[1].errorbar(
        data["bin_centers_wrfswan"],
        data["mean_wrfswan"],
        yerr=data["std_dev_wrfswan"],
        color="k",
        fmt="o--",
        capsize=5,
        label="Mean ± SD",
    )
    ax[1].set_title("Coupled: WRF-SWAN", pad=10)
    ax[1].set_xlabel(r"Wspd$_{10}$ [m s$^{-1}$]")
    ax[1].set_ylabel(r"1000 × C$_{D}$")
    ax[1].grid(True, alpha=0.6)
    ax[1].legend(loc="upper left")
    ax[1].set_xlim(0.5, 12.6)
    ax[1].set_ylim(0.68, 2.76)

    cbar1 = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm_wave_age, cmap="RdYlBu_r"),
        ax=ax[0],
        location="right",
        fraction=0.05,
        pad=0.02,
    )
    cbar1.set_label(r"$u_{*}/C_{p}$")

    cbar2 = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm_angle, cmap="coolwarm"),
        ax=ax[1],
        location="right",
        fraction=0.05,
        pad=0.02,
    )
    cbar2.set_label("Wind-Wave alignment [°]")

    fig.savefig(output_path, dpi=500, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    data = build_drag_arrays(
        wrfswan_sfc_path=args.wrfswan_sfc,
        wrfstand_sfc_path=args.wrfstand_sfc,
        wrfswan_100_path=args.wrfswan_100,
        wrfstand_100_path=args.wrfstand_100,
        spec_path=args.spec,
        lat_start=args.lat_start,
        lat_stop=args.lat_stop,
        lat_step=args.lat_step,
        lon_start=args.lon_start,
        lon_stop=args.lon_stop,
        lon_step=args.lon_step,
        wave_direction_var=args.wave_direction_var,
    )

    output_dir = Path(args.output_dir)
    main_path = make_main_drag_figure(
        data=data,
        output_path=output_dir / "Figure4_CDN_vs_U10N_WRF-SWAN_vs_WRF.png",
    )
    wave_age_path = make_wave_age_drag_figure(
        data=data,
        output_path=output_dir / "Cd10N_wave_age_wrfswan.png",
    )
    angle_path = make_wave_age_angle_drag_figure(
        data=data,
        output_path=output_dir / "Cd10N_wave_age_angle_analysis_wave_age_wrfswan.png",
    )

    print(f"Saved main drag figure to: {main_path.resolve()}")
    if wave_age_path is not None:
        print(f"Saved wave-age drag figure to: {wave_age_path.resolve()}")
    else:
        print("Wave-age drag figure skipped: tm01 was not found in the spec file.")
    if angle_path is not None:
        print(f"Saved angle drag figure to: {angle_path.resolve()}")
    else:
        print(
            "Angle drag figure skipped: "
            f"wave-direction variable '{args.wave_direction_var}' was not found in the spec file."
        )
    if data["wave_direction_var_used"] is not None:
        print(f"Wave-direction variable used: {data['wave_direction_var_used']}")
    print(f"Samples used: {data['wspd10_wrfswan'].size}")
    print(
        "Coupled Wspd10 range: "
        f"{np.nanmin(data['wspd10_wrfswan']):.4f} to {np.nanmax(data['wspd10_wrfswan']):.4f}"
    )
    print(
        "Stand-alone Wspd10 range: "
        f"{np.nanmin(data['wspd10_wrfstand']):.4f} to {np.nanmax(data['wspd10_wrfstand']):.4f}"
    )
    print(
        "Coupled Cd range: "
        f"{np.nanmin(data['cd_wrfswan'] * 1000.0):.4f} to {np.nanmax(data['cd_wrfswan'] * 1000.0):.4f}"
    )
    print(
        "Stand-alone Cd range: "
        f"{np.nanmin(data['cd_wrfstand'] * 1000.0):.4f} to {np.nanmax(data['cd_wrfstand'] * 1000.0):.4f}"
    )


if __name__ == "__main__":
    main()
