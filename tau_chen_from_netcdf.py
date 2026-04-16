"""Compute tau-Chen from a spectrum NetCDF created for the coupled WRF-SWAN run."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


def _integrate(y: np.ndarray, x: np.ndarray) -> float:
    """Use the newest NumPy integration API when available, with legacy fallback."""

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


class SwellWave:
    """Chen swell/wind-sea partitioning based on a 1D wave spectrum."""

    def __init__(self, u10: float, spec1d: np.ndarray, freq_band: np.ndarray):
        self.u10 = float(u10)
        self.spec1d = np.asarray(spec1d, dtype=float)
        self.freq_band = np.asarray(freq_band, dtype=float)

    def _cutoff_frequency(self) -> float:
        g = 9.8
        return (0.83 * g) / (2 * np.pi * self.u10)

    def swell_part(self) -> tuple[np.ndarray, np.ndarray]:
        fc = self._cutoff_frequency()
        mask = self.freq_band < fc
        return self.freq_band[mask], self.spec1d[mask]

    def sea_part(self) -> tuple[np.ndarray, np.ndarray]:
        fc = self._cutoff_frequency()
        mask = self.freq_band > fc
        return self.freq_band[mask], self.spec1d[mask]

    def m0_swell_part(self) -> tuple[float, float]:
        freq_swell, spec_swell = self.swell_part()
        m0_swell = _integrate(spec_swell, freq_swell) if freq_swell.size else 0.0
        m0_total = _integrate(self.spec1d, self.freq_band)
        return float(m0_swell), float(m0_total)

    def swell_index(self) -> float:
        m0_swell, m0_total = self.m0_swell_part()
        if np.isclose(m0_total, 0.0):
            return np.nan
        return float(m0_swell / m0_total)

    def tau_chen(self) -> float:
        g = 9.8
        freq_swell, spec_swell = self.swell_part()
        freq_sea, spec_sea = self.sea_part()

        tau_swell = 0.0
        tau_sea = 0.0

        if freq_swell.size:
            c_swell = g / (2 * np.pi * freq_swell)
            cb_swell = -30.0
            term_swell = (cb_swell * (2 * np.pi * freq_swell) ** 2) / c_swell**2
            tau_swell = _integrate(term_swell * spec_swell, freq_swell)

        if freq_sea.size:
            c_sea = g / (2 * np.pi * freq_sea)
            cb_sea = 16.0
            term_sea = (cb_sea * (2 * np.pi * freq_sea) ** 2) / c_sea**2
            tau_sea = _integrate(term_sea * spec_sea, freq_sea)

        return tau_swell + tau_sea


def compute_tau_chen_dataset(
    dataset: xr.Dataset,
    spectrum_var: str = "spec_1d",
    wind_speed_var: str = "wind_speed",
    frequency_coord: str = "frequency",
) -> xr.Dataset:
    """Compute tau-Chen and swell index for each time/lat/lon point in the dataset."""

    if spectrum_var not in dataset:
        raise KeyError(f"Dataset does not contain spectrum variable '{spectrum_var}'.")
    if wind_speed_var not in dataset:
        raise KeyError(f"Dataset does not contain wind-speed variable '{wind_speed_var}'.")
    if frequency_coord not in dataset.coords:
        raise KeyError(f"Dataset does not contain frequency coordinate '{frequency_coord}'.")

    spectrum = dataset[spectrum_var]
    wind_speed = dataset[wind_speed_var]
    frequency = dataset[frequency_coord].values

    expected_spectrum_dims = ("time", "lat", "lon", frequency_coord)
    if spectrum.dims != expected_spectrum_dims:
        raise ValueError(
            f"Expected spectrum dims {expected_spectrum_dims}, got {spectrum.dims}."
        )

    expected_wind_dims = ("time", "lat", "lon")
    if wind_speed.dims != expected_wind_dims:
        raise ValueError(
            f"Expected wind-speed dims {expected_wind_dims}, got {wind_speed.dims}."
        )

    tau = np.zeros(wind_speed.shape, dtype=float)
    swell_index = np.zeros(wind_speed.shape, dtype=float)

    time_size, lat_size, lon_size = wind_speed.shape
    for time_index in range(time_size):
        for lat_index in range(lat_size):
            for lon_index in range(lon_size):
                u10 = float(wind_speed.values[time_index, lat_index, lon_index])
                spec1d = spectrum.values[time_index, lat_index, lon_index, :]

                if not np.isfinite(u10) or u10 <= 0.0:
                    continue
                if not np.all(np.isfinite(spec1d)):
                    continue

                swell = SwellWave(u10=u10, spec1d=spec1d, freq_band=frequency)
                tau[time_index, lat_index, lon_index] = swell.tau_chen()
                swell_index[time_index, lat_index, lon_index] = swell.swell_index()

    result = dataset.copy()
    result["tau_chen"] = xr.DataArray(
        tau,
        dims=("time", "lat", "lon"),
        coords={
            "time": dataset["time"],
            "lat": dataset["lat"],
            "lon": dataset["lon"],
        },
        attrs={
            "long_name": "Chen wave-induced stress parameter",
            "source": "Computed from SWAN 1D spectrum and WRF wind speed",
            "method": "Chen swell/wind-sea partitioning",
        },
    )
    result["swell_index"] = xr.DataArray(
        swell_index,
        dims=("time", "lat", "lon"),
        coords={
            "time": dataset["time"],
            "lat": dataset["lat"],
            "lon": dataset["lon"],
        },
        attrs={
            "long_name": "Swell energy fraction",
            "source": "Computed from SWAN 1D spectrum",
            "method": "m0_swell / m0_total",
            "units": "1",
        },
    )

    result.attrs.update(
        {
            "tau_chen_source": (
                "Tau-Chen computed from SWAN 1D wave spectrum and WRF wind speed "
                "from a coupled WRF-SWAN simulation."
            ),
            "tau_chen_reference": "Chen swell/wind-sea partitioning implemented from project notebook.",
        }
    )

    return result


def export_tau_chen_netcdf(
    input_path: str | Path,
    output_path: str | Path,
    spectrum_var: str = "spec_1d",
    wind_speed_var: str = "wind_speed",
    frequency_coord: str = "frequency",
) -> xr.Dataset:
    dataset = xr.open_dataset(input_path)
    result = compute_tau_chen_dataset(
        dataset=dataset,
        spectrum_var=spectrum_var,
        wind_speed_var=wind_speed_var,
        frequency_coord=frequency_coord,
    )
    result.to_netcdf(output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute tau-Chen from a spectrum NetCDF."
    )
    parser.add_argument("--input", required=True, help="Input NetCDF file.")
    parser.add_argument("--output", required=True, help="Output NetCDF file.")
    parser.add_argument(
        "--spectrum-var",
        default="spec_1d",
        help="Spectrum variable name in the NetCDF file.",
    )
    parser.add_argument(
        "--wind-speed-var",
        default="wind_speed",
        help="Wind-speed variable name in the NetCDF file.",
    )
    parser.add_argument(
        "--frequency-coord",
        default="frequency",
        help="Frequency coordinate name in the NetCDF file.",
    )
    args = parser.parse_args()

    result = export_tau_chen_netcdf(
        input_path=args.input,
        output_path=args.output,
        spectrum_var=args.spectrum_var,
        wind_speed_var=args.wind_speed_var,
        frequency_coord=args.frequency_coord,
    )
    print(result)
    print(f"NetCDF written to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
