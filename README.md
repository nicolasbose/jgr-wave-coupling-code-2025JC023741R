# Supporting Information – Data and Code for Review

## Manuscript

**Title:**
The Role of Wave-Induced Stress and Drag Coefficients in Offshore Wind Power Production

**Journal:**
Journal of Geophysical Research: Oceans

---

## Repository Description

This repository contains the analysis scripts and model configuration files used in the study. It is provided to support reproducibility during the peer review process.

Due to file size constraints and data policies, not all datasets are included in this repository.

---

## Data Availability Status

Data and code archiving are currently underway.

The authors plan to make all datasets and scripts publicly available through a GitHub repository, which will be permanently archived in Zenodo upon manuscript acceptance, ensuring long-term preservation and DOI assignment.

For the purpose of peer review, all necessary data, model outputs, and supporting materials have been provided as **Supporting Information**.

---

## Repository Structure

```text
.
├── drag_plots.py
├── tau_chen_from_netcdf.py
├── wave_stress_plots.py
├── wind_validation_energy_analysis.py
│
├── model_configuration/
│   ├── WRF configuration files
│   ├── SWAN input files
│   └── coupling configuration
│
├── database/        # Placeholder (data not included here)
├── figures/         
```

---

## Reproducibility Instructions

To reproduce the main results:

1. Install the required Python dependencies (NumPy, SciPy, xarray, matplotlib, etc.)

2. Execute the scripts in the following order:

   1. `tau_chen_from_netcdf.py`
   2. `wind_validation_energy_analysis.py`
   3. `wave_stress_plots.py`
   4. `drag_plots.py`

3. Generated outputs include:

   * statistical validation metrics
   * drag coefficient analysis
   * wave-induced stress diagnostics
   * wind energy production estimates

---

## Data Sources

* **COAWST**: publicly available
* **ERA5 reanalysis data**: publicly available via the Copernicus Climate Data Store
* **GEBCO bathymetry**: publicly available
* **LiDAR wind observations**: proprietary dataset subject to institutional restrictions

---

## Notes on Data Availability

* Full-resolution datasets (e.g., NetCDF model outputs and LiDAR observations) are not included in this repository due to file size and access restrictions.
* A subset of data and/or processed outputs sufficient for reproducibility is provided in the Supporting Information submitted with the manuscript.

---

## Contact

Data and materials can be made available upon request during the review process.

