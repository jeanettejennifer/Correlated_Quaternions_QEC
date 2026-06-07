# IQM Emerald Surface-Code QEC Pipelines

This repository contains the project-submission version of our IQM Emerald QEC work.

## Main Files

- `iqm_qec_pipeline.py`  
  Standard rotated surface-code memory experiment pipeline:
  Stim circuit generation, IQM layout optimization, Stim-to-Qiskit conversion,
  hardware/synthetic execution, syndrome extraction, MWPM decoding, and logical
  error-rate fitting.

- `snl_iqm_pipeline.py`  
  Defect-aware Snakes-and-Ladders surface-code pipeline:
  builds defective/deformed patches, runs hardware or calibration-noise
  simulations, and extracts logical error rates over round sweeps.

- `helper_visualisation.py`  
  Visualization utilities for IQM Emerald layouts, used patches, faulty
  couplers, gate timeslices, and Snakes-and-Ladders layouts.

- `qec_pipeline_showcase.ipynb`  
  Notebook showing the main functionality and project results.

## Data

- `calibration_data/` contains the IQM calibration JSON files used by the
  pipelines and synthetic noise models.

## Setup

```bash
pip install -r requirements.txt
```

For Snakes-and-Ladders functionality, `snl_iqm_pipeline.py` automatically
clones the Amazon Science repository on first import:

```bash
git clone https://github.com/amazon-science/snakes_and_ladders_adapting_the_surface_code_to_defects.git
```

If network access is unavailable, clone that repository manually next to this
repository, or set `SNL_REPO` to an existing checkout.

For hardware runs, provide an IQM Resonance token in the notebook or via:

```bash
export IQM_TOKEN="..."
export IQM_API_URL="https://resonance.iqm.tech/"
```
