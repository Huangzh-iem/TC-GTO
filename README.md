# TC-GTO

TC-GTO is a graph-temporal neural operator for reconstructing structural response fields and ground excitation from sparse floor acceleration measurements. The model combines geometry-aware low-frequency transmissibility features with heterogeneous spatial message passing and temporal sequence modeling.

## Contents

- `scripts/generate_opensees_dataset.py`: builds the five-family OpenSeesPy simulation set.
- `scripts/prepare_lf_features.py`: prepares low-frequency transmissibility features from the generated training set.
- `scripts/train.py`: trains TC-GTO.
- `tcgto/`: model and structural-system implementation.

## Setup

```powershell
pip install -r requirements.txt
```

## Run

```powershell
python scripts/opensees_smoke_test.py
python scripts/generate_opensees_dataset.py --workers 8
python scripts/prepare_lf_features.py
python scripts/train.py --train
```

Generated datasets, checkpoints, and run outputs are written locally to `data/` and `runs/`; neither directory is tracked by Git.
