# TC-REJECT

Anonymous implementation accompanying *When Coarse Annotations Conceal Novel States:
TC-REJECT for Auditing Time-Series Labels*.

This minimal release contains the TC-REJECT pipeline, its synthetic data generator, dataset
adapters, the main command-line runner, and the five final rendered manuscript figures. It does
not contain raw data, result caches, private clinical material, development notebooks, search
histories, or internal diagnostic scripts.

## Contents

- `tcreject/`: features, grouped calibration, learned channel, e-sum fusion, rank-average
  fusion, evaluation, and dataset adapters.
- `tcreject/datasets/synthetic.py`: controlled multichannel synthetic generator.
- `run_tcreject.py`: main runner for synthetic, PAMAP2, and AJILE12 experiments.
- `run_dbs.py`: runner for an authorized local copy of the protected DBS recordings.
- `extract_ajile12.py`: extraction of public AJILE12 sessions from Dandiset 000055.
- `figures/Figure1.png` through `figures/Figure5.png`: final rendered figures extracted from
  the submission PDF.

## Installation

```text
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

On Linux or macOS, use `.venv/bin/python`.

## Run the synthetic benchmark

```text
python run_tcreject.py synthetic
```

To run selected fixed seeds:

```text
python run_tcreject.py synthetic --seeds 11,23
```

The saved score file contains the adaptive e-sum score under `TC-REJECT` and the second reported
fusion route under `TC-REJECT(rank-avg)`.

## Public measured datasets

PAMAP2 must be downloaded from the UCI Machine Learning Repository. Point `--data_root` to a
directory containing `Protocol/subject101.dat` through `Protocol/subject109.dat`.

AJILE12 is available as Dandiset 000055. Extract a session before running the adapter:

```text
python extract_ajile12.py --subject sub-01 --session ses-4 --data_root <data_root>
python run_tcreject.py ajile --subject sub-01 --session ses-4 --data_root <data_root>
```

The DBS recordings are protected and are not distributed. `run_dbs.py` is provided only for
authorized local use with the expected input layout documented in
`tcreject/datasets/dbs_adapter.py`.

## Anonymity and data policy

Do not commit downloaded datasets, model checkpoints, result caches, credentials, or protected
clinical recordings. The included `.gitignore` excludes common forms of these files.
