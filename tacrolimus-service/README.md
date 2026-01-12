# Tacrolimus Dose Prediction

PyTorch-based machine learning models for predicting individualized tacrolimus dosing in kidney transplant patients during the early post-operative period (POD 0-8).

## Overview

This project develops ML models to automatically determine tacrolimus dosing for preventing acute rejection after kidney transplantation. The models address challenges in dose determination due to narrow therapeutic index and high pharmacokinetic variability.

## Installation

```bash
git clone https://github.com/yourusername/tacrolimus-dose-prediction.git
cd tacrolimus-dose-prediction
pip install -r requirements.txt
```

Place data files in `data/`:
- `data/Tac_dose_tdm.csv`
- `data/2017_2022_registry.xlsx`

## Usage

### Basic Training

```bash
# Tabular model
python tacrolimus_dose_prediction.py --model_type both --epochs 100

# Time-series model (LSTM)
python tacrolimus_dose_prediction.py --model_type both --use_timeseries --epochs 100

# With static features
python tacrolimus_dose_prediction.py --model_type both --use_timeseries --use_static_features --epochs 100
```

## Model Architecture

### Tabular Models
- **PM Model**: `[prev_dose_pm, today_dose_am, today_tdm]` → `TODAY_DOSE_PM`
- **AM Model**: `[prev_dose_pm, today_dose_am, today_tdm, TODAY_DOSE_PM]` → `NEXT_DOSE_AM`
- Supports `--time_window` for historical data

### Time-Series Models (RNN)
- **PM Model**: Sequence data `(batch_size, seq_len, 3)` where each step is `[prev_dose_pm, today_dose_am, today_tdm]` → `TODAY_DOSE_PM`
- **AM Model**: Sequence data `(batch_size, seq_len, 4)` where each step is `[prev_dose_pm, today_dose_am, today_tdm, today_dose_pm]` → `NEXT_DOSE_AM`
- Supports LSTM/GRU with optional static features


## Data

- **Source**: Asan Medical Center KT cohort (2017.01-2022.12)
- **Patients**: 2,010 patients, 18,717 dosing events
- **Filtering**: 4-23 events per patient, dose ≤ 7.0mg