# Turbofan Engine RUL Prediction

MSc Data Science dissertation project: a comparative study of machine learning models for predicting the **Remaining Useful Life (RUL)** of aircraft turbofan engines, using the NASA C-MAPSS FD001 dataset.

## Overview

RUL is the number of operating cycles an engine has left before failure. Predicting it helps maintenance teams plan work before a failure occurs. This project compares:

- Mean RUL baseline
- Linear Regression
- Random Forest
- Support Vector Regression (SVR)
- LSTM (sequences of 30 cycles)

Key choices in the pipeline:
- RUL capped at 125 cycles
- Engine-level train/validation split and grouped cross-validation, so cycles from the same engine never appear in both training and validation sets
- Hyperparameter tuning for Random Forest and SVR
- Fixed random seed (42) and CPU-only TensorFlow for reproducible results
- Evaluation on the official NASA FD001 test set, plus Random Forest feature importance

## Results (official NASA FD001 test set)

| Model | MAE (cycles) | RMSE (cycles) | R² | NASA PHM score |
| --- | --- | --- | --- | --- |
| LSTM (best overall) | 10.42 | 14.60 | 0.867 | 314 |
| Random Forest (best classical model) | 12.40 | 16.70 | 0.826 | 491 |

The LSTM was the most accurate. Random Forest was a strong alternative because it needs less computation and is easier to interpret. FD001 is simulated and covers one operating condition and one fault type, so the findings should not be applied directly to real aircraft engines without further testing. See the full dissertation for all models and analysis.

## Repository contents

| Path | Description |
| --- | --- |
| `rul_prediction.py` | Full pipeline: data loading, preprocessing, EDA, training, tuning, evaluation and interpretation |
| `Turbofan_RUL_Prediction_MSc_Dissertation.pdf` | The full written dissertation |
| `Data/` | NASA C-MAPSS FD001 files used by the script (see `Data/README.md`) |
| `requirements.txt` | Python dependencies |

## Data

The three NASA C-MAPSS FD001 files used by the script are included in the `Data/` folder, with column headers added. The original dataset is from the NASA Prognostics Center of Excellence.

## How to run

```bash
pip install -r requirements.txt
python rul_prediction.py
```

Run the script from the repository's top-level folder so it can find `Data/`. Tables, figures and trained models are saved to an `Outputs/` folder.

## Author

Victor Osasumwen Ogedegbe, MSc Data Science, University of East London.

## Acknowledgements

NASA Prognostics Center of Excellence for the C-MAPSS dataset.
