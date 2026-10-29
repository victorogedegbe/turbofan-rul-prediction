# Data Folder — NASA C-MAPSS FD001

This folder contains the three official NASA C-MAPSS FD001 files used in
`rul_prediction.py`. All three are required — the script will not run with
any one of them missing.

## Files

### `train_FD001.csv`
100 engines, each one **run all the way to failure**. Because the complete
life of every engine is recorded, Remaining Useful Life (RUL) can be
calculated directly from this file: `RUL = final_cycle − current_cycle`
(capped at 125, per Section 3.5.2 of the dissertation). This is the file
used to train every model (Steps 1–8 of the script).

### `test_FD001.csv`
A different set of 100 engines, but each one is **truncated part-way
through its life, before failure**. Sensor readings are only provided up
to an arbitrary stopping point — the actual failure is never shown. This
is the unseen data the final trained models are evaluated against
(Step 9).

### `RUL_FD001.csv`
One true RUL value per test engine — the number of cycles remaining at
the exact point each `test_FD001.csv` trajectory was cut off. Because the
test file stops before failure, its true RUL **cannot** be calculated the
way it is for the training file, so NASA supplies it separately as the
ground truth used to score predictions (Step 9). It is never used for
training, only for evaluation.

## Why three files instead of one

This design deliberately mirrors a real prognostics scenario: in practice
you never have sensor data running all the way to failure, only data
"up to now," and the model must predict what's left. The training set
teaches the model what full degradation looks like; the test set forces
it to predict RUL from a partial, unfinished trajectory — the actual task
RUL prediction exists to solve. `RUL_FD001.csv` is simply the answer key
used to grade those predictions, since it cannot be inferred from the
truncated test data itself.

## Source

NASA Prognostics Center of Excellence, C-MAPSS dataset, FD001 subset
(Saxena et al., 2008).
