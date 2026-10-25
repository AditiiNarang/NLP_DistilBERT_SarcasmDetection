# Sarcasm Detection Project Summary

This project fine-tuned a DistilBERT model for binary sarcasm detection using a balanced training set from the `train-balanced-sarcasm.csv` dataset.

## Current status

- Model: DistilBERT base uncased
- Task: binary classification
- Train sample size: 2,000 rows (local run)
- Validation split: 80/20
- Output directory: `models/distilbert_sarcasm/final_model`

## Training results

The training run produced a loss curve with a clear drop during the first epoch, including these highlights:

- Step 10: loss = 0.7389
- Step 40: loss = 0.2686
- Step 80: loss = 0.2509
- Final recorded loss: 0.4021 at step 100

The metrics are saved in `results/training_metrics.csv` and visualized in `results/training_loss.png`.

## Saved artifacts

- `models/distilbert_sarcasm/final_model/` — trained model weights
- `results/training_metrics.csv` — training loss metrics
- `results/training_loss.png` — loss visualization
- `results/sample_predictions.csv` — sample inference outputs
- `visualizations/` — class distribution and text-length plots

## Example inference

The project includes `inference.py`, which loads the trained model and classifies a small set of sarcastic and non-sarcastic examples.

To run it:

```bash
python inference.py
```

This produces a CSV report in `results/sample_predictions.csv` with labels, confidence, and prediction probability.

## Project value

The model is ready for small-scale experimentation, demo use, and extension to larger data or stronger transformer backbones. It is a clean starting point for a sarcasm classification app, research notebook, or deployment pipeline.
