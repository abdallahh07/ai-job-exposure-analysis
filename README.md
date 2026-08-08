# Will AI Take My Job? — Exposure, Skills & Wages

## Summary (Non-Technical)

This project looks at 271 U.S. jobs and asks a simple question: **which jobs are most likely to be affected by AI, and why?**

Instead of guessing, it uses real data from the U.S. government's own labor statistics (wages, employment, education requirements) combined with published academic research measuring how exposed each job is to AI tools like ChatGPT.

**What we found:**
- Jobs that rely heavily on **verbal and analytical skills** (writing, research, financial analysis) tend to be the most exposed to AI — occupations like financial examiners, actuaries, and budget analysts sit at the high end.
- Jobs that rely on **spatial and physical skills** (construction, mechanical trades) tend to be the least exposed — roofers and electricians sit at the low end.
- We built a model that predicts a job's AI exposure level (Low / Medium / High) using its required skills and wage data. It correctly predicts the right category about **67% of the time** — solid for a dataset this small (for comparison, random guessing would only be right about 33% of the time), but not precise enough to rely on for any single occupation in isolation.
- The model consistently struggles most with the "Medium" category — which makes intuitive sense, since it sits between two more clearly defined extremes.

**Bottom line**: this is a useful exploratory tool for understanding *patterns* across occupations, not a precise predictor for any one job. A larger dataset (expanding from 271 to the ~774 occupations covered by the original academic research) would likely improve accuracy further — noted as a next step below.

---

## Technical Overview

### Data Sources

| Source | What it provides |
|---|---|
| U.S. Bureau of Labor Statistics (BLS) | Median wage, employment counts, 10-year growth projections, required education (May 2024 data, 2024–2034 projections) |
| O\*NET | Cognitive ability profiles per occupation — verbal, reasoning, quantitative, memory, perceptual, spatial, attention |
| Eloundou et al., "GPTs are GPTs" | Share of occupational tasks exposed to large language models |
| Felten AIOE | Broader AI-occupational-exposure index |

### Target Variable

`ai_exposure_level` (Low / Medium / High) — a 3-class classification target derived from the continuous AIOE exposure score.

### Dataset Size & Limitation

271 occupations (~216 training rows after an 80/20 split). This is a small sample for machine learning, and it directly affected modeling decisions throughout — see the Model Selection and Hyperparameter Tuning sections below.

### Model Comparison

Four classifiers were compared using 5-fold stratified cross-validation: Logistic Regression, Random Forest, XGBoost, and LightGBM. The tree-based ensembles consistently overfit (training accuracy ~0.97, test accuracy ~0.65–0.67), while Logistic Regression showed the smallest train-test gap and the most stable cross-validation scores — consistent with the general principle that simpler models tend to generalize better on small datasets.

### Hyperparameter Tuning

`RandomizedSearchCV` was used to tune Logistic Regression's regularization (`C`, `penalty`, `solver`). The search selected `C=0.1` (relatively strong regularization) and `penalty='l2'`, narrowing the train-test gap from 0.128 to 0.083 without changing test accuracy — a more honestly-calibrated model, even though the headline accuracy number stayed flat. Full analysis in `hyperparameter_analysis.md`.

### Final Result

- **Test accuracy: 67.3%** (vs. 33% random-guess baseline for 3 classes)
- **Weakest class**: "Medium" exposure (precision 0.47, recall 0.56) — consistent across every model and configuration tested, suggesting this reflects a genuine property of the data rather than a fixable modeling issue.

### Key Finding

`oil_brent`-style dominant correlation doesn't apply here, but a comparable pattern does: **verbal ability, reasoning ability, and quantitative ability** show the strongest positive correlation with AI exposure, while **spatial ability** shows a clear negative correlation — directly consistent with the dataset's own stated thesis.

### Limitations & Next Steps

- **Sample size**: 271 occupations is likely near the practical performance ceiling for this feature set. The full AIOE dataset (~774 occupations, same methodology, freely available) would substantially increase training data without changing the underlying approach.
- **Rare categories**: Some `job_category` values have very few occurrences, adding noise to one-hot encoded features — collapsing rare categories into "Other" is a planned refinement.
- **3-class framing**: Given the persistent weakness of the "Medium" class, a binary reframing (e.g., High vs. Not-High exposure) is worth testing as a more robust alternative.

### Project Structure

```
├── config/            # Configuration (paths, features, target)
├── processing/         # Data loading & feature engineering
├── pipeline.py          # Model pipeline definition
├── train_pipeline.py     # Training script
├── predict.py              # Prediction function
├── app/                       # FastAPI serving layer
├── Dockerfile
└── requirements.txt
```

### Setup

```bash
pip install -r requirements.txt
```

*(Training/serving instructions to be added once the production pipeline is finalized.)*
