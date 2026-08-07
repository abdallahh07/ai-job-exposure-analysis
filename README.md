# Will AI Take My Job? — Exposure, Skills & Wages

Analyzing which U.S. occupations are most exposed to AI, and how that exposure relates to required skills and wages.

## Overview

This project explores 271 U.S. occupations using real, named-source data — not a synthetic dataset — combining AI exposure research, cognitive skill profiles, and labor market statistics to understand which kinds of work (and which human skills) are most and least exposed to AI.

*Modeling approach and target variable: TBD.*

## Data Sources

| File | Contents |
|---|---|
| `ai_job_exposure.csv` | AI exposure scores (LLM task exposure, Felten AIOE index), job category, education required, median wage, employment figures |
| `cognitive_ability_ai_exposure.csv` | Cognitive ability scores (verbal, reasoning, quantitative, memory, perceptual, spatial, attention) per occupation, from O*NET |
| `occupation_cognitive_profile.csv` | Occupation-level cognitive skill profiles |

Sources: U.S. Bureau of Labor Statistics (BLS, May 2024 wage/employment data, 2024–2034 projections), O*NET, and published AI-exposure research (Eloundou et al. "GPTs are GPTs"; Felten AIOE).

## Project Structure

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

## Status

🚧 Early stage — data collected.

## Setup

```bash
pip install -r requirements.txt
```

*(Training/serving instructions to be added once the pipeline is built.)*
