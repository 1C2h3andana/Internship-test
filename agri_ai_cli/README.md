# AgriAI CLI v3.0 - AI-Powered Crop Disease Detection & Farm Intelligence

A comprehensive AI-powered agricultural CLI application built entirely in **Python stdlib** (no external dependencies). Implements 18 features across 7 Agile sprints, including custom ML algorithms built from scratch.

## Features (18 total)

| # | Feature | Sprint | Algorithm/Method |
|---|---------|--------|-----------------|
| 1 | Disease Detection | Sprint 1 | Random Forest + TF-IDF (custom) |
| 2 | Yield Prediction | Sprint 1 | Gradient Boosting Regressor (custom) |
| 3 | Soil Analysis | Sprint 1 | K-Nearest Neighbors (custom) |
| 4 | Weather Risk Assessment | Sprint 1 | Weighted Multi-Factor Scoring |
| 5 | Pest Management (IPM) | Sprint 2 | Decision Tree + IPM Knowledge Base |
| 6 | Irrigation Scheduling | Sprint 2 | FAO-56 Penman-Monteith |
| 7 | Crop Rotation Planning | Sprint 3 | Multi-Objective Graph Scoring |
| 8 | Market Price Analysis | Sprint 3 | Exponential Smoothing + Linear Trend |
| 9 | Farm History Tracker | Sprint 4 | JSON Persistent Database |
| 10 | Seasonal Calendar | Sprint 4 | Agricultural Knowledge Base |
| 11 | Fertilizer Calculator | Sprint 5 | ICAR Target Yield Method |
| 12 | Export Manager | Sprint 5 | JSON/CSV/TXT Report Generation |
| 13 | Carbon Footprint Analyzer | Sprint 6 | IPCC Emission Factor Method |
| 14 | Farm Financial Ledger | Sprint 6 | JSON Database + Analytics |
| 15 | Satellite NDVI Simulator | Sprint 6 | Simulated Remote Sensing |
| 16 | Multi-Farm Management | Sprint 7 | Portfolio Analytics |
| 17 | Crop Insurance Advisor | Sprint 7 | PMFBY Premium Calculator |
| 18 | ML AutoTuner | Sprint 7 | Grid Search + Model Comparison |

## Custom ML Library (No Dependencies)

All machine learning algorithms are implemented from scratch using only Python stdlib:

- **TF-IDF Vectorizer** - Text vectorization for NLP
- **Naive Bayes Classifier** - Gaussian naive bayes
- **K-Nearest Neighbors** - Distance-based classification
- **Decision Tree** (Classifier + Regressor) - Gini impurity / variance reduction
- **Random Forest** - Bootstrap aggregation of decision trees
- **Gradient Boosting Regressor** - Additive ensemble of decision stumps
- **Linear Regression** - Gradient descent optimization
- **Simple Neural Network** - Multi-layer perceptron with backpropagation

## Project Structure

```
agri_ai_cli/
├── main.py                          # CLI entry point (18 commands)
├── README.md                        # This file
├── AGILE_DOCS.md                    # Full Agile project documentation
├── core/
│   ├── ai_advisor.py                # NLP query routing engine
├── models/                          # Sprint 1 - Core ML models
│   ├── disease_detector.py          # Random Forest + TF-IDF
│   ├── yield_predictor.py           # Gradient Boosting
│   ├── soil_analyzer.py             # KNN classifier
│   └── weather_risk.py              # Multi-factor scorer
├── features/                        # Sprint 2-7 feature modules
│   ├── pest_manager.py              # Decision Tree + IPM (Sprint 2)
│   ├── irrigation_scheduler.py      # FAO-56 method (Sprint 2)
│   ├── crop_rotation.py             # Graph scoring (Sprint 3)
│   ├── market_analyzer.py           # Time-series analysis (Sprint 3)
│   ├── farm_tracker.py              # JSON database (Sprint 4)
│   ├── seasonal_calendar.py         # Activity planner (Sprint 4)
│   ├── fertilizer_calc.py           # ICAR method (Sprint 5)
│   ├── export_manager.py            # Report export (Sprint 5)
│   ├── carbon_footprint.py          # Emission analysis (Sprint 6)
│   ├── financial_ledger.py          # Financial tracking (Sprint 6)
│   ├── ndvi_simulator.py            # Satellite simulation (Sprint 6)
│   ├── multi_farm.py                # Portfolio management (Sprint 7)
│   ├── crop_insurance.py            # PMFBY calculator (Sprint 7)
│   └── ml_autotuner.py              # Model comparison (Sprint 7)
├── utils/
│   ├── terminal_ui.py               # Pure Python ANSI terminal UI
│   ├── ml_algorithms.py             # Custom ML library (stdlib only)
│   └── data_generator.py            # Synthetic training data
├── tests/
│   └── test_all.py                  # 79 tests across all features
└── data/                            # Runtime data storage
```

## Quick Start

```bash
# No installation needed - pure Python 3.6+
cd agri_ai_cli

# Show all commands
python3 main.py dashboard

# Run full demo (all 18 features)
python3 main.py demo

# Individual commands
python3 main.py detect "rice plant brown spots on leaves"
python3 main.py predict-yield
python3 main.py analyze-soil
python3 main.py weather
python3 main.py pest
python3 main.py irrigate
python3 main.py rotate
python3 main.py market
python3 main.py fertilizer
python3 main.py carbon
python3 main.py insurance
python3 main.py ndvi
python3 main.py finance
python3 main.py multi-farm
python3 main.py autotuner
python3 main.py train
```

## Requirements

- Python 3.6+ (stdlib only - no pip install needed)
- No external dependencies whatsoever

## Testing

```bash
python3 agri_ai_cli/tests/test_all.py
# Expected: 79/79 tests passed
```

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                  CLI Entry (main.py)                 │
│              18 commands + dashboard                 │
├─────────────────────────────────────────────────────┤
│                Presentation Layer                    │
│     terminal_ui.py (ANSI colors, tables, bars)      │
├─────────────────────────────────────────────────────┤
│                  ML Model Layer                      │
│  ml_algorithms.py (RF, GBR, KNN, DT, NB, LR, NN)  │
├─────────────────────────────────────────────────────┤
│               Feature Modules Layer                  │
│  disease | yield | soil | weather | pest | irrigate │
│  rotation | market | tracker | calendar | fertilizer│
│  carbon | finance | ndvi | multi_farm | insurance   │
├─────────────────────────────────────────────────────┤
│              Data & Knowledge Layer                  │
│  data_generator.py | JSON databases | Knowledge DBs │
└─────────────────────────────────────────────────────┘
```

## Lines of Code

| Component | Lines |
|-----------|-------|
| Custom ML Library | ~550 |
| Terminal UI | ~250 |
| Data Generator | ~350 |
| Sprint 1 Models | ~550 |
| Sprint 2-5 Features | ~1,100 |
| Sprint 6 Features | ~600 |
| Sprint 7 Features | ~550 |
| Main CLI | ~450 |
| Test Suite | ~250 |
| **Total** | **~4,650** |
