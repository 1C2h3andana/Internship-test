# AgriAI CLI v3.0 - Agile Project Documentation

## Project Vision

Build a comprehensive AI-powered agricultural CLI tool that helps farmers make data-driven decisions about crop management, disease detection, resource optimization, and financial planning -- using only Python standard library with custom ML algorithms.

## Agile Methodology

This project follows Scrum with 7 two-week sprints, each delivering working, tested features.

---

## Sprint 1 - MVP (Core ML Models)

**Goal:** Deliver four foundational AI models for crop management.

### User Stories
| ID | Story | Points | Status |
|----|-------|--------|--------|
| US-01 | As a farmer, I want to identify crop diseases from symptom descriptions | 8 | Done |
| US-02 | As a farmer, I want to predict crop yield from environmental conditions | 8 | Done |
| US-03 | As a farmer, I want soil type classification and crop recommendations | 5 | Done |
| US-04 | As a farmer, I want weather risk assessment for my crops | 5 | Done |
| US-05 | As a developer, I want a custom ML library with no dependencies | 13 | Done |

### Deliverables
- `ml_algorithms.py` - Custom ML library (RF, GBR, KNN, DT, NB, LR, NN, TF-IDF)
- `disease_detector.py` - Random Forest + TF-IDF classifier
- `yield_predictor.py` - Gradient Boosting regressor
- `soil_analyzer.py` - KNN soil classifier
- `weather_risk.py` - Multi-factor risk scorer

### Acceptance Criteria
- [x] Disease detector achieves >0% accuracy on test set (synthetic data)
- [x] Yield predictor returns positive kg values with confidence intervals
- [x] Soil analyzer identifies 6 soil types with nutrient scoring
- [x] Weather risk scores 5 dimensions for 7 crop types

---

## Sprint 2 - Pest Management & Irrigation

**Goal:** Add IPM pest management and precision irrigation scheduling.

### User Stories
| ID | Story | Points | Status |
|----|-------|--------|--------|
| US-06 | As a farmer, I want pest identification with IPM treatment plans | 8 | Done |
| US-07 | As a farmer, I want science-based irrigation scheduling | 8 | Done |

### Deliverables
- `pest_manager.py` - Decision Tree pest classifier with 4-tier IPM strategy
- `irrigation_scheduler.py` - FAO-56 Penman-Monteith ET0 calculation

### Acceptance Criteria
- [x] Identifies 8 pest types with cultural/biological/mechanical/chemical controls
- [x] Calculates ET0, ETc, and net irrigation requirement
- [x] Compares drip, sprinkler, and flood irrigation methods

---

## Sprint 3 - Crop Rotation & Market Analysis

**Goal:** Add crop rotation optimization and commodity price forecasting.

### User Stories
| ID | Story | Points | Status |
|----|-------|--------|--------|
| US-08 | As a farmer, I want optimal crop rotation plans for soil health | 5 | Done |
| US-09 | As a farmer, I want market price trends and selling signals | 8 | Done |

### Deliverables
- `crop_rotation.py` - Multi-objective rotation optimizer
- `market_analyzer.py` - Time-series price analysis with exponential smoothing

### Acceptance Criteria
- [x] Generates multi-season rotation considering family diversity and N-fixation
- [x] Provides 6-month price forecast with confidence intervals
- [x] Calculates volatility and trading signals

---

## Sprint 4 - Farm Tracking & Calendar

**Goal:** Add persistent data tracking and seasonal planning tools.

### User Stories
| ID | Story | Points | Status |
|----|-------|--------|--------|
| US-10 | As a farmer, I want to track my farm activities and yield history | 5 | Done |
| US-11 | As a farmer, I want a seasonal calendar with government scheme info | 5 | Done |

### Deliverables
- `farm_tracker.py` - JSON-based persistent farm history database
- `seasonal_calendar.py` - 3-season calendar with 6 government schemes

### Acceptance Criteria
- [x] Stores and retrieves farm records with yield trend analysis
- [x] Shows crop activities by month with upcoming schedule
- [x] Lists relevant government schemes (PM-KISAN, PMFBY, KCC, etc.)

---

## Sprint 5 - Precision Fertilization & Reporting

**Goal:** Add precision fertilizer calculation and multi-format data export.

### User Stories
| ID | Story | Points | Status |
|----|-------|--------|--------|
| US-12 | As a farmer, I want precise fertilizer recommendations based on soil tests | 8 | Done |
| US-13 | As a farmer, I want to export analysis reports in multiple formats | 5 | Done |

### Deliverables
- `fertilizer_calc.py` - ICAR target yield method with NPK optimization
- `export_manager.py` - JSON, CSV, and TXT report generation

### Acceptance Criteria
- [x] Calculates DAP, Urea, MOP quantities with split application schedule
- [x] Accounts for soil nutrient availability and pH adjustment
- [x] Exports to JSON, CSV, and formatted TXT reports

---

## Sprint 6 - Sustainability & Remote Sensing

**Goal:** Add carbon footprint analysis, financial tracking, and satellite crop monitoring.

### User Stories
| ID | Story | Points | Status |
|----|-------|--------|--------|
| US-14 | As a farmer, I want to know my farm's carbon footprint | 8 | Done |
| US-15 | As a farmer, I want to track farm income and expenses | 5 | Done |
| US-16 | As a farmer, I want satellite-based crop health monitoring | 8 | Done |

### Deliverables
- `carbon_footprint.py` - IPCC emission factor calculation with reduction strategies
- `financial_ledger.py` - Income/expense tracking with profitability analysis
- `ndvi_simulator.py` - Simulated satellite NDVI analysis with field mapping

### Acceptance Criteria
- [x] Calculates emissions from 8+ sources with national benchmarking
- [x] Tracks 10 expense and 5 income categories with ROI calculation
- [x] Generates pixel-level NDVI maps with anomaly detection
- [x] Provides carbon credit potential estimation

---

## Sprint 7 - Enterprise & ML Operations

**Goal:** Add multi-farm management, crop insurance, and automated ML model tuning.

### User Stories
| ID | Story | Points | Status |
|----|-------|--------|--------|
| US-17 | As a farm manager, I want to manage multiple farms with comparison | 8 | Done |
| US-18 | As a farmer, I want crop insurance premium calculation and claim guidance | 5 | Done |
| US-19 | As a developer, I want automated model comparison and hyperparameter tuning | 13 | Done |

### Deliverables
- `multi_farm.py` - Multi-farm portfolio management with comparison analytics
- `crop_insurance.py` - PMFBY premium calculator with claim process guidance
- `ml_autotuner.py` - Grid search across 8+ model configurations

### Acceptance Criteria
- [x] Registers multiple farms with seasonal performance tracking
- [x] Calculates PMFBY premiums with government subsidy breakdown
- [x] Compares 8 classifier configs and 6 regressor configs with timing
- [x] Provides overfitting detection and speed/accuracy tradeoff analysis

---

## Sprint Velocity

| Sprint | Story Points | Features Delivered | Tests Added |
|--------|-------------|-------------------|-------------|
| Sprint 1 | 39 | 5 (+ ML library) | 28 |
| Sprint 2 | 16 | 2 | 6 |
| Sprint 3 | 13 | 2 | 6 |
| Sprint 4 | 10 | 2 | 6 |
| Sprint 5 | 13 | 2 | 6 |
| Sprint 6 | 21 | 3 | 10 |
| Sprint 7 | 26 | 3 | 10 |
| **Total** | **138** | **19** | **79** |

## Definition of Done

1. Feature code is complete and follows project patterns
2. All existing tests continue to pass
3. New feature has at least 2 tests
4. Feature is accessible via CLI command
5. Code uses only Python standard library
6. Documentation is updated

## Technical Debt Register

| Item | Priority | Status |
|------|----------|--------|
| ML models use synthetic data only | Medium | Known limitation |
| No persistent model serialization (retrains each run) | Low | By design (no pickle) |
| Terminal UI assumes UTF-8 support | Low | Fallback box chars available |
| Financial data not encrypted | Medium | Future enhancement |

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|-----------|
| No real training data | High | Synthetic data generators with domain knowledge |
| No external ML libraries | Medium | Custom implementations validated against known results |
| Single-threaded execution | Low | Acceptable for CLI tool |
