#!/usr/bin/env python3
"""
Comprehensive test suite for AgriAI CLI v3.0.
Tests all 18 features across 7 sprints.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

passed = 0
failed = 0
total = 0


def test(name, condition):
    global passed, failed, total
    total += 1
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}")


def run_tests():
    global passed, failed, total
    print("=" * 60)
    print("  AgriAI CLI v3.0 - Test Suite")
    print("=" * 60)

    # ── Utils Tests ──────────────────────────────────────────
    print("\n--- Utils: ML Algorithms ---")
    from agri_ai_cli.utils.ml_algorithms import (
        dot, vec_mean, sigmoid, softmax, TFIDFVectorizer,
        NaiveBayesClassifier, KNNClassifier, DecisionTreeClassifier,
        RandomForestClassifier, LinearRegression, GradientBoostingRegressor,
        train_test_split, confusion_matrix
    )

    test("dot product", dot([1, 2, 3], [4, 5, 6]) == 32)
    test("vec_mean", vec_mean([[1, 2], [3, 4]]) == [2.0, 3.0])
    test("sigmoid(0) = 0.5", abs(sigmoid(0) - 0.5) < 0.01)
    test("softmax sums to 1", abs(sum(softmax([1, 2, 3])) - 1.0) < 0.01)

    # TF-IDF
    v = TFIDFVectorizer(max_features=50)
    docs = ["hello world", "world of python", "hello python programming"]
    X = v.fit_transform(docs)
    test("TF-IDF produces vectors", len(X) == 3 and len(X[0]) > 0)

    # Train-test split
    X_data = [[i] for i in range(100)]
    y_data = list(range(100))
    X_tr, X_te, y_tr, y_te = train_test_split(X_data, y_data, test_ratio=0.2)
    test("train_test_split sizes", len(X_tr) == 80 and len(X_te) == 20)

    print("\n--- Utils: Data Generator ---")
    from agri_ai_cli.utils.data_generator import (
        generate_disease_data, generate_yield_data, generate_soil_data,
        generate_pest_data, generate_weather_scenarios, generate_market_data,
        generate_carbon_data
    )
    texts, labels = generate_disease_data(50)
    test("disease data generation", len(texts) == 50 and len(labels) == 50)

    X, y, crops = generate_yield_data(50)
    test("yield data generation", len(X) == 50 and len(y) == 50)

    X, y, types = generate_soil_data(50)
    test("soil data generation", len(X) == 50 and len(types) > 0)

    X, y, pests, crops = generate_pest_data(50)
    test("pest data generation", len(X) == 50)

    scenarios = generate_weather_scenarios(10)
    test("weather scenarios", len(scenarios) == 10 and "risk_level" in scenarios[0])

    prices = generate_market_data("rice", 12)
    test("market data generation", len(prices) == 12)

    carbon = generate_carbon_data(10)
    test("carbon data generation", len(carbon) == 10 and "total_carbon_kg" in carbon[0])

    # ── Sprint 1: Core Models ────────────────────────────────
    print("\n--- Sprint 1: Disease Detector ---")
    from agri_ai_cli.models.disease_detector import DiseaseDetector
    dd = DiseaseDetector()
    info = dd.train(200)
    test("disease model trains", dd.is_trained)
    test("disease accuracy > 0", info["accuracy"] > 0)
    result = dd.predict("rice plant has diamond shaped lesions gray centers")
    test("disease prediction works", "disease" in result and "confidence" in result)
    test("disease has treatment", len(result["treatment"]) > 0)

    print("\n--- Sprint 1: Yield Predictor ---")
    from agri_ai_cli.models.yield_predictor import YieldPredictor
    yp = YieldPredictor()
    info = yp.train(200)
    test("yield model trains", yp.is_trained)
    result = yp.predict(28, 1100, 75, 6.2, 150, 40, 80, 8, "rice")
    test("yield prediction works", result["predicted_yield_kg"] > 0)
    test("yield has revenue", result["revenue_inr"] > 0)
    test("yield has factors", len(result["factors"]) > 0)

    print("\n--- Sprint 1: Soil Analyzer ---")
    from agri_ai_cli.models.soil_analyzer import SoilAnalyzer
    sa = SoilAnalyzer()
    info = sa.train(200)
    test("soil model trains", sa.is_trained)
    result = sa.analyze(200, 35, 180, 6.5, 0.8, 45, 28)
    test("soil analysis works", "soil_type" in result)
    test("soil has health score", 0 <= result["health_score"] <= 1)
    test("soil has recommendations", len(result["recommended_crops"]) > 0)

    print("\n--- Sprint 1: Weather Risk ---")
    from agri_ai_cli.models.weather_risk import WeatherRiskAssessor
    wr = WeatherRiskAssessor()
    result = wr.assess(38, 85, 120, 25, 60, "rice")
    test("weather assessment works", "risk_level" in result)
    test("weather has risk scores", len(result["individual_risks"]) == 5)
    test("weather has advisories", len(result["advisories"]) > 0)

    # ── Sprint 2: Pest + Irrigation ──────────────────────────
    print("\n--- Sprint 2: Pest Manager ---")
    from agri_ai_cli.features.pest_manager import PestManager
    pm = PestManager()
    info = pm.train(200)
    test("pest model trains", pm.is_trained)
    result = pm.identify(28, 75, "rice", 8, 100)
    test("pest identification works", "pest" in result)
    test("pest has IPM strategy", len(result["ipm_strategy"]) == 4)

    print("\n--- Sprint 2: Irrigation Scheduler ---")
    from agri_ai_cli.features.irrigation_scheduler import IrrigationScheduler
    irr = IrrigationScheduler()
    result = irr.schedule("rice", 32, 65, 3, 8, 45, 2.5, "mid")
    test("irrigation schedule works", result["et0_mm_day"] > 0)
    test("irrigation has recommendation", "next_irrigation" in result["recommendation"])
    test("irrigation compares methods", len(result["method_comparison"]) == 3)

    # ── Sprint 3: Rotation + Market ──────────────────────────
    print("\n--- Sprint 3: Crop Rotation ---")
    from agri_ai_cli.features.crop_rotation import CropRotationPlanner
    crp = CropRotationPlanner()
    result = crp.plan("rice", "alluvial", 4)
    test("rotation plan works", len(result["rotation"]) == 4)
    test("rotation has scores", len(result["scores"]) > 0)
    test("rotation has soil health", "diversity_score" in result["soil_health_impact"])

    print("\n--- Sprint 3: Market Analyzer ---")
    from agri_ai_cli.features.market_analyzer import MarketAnalyzer
    ma = MarketAnalyzer()
    result = ma.analyze("rice", 24)
    test("market analysis works", "statistics" in result)
    test("market has trend", result["trend"]["direction"] in ("RISING", "FALLING", "STABLE"))
    test("market has forecast", len(result["forecast"]) == 6)

    # ── Sprint 4: Tracker + Calendar ─────────────────────────
    print("\n--- Sprint 4: Farm Tracker ---")
    from agri_ai_cli.features.farm_tracker import FarmTracker
    ft = FarmTracker(db_path="/tmp/test_farm_tracker.json")
    ft.add_record("rice", 2.5, 11000, "kharif", 45000, 242000)
    ft.add_record("wheat", 2.5, 9500, "rabi", 38000, 237500)
    records = ft.get_records()
    test("farm tracker records", len(records) == 2)
    summary = ft.get_summary()
    test("farm tracker summary", summary["total_records"] == 2)
    trend = ft.yield_trend()
    test("farm tracker trend", "trend_direction" in trend)

    print("\n--- Sprint 4: Seasonal Calendar ---")
    from agri_ai_cli.features.seasonal_calendar import SeasonalCalendar
    sc = SeasonalCalendar()
    result = sc.get_calendar(7)
    test("calendar works", result["current_season"] == "kharif")
    test("calendar has activities", isinstance(result["current_activities"], list))
    crop_cal = sc.get_crop_calendar("rice")
    test("crop calendar works", crop_cal["season"] == "kharif")

    # ── Sprint 5: Fertilizer + Export ────────────────────────
    print("\n--- Sprint 5: Fertilizer Calculator ---")
    from agri_ai_cli.features.fertilizer_calc import FertilizerCalculator
    fc = FertilizerCalculator()
    result = fc.calculate("rice", 2.5, 180, 30, 150, 6.5, 0.7)
    test("fertilizer calc works", result["total_cost_inr"] > 0)
    test("fertilizer has plan", len(result["fertilizer_plan"]) > 0)
    test("fertilizer has schedule", len(result["application_schedule"]) == 3)

    print("\n--- Sprint 5: Export Manager ---")
    from agri_ai_cli.features.export_manager import ExportManager
    em = ExportManager(output_dir="/tmp/test_exports")
    data = {"test": "data", "value": 42}
    json_path = em.export_json(data, "test.json")
    test("JSON export works", os.path.exists(json_path))
    csv_path = em.export_csv(["A", "B"], [["1", "2"], ["3", "4"]], "test.csv")
    test("CSV export works", os.path.exists(csv_path))
    txt_path = em.export_txt("Test report content", "test.txt")
    test("TXT export works", os.path.exists(txt_path))

    # ── Sprint 6: Carbon + Finance + NDVI ────────────────────
    print("\n--- Sprint 6: Carbon Footprint ---")
    from agri_ai_cli.features.carbon_footprint import CarbonFootprintAnalyzer
    cfa = CarbonFootprintAnalyzer()
    result = cfa.analyze("rice", 2.5, 150, 800, 250, 100, 50, 10, 300, 50, 5)
    test("carbon analysis works", result["total_emissions_kg_co2"] > 0)
    test("carbon has benchmark", "performance" in result["benchmark"])
    test("carbon has equivalents", "trees_to_offset" in result["equivalents"])
    test("carbon has reductions", len(result["reduction_plan"]["strategies"]) > 0)

    print("\n--- Sprint 6: Financial Ledger ---")
    from agri_ai_cli.features.financial_ledger import FinancialLedger
    fl = FinancialLedger(db_path="/tmp/test_financial.json")
    fl.add_expense(50000, "fertilizer", "Urea+DAP", "rice", "kharif")
    fl.add_income(200000, "crop_sale", "Rice 10t", "rice", "kharif")
    summary = fl.get_summary()
    test("finance summary works", summary["net_profit_inr"] == 150000)
    test("finance has ROI", summary["roi_pct"] > 0)

    print("\n--- Sprint 6: NDVI Simulator ---")
    from agri_ai_cli.features.ndvi_simulator import NDVISimulator
    ndvi = NDVISimulator()
    result = ndvi.analyze("rice", "reproductive", 5.0, 0.0, seed=42)
    test("NDVI analysis works", 0 < result["statistics"]["mean_ndvi"] < 1)
    test("NDVI has health", result["health_assessment"]["status"] in
         ("HEALTHY", "MODERATE STRESS", "SEVERE STRESS"))
    test("NDVI has field map", len(result["field_map"]["grid"]) > 0)
    test("NDVI has anomalies", "anomaly_percentage" in result["anomalies"])

    # ── Sprint 7: Multi-Farm + Insurance + AutoTuner ─────────
    print("\n--- Sprint 7: Multi-Farm Manager ---")
    from agri_ai_cli.features.multi_farm import MultiFarmManager
    mfm = MultiFarmManager(db_path="/tmp/test_multi_farm.json")
    farm = mfm.add_farm("Test Farm", 5.0, "Test Village", "alluvial", "rice")
    test("farm creation works", farm["id"] is not None)
    mfm.add_season_record(farm["id"], "rice", "kharif", 20000, 50000, 400000)
    farms = mfm.list_farms()
    test("farm listing works", len(farms) == 1)
    portfolio = mfm.portfolio_summary()
    test("portfolio summary works", portfolio["n_farms"] == 1)

    print("\n--- Sprint 7: Crop Insurance ---")
    from agri_ai_cli.features.crop_insurance import CropInsuranceAdvisor
    ci = CropInsuranceAdvisor()
    result = ci.calculate_premium("rice", 2.5, "kharif")
    test("insurance premium works", result["farmer_premium_inr"] > 0)
    test("insurance has subsidy", result["government_subsidy_inr"] > 0)
    test("insurance has recommendation", result["recommendation"]["verdict"] is not None)
    claim = ci.assess_claim("rice", 2.5, 40, "flood damage", 3000, 5000)
    test("insurance claim assessment", claim["claim_eligible"] is True)

    print("\n--- Sprint 7: ML AutoTuner ---")
    from agri_ai_cli.features.ml_autotuner import MLAutoTuner
    at = MLAutoTuner()
    result = at.tune_classification("soil_classification", 100)
    test("autotuner classification works", len(result["results"]) > 0)
    test("autotuner finds best model", result["best_model"]["accuracy"] > 0)
    reg_result = at.tune_regression("yield_prediction", 100)
    test("autotuner regression works", len(reg_result["results"]) > 0)

    # ── Core: AI Advisor ─────────────────────────────────────
    print("\n--- Core: AI Advisor ---")
    from agri_ai_cli.core.ai_advisor import AIAdvisor
    advisor = AIAdvisor()
    result = advisor.process_query("hello")
    test("advisor greeting", result["intent"] == "greeting")
    result = advisor.process_query("my rice has brown spots")
    test("advisor disease intent", result["intent"] == "disease")
    result = advisor.process_query("what fertilizer for wheat")
    test("advisor fertilizer intent", result["intent"] == "fertilizer")
    result = advisor.process_query("will it rain tomorrow")
    test("advisor weather intent", result["intent"] == "weather")

    # ── Terminal UI ──────────────────────────────────────────
    print("\n--- Utils: Terminal UI ---")
    from agri_ai_cli.utils.terminal_ui import (
        strip_ansi, visible_len, colorize, Color
    )
    test("strip_ansi works", strip_ansi("\033[31mhello\033[0m") == "hello")
    test("visible_len works", visible_len("\033[31mtest\033[0m") == 4)
    colored = colorize("test", Color.RED)
    test("colorize works", len(colored) >= 4)  # may equal 4 if no color support (non-TTY)

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("  ALL TESTS PASSED")
    else:
        print(f"  {failed} TESTS FAILED")
    print("=" * 60)

    # Clean up temp files
    for f in ["/tmp/test_farm_tracker.json", "/tmp/test_financial.json",
              "/tmp/test_multi_farm.json"]:
        if os.path.exists(f):
            os.remove(f)
    import shutil
    if os.path.exists("/tmp/test_exports"):
        shutil.rmtree("/tmp/test_exports")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
