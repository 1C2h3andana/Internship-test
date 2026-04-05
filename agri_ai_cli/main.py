#!/usr/bin/env python3
"""
AgriAI CLI v3.0 - AI-Powered Crop Disease Detection & Farm Intelligence
A comprehensive agricultural AI assistant with 18+ features across 7 Agile sprints.
Pure Python implementation - no external dependencies required.
"""

import sys
import os
import random

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agri_ai_cli.utils.terminal_ui import (
    print_banner, print_panel, print_table, print_section, print_kv,
    print_list, print_status, confidence_bar, progress_bar, spinner_task,
    divider, bold, success, error, warning, info, Color, colorize
)


def get_models():
    """Lazy-load all models and features."""
    from agri_ai_cli.models.disease_detector import DiseaseDetector
    from agri_ai_cli.models.yield_predictor import YieldPredictor
    from agri_ai_cli.models.soil_analyzer import SoilAnalyzer
    from agri_ai_cli.models.weather_risk import WeatherRiskAssessor
    from agri_ai_cli.features.pest_manager import PestManager
    from agri_ai_cli.features.irrigation_scheduler import IrrigationScheduler
    from agri_ai_cli.features.crop_rotation import CropRotationPlanner
    from agri_ai_cli.features.market_analyzer import MarketAnalyzer
    from agri_ai_cli.features.farm_tracker import FarmTracker
    from agri_ai_cli.features.seasonal_calendar import SeasonalCalendar
    from agri_ai_cli.features.fertilizer_calc import FertilizerCalculator
    from agri_ai_cli.features.export_manager import ExportManager
    from agri_ai_cli.features.carbon_footprint import CarbonFootprintAnalyzer
    from agri_ai_cli.features.financial_ledger import FinancialLedger
    from agri_ai_cli.features.ndvi_simulator import NDVISimulator
    from agri_ai_cli.features.multi_farm import MultiFarmManager
    from agri_ai_cli.features.crop_insurance import CropInsuranceAdvisor
    from agri_ai_cli.features.ml_autotuner import MLAutoTuner
    from agri_ai_cli.core.ai_advisor import AIAdvisor

    return {
        "disease": DiseaseDetector(),
        "yield": YieldPredictor(),
        "soil": SoilAnalyzer(),
        "weather": WeatherRiskAssessor(),
        "pest": PestManager(),
        "irrigation": IrrigationScheduler(),
        "rotation": CropRotationPlanner(),
        "market": MarketAnalyzer(),
        "tracker": FarmTracker(),
        "calendar": SeasonalCalendar(),
        "fertilizer": FertilizerCalculator(),
        "export": ExportManager(),
        "carbon": CarbonFootprintAnalyzer(),
        "finance": FinancialLedger(),
        "ndvi": NDVISimulator(),
        "multi_farm": MultiFarmManager(),
        "insurance": CropInsuranceAdvisor(),
        "autotuner": MLAutoTuner(),
        "advisor": AIAdvisor(),
    }


# ── Command Handlers ────────────────────────────────────────────────

def cmd_demo(models):
    """Run full demonstration of all features."""
    print_banner("AgriAI CLI v3.0 - Full Demo", "AI-Powered Farm Intelligence System")

    # 1. Disease Detection
    print_section("1. Disease Detection", ">>")
    spinner_task("Training disease detection model", 0.5)
    result = models["disease"].predict("my rice plant has diamond shaped lesions with gray centers")
    print_kv("Disease", result["disease_name"])
    print_kv("Crop", result["crop"])
    print_kv("Severity", result["severity_label"])
    print(f"    Confidence:")
    confidence_bar(result["confidence"], label="  ")
    print_kv("Treatment", result["treatment"][0])
    divider()

    # 2. Yield Prediction
    print_section("2. Yield Prediction", ">>")
    spinner_task("Training yield prediction model", 0.5)
    result = models["yield"].predict(28, 1100, 75, 6.2, 150, 40, 80, 8, "rice")
    print_kv("Predicted Yield", f"{result['predicted_yield_kg']} kg/ha ({result['predicted_yield_tons']} tons)")
    print_kv("Revenue Estimate", f"INR {result['revenue_inr']:,.0f}")
    print_kv("Rating", result["yield_rating"])
    divider()

    # 3. Soil Analysis
    print_section("3. Soil Analysis", ">>")
    spinner_task("Training soil classifier", 0.5)
    result = models["soil"].analyze(200, 35, 180, 6.5, 0.8, 45, 28)
    print_kv("Soil Type", result["soil_type_name"])
    print_kv("Health Score", f"{result['health_score']} ({result['health_label']})")
    print_kv("Recommended Crops", ", ".join(result["recommended_crops"]))
    divider()

    # 4. Weather Risk
    print_section("4. Weather Risk Assessment", ">>")
    result = models["weather"].assess(38, 85, 120, 25, 60, "rice")
    print_kv("Risk Level", result["risk_level"])
    print_kv("Overall Score", f"{result['overall_score']:.3f}")
    for risk, val in result["individual_risks"].items():
        confidence_bar(val, label=f"  {risk:12s}")
    divider()

    # 5. Pest Management
    print_section("5. Pest Management (IPM)", ">>")
    spinner_task("Training pest detection model", 0.5)
    result = models["pest"].identify(28, 75, "rice", 8, 100)
    print_kv("Pest Identified", result["pest_name"])
    print_kv("Severity", result["severity"])
    print_kv("Economic Threshold", result["economic_threshold"]["threshold"])
    divider()

    # 6. Irrigation Schedule
    print_section("6. Irrigation Schedule (FAO-56)", ">>")
    result = models["irrigation"].schedule("rice", 32, 65, 3, 8, 45, 2.5, "mid")
    print_kv("ET0", f"{result['et0_mm_day']} mm/day")
    print_kv("Crop ET", f"{result['etc_mm_day']} mm/day")
    print_kv("Next Irrigation", result["recommendation"]["next_irrigation"])
    print_kv("Volume Needed", f"{result['recommendation']['volume_liters_per_ha']:,.0f} L/ha")
    divider()

    # 7. Crop Rotation
    print_section("7. Crop Rotation Plan", ">>")
    result = models["rotation"].plan("rice", "alluvial", 4)
    rotation_str = " -> ".join(result["rotation"])
    print_kv("Plan", rotation_str)
    print_kv("Avg Score", str(result["average_score"]))
    divider()

    # 8. Market Analysis
    print_section("8. Market Price Analysis", ">>")
    result = models["market"].analyze("rice", 24)
    print_kv("Current Price", f"INR {result['statistics']['current']}/kg")
    print_kv("Trend", result["trend"]["direction"])
    print_kv("6-Month Forecast", f"INR {result['forecast'][0]['predicted_price']}/kg")
    print_kv("Signal", result["trading_signal"]["signal"])
    divider()

    # 9. Fertilizer Calculator
    print_section("9. Fertilizer Calculator (ICAR)", ">>")
    result = models["fertilizer"].calculate("rice", 2.5, 180, 30, 150, 6.5, 0.7)
    print_kv("Target Yield", f"{result['target_yield']} tons/ha")
    print_kv("Total Cost", f"INR {result['total_cost_inr']:,}")
    for item in result["fertilizer_plan"]:
        print_kv(f"  {item['fertilizer']}", f"{item['quantity_kg_ha']} kg/ha")
    divider()

    # 10. Carbon Footprint
    print_section("10. Carbon Footprint Analysis", ">>")
    result = models["carbon"].analyze("rice", 2.5, 150, 800, 250, 100, 50, 10, 300, 50, 5)
    print_kv("Total Emissions", f"{result['total_emissions_kg_co2']:,.1f} kg CO2")
    print_kv("Per Hectare", f"{result['emissions_per_ha']:,.1f} kg CO2/ha")
    print_kv("Performance", result["benchmark"]["performance"])
    print_kv("Trees to Offset", str(int(result["equivalents"]["trees_to_offset"])))
    divider()

    # 11. Crop Insurance
    print_section("11. Crop Insurance (PMFBY)", ">>")
    result = models["insurance"].calculate_premium("rice", 2.5, "kharif")
    print_kv("Sum Insured", f"INR {result['total_sum_insured']:,}")
    print_kv("Farmer Premium", f"INR {result['farmer_premium_inr']:,}")
    print_kv("Gov Subsidy", f"INR {result['government_subsidy_inr']:,}")
    print_kv("Recommendation", result["recommendation"]["verdict"])
    divider()

    # 12. NDVI Analysis
    print_section("12. Satellite NDVI Analysis", ">>")
    result = models["ndvi"].analyze("rice", "reproductive", 5.0, 0.1, seed=42)
    print_kv("Mean NDVI", f"{result['statistics']['mean_ndvi']:.4f}")
    print_kv("Health Status", result["health_assessment"]["status"])
    print_kv("Uniformity", f"{result['health_assessment']['uniformity_score']:.2f}")
    # Print field map
    field_map = result["field_map"]
    map_str = "    "
    for row in field_map["grid"]:
        map_str += "".join(cell["symbol"] for cell in row) + "\n    "
    print(f"    Field Map:\n{map_str}")
    divider()

    # 13. Seasonal Calendar
    print_section("13. Seasonal Calendar", ">>")
    result = models["calendar"].get_calendar(7)
    print_kv("Month", result["month_name"])
    print_kv("Season", result["current_season"])
    if result["current_activities"]:
        for act in result["current_activities"][:3]:
            print_kv(f"  {act['crop']}", f"{act['activity']} ({act['timing']})")
    divider()

    # Summary
    print_section("Demo Complete", "**")
    print_table(
        ["Feature", "Sprint", "Algorithm"],
        [
            ["Disease Detection", "Sprint 1", "Random Forest + TF-IDF"],
            ["Yield Prediction", "Sprint 1", "Gradient Boosting"],
            ["Soil Analysis", "Sprint 1", "K-Nearest Neighbors"],
            ["Weather Risk", "Sprint 1", "Multi-Factor Scoring"],
            ["Pest Management", "Sprint 2", "Decision Tree + IPM"],
            ["Irrigation", "Sprint 2", "FAO-56 Penman-Monteith"],
            ["Crop Rotation", "Sprint 3", "Graph Scoring"],
            ["Market Analysis", "Sprint 3", "Exponential Smoothing"],
            ["Farm Tracker", "Sprint 4", "JSON Database"],
            ["Seasonal Calendar", "Sprint 4", "Knowledge Base"],
            ["Fertilizer Calc", "Sprint 5", "ICAR Target Yield"],
            ["Export Manager", "Sprint 5", "JSON/CSV/TXT"],
            ["Carbon Footprint", "Sprint 6", "IPCC Emission Factors"],
            ["Financial Ledger", "Sprint 6", "JSON Database"],
            ["NDVI Simulator", "Sprint 6", "Remote Sensing Sim"],
            ["Multi-Farm", "Sprint 7", "Portfolio Analytics"],
            ["Crop Insurance", "Sprint 7", "PMFBY Calculator"],
            ["ML AutoTuner", "Sprint 7", "Grid Search"],
        ],
        title="All 18 Features"
    )


def cmd_detect(models, symptoms=None):
    """Disease detection command."""
    print_section("Disease Detection", ">>")
    if not symptoms:
        symptoms = "rice plant has diamond shaped lesions with gray center brown borders"
    spinner_task("Analyzing symptoms", 0.5)
    result = models["disease"].predict(symptoms)
    print_kv("Disease", result["disease_name"])
    print_kv("Crop", result["crop"])
    print_kv("Severity", f"{result['severity']:.1f}/10 ({result['severity_label']})")
    print(f"\n    Confidence:")
    confidence_bar(result["confidence"], label="  ")
    print_section("Top Predictions", "  ")
    for pred in result["top_predictions"][:5]:
        confidence_bar(pred["probability"], label=f"  {pred['disease']:30s}")
    print_section("Treatment", "  ")
    print_list(result["treatment"])
    print_section("Prevention", "  ")
    print_list(result["prevention"])


def cmd_predict_yield(models):
    """Yield prediction command."""
    print_section("Yield Prediction", ">>")
    spinner_task("Training yield model", 0.5)
    result = models["yield"].predict(28, 1100, 75, 6.2, 150, 40, 80, 8, "rice")
    print_kv("Crop", result["crop"])
    print_kv("Predicted Yield", f"{result['predicted_yield_kg']:,.1f} kg/ha")
    print_kv("In Tons", f"{result['predicted_yield_tons']} tons/ha")
    print_kv("Revenue", f"INR {result['revenue_inr']:,.0f}")
    print_kv("CI", f"{result['confidence_interval']['lower']:,.0f} - {result['confidence_interval']['upper']:,.0f} kg")
    print_kv("Rating", result["yield_rating"])
    print_section("Factor Analysis", "  ")
    for factor, data in result["factors"].items():
        if "score" in data:
            confidence_bar(data["score"], label=f"  {factor:15s}")
    print_section("Recommendations", "  ")
    print_list(result["recommendations"])


def cmd_soil(models):
    """Soil analysis command."""
    print_section("Soil Analysis", ">>")
    spinner_task("Training soil classifier", 0.5)
    result = models["soil"].analyze(180, 30, 150, 6.5, 0.7, 50, 28)
    print_kv("Soil Type", result["soil_type_name"])
    print_kv("Confidence", f"{result['confidence']:.1%}")
    print_kv("Health Score", f"{result['health_score']} ({result['health_label']})")
    print_section("Nutrient Scores", "  ")
    for nutrient, score in result["nutrient_scores"].items():
        confidence_bar(score, label=f"  {nutrient:18s}")
    print_kv("Recommended Crops", ", ".join(result["recommended_crops"]))
    print_section("Fertilizer Plan", "  ")
    print_list(result["fertilizer_plan"])
    print_section("Soil Improvement", "  ")
    print_list(result["soil_improvement"])


def cmd_weather(models):
    """Weather risk command."""
    print_section("Weather Risk Assessment", ">>")
    result = models["weather"].assess(38, 85, 120, 25, 60, "rice")
    print_kv("Risk Level", result["risk_level"])
    print_kv("Overall Score", f"{result['overall_score']:.3f}")
    print_section("Risk Dimensions", "  ")
    for risk, val in result["individual_risks"].items():
        confidence_bar(val, label=f"  {risk:12s}")
    print_section("Advisories", "  ")
    print_list(result["advisories"])
    action = result["forecast_action"]
    print_kv("Urgency", action["urgency"])
    print_section("Action Plan", "  ")
    print_list(action["actions"])


def cmd_pest(models):
    """Pest management command."""
    print_section("Pest Management (IPM)", ">>")
    spinner_task("Training pest detection model", 0.5)
    result = models["pest"].identify(28, 75, "rice", 8, 100)
    print_kv("Pest Identified", result["pest_name"])
    print_kv("Confidence", f"{result['confidence']:.1%}")
    print_kv("Severity", result["severity"])
    print_kv("Economic Threshold", result["economic_threshold"]["threshold"])
    print_section("IPM Strategy", "  ")
    for category, controls in result["ipm_strategy"].items():
        print(f"\n    {colorize(category.upper(), Color.BOLD)}:")
        print_list(controls, indent=6)
    print_section("Top Threats", "  ")
    for threat in result["top_threats"][:5]:
        confidence_bar(threat["probability"], label=f"  {threat['pest']:25s}")


def cmd_irrigate(models):
    """Irrigation scheduling command."""
    print_section("Irrigation Schedule (FAO-56)", ">>")
    result = models["irrigation"].schedule("rice", 32, 65, 3, 8, 45, 2.5, "mid")
    print_kv("Crop", result["crop"])
    print_kv("Growth Stage", result["growth_stage"])
    print_kv("ET0 (Reference)", f"{result['et0_mm_day']} mm/day")
    print_kv("ETc (Crop)", f"{result['etc_mm_day']} mm/day")
    print_kv("Kc Coefficient", str(result["kc"]))
    print_section("Soil Moisture Status", "  ")
    soil = result["soil_status"]
    print_kv("Current Moisture", f"{soil['current_moisture_pct']}%", indent=6)
    print_kv("Needs Irrigation", str(soil["needs_irrigation"]), indent=6)
    rec = result["recommendation"]
    print_section("Recommendation", "  ")
    print_kv("Next Irrigation", rec["next_irrigation"])
    print_kv("Depth", f"{rec['irrigation_depth_mm']} mm")
    print_kv("Volume", f"{rec['volume_liters_per_ha']:,.0f} L/ha")
    print_kv("Frequency", f"Every {rec['frequency_days']} days")
    print_section("Method Comparison", "  ")
    headers = ["Method", "Duration (hrs)", "Efficiency", "Water Saved"]
    rows = []
    for method, data in result["method_comparison"].items():
        dur = data.get("duration_hours", data.get("depth_mm", "N/A"))
        rows.append([method.upper(), str(dur), f"{data['efficiency_pct']}%", f"{data['water_saved_pct']}%"])
    print_table(headers, rows)


def cmd_rotate(models):
    """Crop rotation command."""
    print_section("Crop Rotation Planner", ">>")
    result = models["rotation"].plan("rice", "alluvial", 6)
    print_kv("Rotation", " -> ".join(result["rotation"]))
    print_kv("Average Score", str(result["average_score"]))
    print_section("Season Schedule", "  ")
    headers = ["Season", "Crop", "Months"]
    rows = [[s["season"].upper(), s["crop"], s["months"]] for s in result["seasons"]]
    print_table(headers, rows)
    health = result["soil_health_impact"]
    print_section("Soil Health Impact", "  ")
    print_kv("Diversity Score", str(health["diversity_score"]))
    print_kv("N-Fixing Crops", str(health["n_fixing_crops"]))
    print_kv("Overall Health", str(health["overall_health"]))
    print_section("Recommendations", "  ")
    print_list(result["recommendations"])


def cmd_market(models):
    """Market analysis command."""
    print_section("Market Price Analysis", ">>")
    result = models["market"].analyze("rice", 24)
    stats = result["statistics"]
    print_kv("Current Price", f"INR {stats['current']}/kg")
    print_kv("Average", f"INR {stats['mean']}/kg")
    print_kv("Range", f"INR {stats['min']} - {stats['max']}/kg")
    trend = result["trend"]
    print_kv("Trend", f"{trend['direction']} ({trend['monthly_change_pct']}%/month)")
    print_section("Price Forecast (6 months)", "  ")
    headers = ["Month", "Price (INR)", "Confidence", "Range"]
    rows = [[f"+{f['month_ahead']}m", f"{f['predicted_price']:.2f}",
             f"{f['confidence']:.0%}", f"{f['range_low']:.2f}-{f['range_high']:.2f}"]
            for f in result["forecast"]]
    print_table(headers, rows)
    vol = result["volatility"]
    print_kv("Volatility", f"{vol['annualized_volatility']}% ({vol['risk_level']})")
    signal = result["trading_signal"]
    print_kv("Signal", f"{signal['signal']} - {signal['reason']}")


def cmd_fertilizer(models):
    """Fertilizer calculator command."""
    print_section("Fertilizer Calculator (ICAR)", ">>")
    result = models["fertilizer"].calculate("rice", 2.5, 180, 30, 150, 6.5, 0.7)
    print_kv("Crop", result["crop"])
    print_kv("Area", f"{result['area_ha']} ha")
    print_kv("Target Yield", f"{result['target_yield']} tons/ha")
    print_section("Nutrient Requirement (kg/ha)", "  ")
    for k, v in result["nutrient_requirement"].items():
        print_kv(f"  {k.replace('_kg_ha','')}", str(v))
    print_section("Fertilizer Plan", "  ")
    headers = ["Fertilizer", "Qty (kg/ha)", "Total (kg)", "Cost (INR)"]
    rows = [[f["fertilizer"], str(f["quantity_kg_ha"]), str(f["quantity_total_kg"]),
             f"{f['total_cost_inr']:,}"] for f in result["fertilizer_plan"]]
    print_table(headers, rows)
    print_kv("Total Cost", f"INR {result['total_cost_inr']:,}")
    print_section("Application Schedule", "  ")
    for s in result["application_schedule"]:
        print_kv(s["timing"], s["application"])


def cmd_carbon(models):
    """Carbon footprint command."""
    print_section("Carbon Footprint Analysis", ">>")
    result = models["carbon"].analyze("rice", 2.5, 150, 800, 250, 100, 50, 10, 300, 50, 5)
    print_kv("Total Emissions", f"{result['total_emissions_kg_co2']:,.1f} kg CO2")
    print_kv("Per Hectare", f"{result['emissions_per_ha']:,.1f} kg CO2/ha")
    print_kv("Per Ton Produce", f"{result['emissions_per_ton']:,.1f} kg CO2/ton")
    bench = result["benchmark"]
    print_kv("Performance", f"{bench['performance']} ({bench['vs_average_pct']:+.1f}% vs national avg)")
    print_section("Emission Sources", "  ")
    for source, kg in result["emission_sources"].items():
        if kg > 0:
            confidence_bar(kg, max_val=max(result["emission_sources"].values()),
                          label=f"  {source:20s}")
    print_section("Equivalents", "  ")
    eq = result["equivalents"]
    print_kv("Car Driving Days", str(int(eq["cars_equivalent_days"])))
    print_kv("Trees to Offset", str(int(eq["trees_to_offset"])))
    print_kv("Flight Equivalents", str(eq["flights_equivalent"]))
    if result["reduction_plan"]["strategies"]:
        print_section("Reduction Strategies", "  ")
        for s in result["reduction_plan"]["strategies"][:4]:
            print_kv(f"  {s['strategy'][:50]}", f"-{s['reduction_pct']}% ({s['difficulty']})")


def cmd_insurance(models):
    """Insurance calculator command."""
    print_section("Crop Insurance (PMFBY)", ">>")
    result = models["insurance"].calculate_premium("rice", 2.5, "kharif")
    print_kv("Crop", f"{result['crop']} ({result['season']})")
    print_kv("Sum Insured", f"INR {result['total_sum_insured']:,}")
    print_kv("Premium Rate", f"{result['premium_rate_pct']}%")
    print_kv("Farmer Premium", f"INR {result['farmer_premium_inr']:,}")
    print_kv("Gov Subsidy", f"INR {result['government_subsidy_inr']:,} ({result['subsidy_percentage']}%)")
    rec = result["recommendation"]
    print_kv("Recommendation", f"{rec['verdict']}")
    print(f"      {rec['reason']}")
    benefit = result["benefit_analysis"]
    print_section("Benefit Analysis", "  ")
    print_kv("Protection Multiple", f"{benefit['protection_multiple']}x")
    print_kv("Max Claim Possible", f"INR {benefit['max_possible_claim_inr']:,}")
    print_section("Claim Process", "  ")
    for step in result["claim_process"]:
        print_kv(f"  Step {step['step']}", step["action"])


def cmd_ndvi(models):
    """NDVI satellite analysis command."""
    print_section("Satellite NDVI Analysis", ">>")
    result = models["ndvi"].analyze("rice", "reproductive", 5.0, 0.1, seed=42)
    stats = result["statistics"]
    print_kv("Mean NDVI", f"{stats['mean_ndvi']:.4f}")
    print_kv("Std Dev", f"{stats['std_dev']:.4f}")
    print_kv("Range", f"{stats['min_ndvi']:.4f} - {stats['max_ndvi']:.4f}")
    health = result["health_assessment"]
    print_kv("Health Status", health["status"])
    print_kv("Health Score", str(health["health_score"]))
    print_kv("Uniformity", str(health["uniformity_score"]))
    print_section("Field Map", "  ")
    for row in result["field_map"]["grid"]:
        line = "    " + " ".join(cell["symbol"] for cell in row)
        print(line)
    print(f"    Legend: # Dense  + Good  . Moderate  - Sparse")
    anom = result["anomalies"]
    print_section("Anomaly Detection", "  ")
    print_kv("Anomaly Coverage", f"{anom['anomaly_percentage']}%")
    print_kv("Severity", anom["severity"])
    print_section("Recommendations", "  ")
    print_list(result["recommendations"])


def cmd_finance(models):
    """Financial ledger command."""
    print_section("Farm Financial Ledger", ">>")
    models["finance"].add_demo_data()
    summary = models["finance"].get_summary()
    print_kv("Total Income", f"INR {summary['total_income_inr']:,}")
    print_kv("Total Expenses", f"INR {summary['total_expenses_inr']:,}")
    print_kv("Net Profit", f"INR {summary['net_profit_inr']:,}")
    print_kv("ROI", f"{summary['roi_pct']}%")
    print_kv("Status", summary["profit_status"])
    if summary.get("expense_analysis"):
        print_section("Expense Breakdown", "  ")
        headers = ["Category", "Amount (INR)", "%", "Assessment"]
        rows = [[e["category"], f"{e['amount_inr']:,}", f"{e['percentage']}%",
                 e["assessment"][:40]] for e in summary["expense_analysis"][:6]]
        print_table(headers, rows)
    # Crop profitability
    profitability = models["finance"].get_crop_profitability()
    if profitability:
        print_section("Crop Profitability Ranking", "  ")
        headers = ["Crop", "Income", "Expenses", "Profit", "ROI"]
        rows = [[p["crop"], f"{p['income']:,}", f"{p['expenses']:,}",
                 f"{p['profit']:,}", f"{p['roi_pct']}%"] for p in profitability]
        print_table(headers, rows)


def cmd_multi_farm(models):
    """Multi-farm management command."""
    print_section("Multi-Farm Management", ">>")
    models["multi_farm"].add_demo_data()
    farms = models["multi_farm"].list_farms()
    print_section("Registered Farms", "  ")
    headers = ["Name", "Area (ha)", "Location", "Primary Crop", "Seasons"]
    rows = [[f["name"], str(f["area_ha"]), f["location"],
             f["primary_crop"], str(f["n_seasons"])] for f in farms]
    print_table(headers, rows)

    comparison = models["multi_farm"].compare_farms()
    if comparison:
        print_section("Farm Performance Comparison", "  ")
        headers = ["Farm", "Yield/ha", "Profit/ha", "ROI%"]
        rows = [[c["farm_name"], f"{c['avg_yield_per_ha']:,.0f}",
                 f"{c['avg_profit_per_ha']:,.0f}", f"{c['roi_pct']}%"]
                for c in comparison]
        print_table(headers, rows)

    portfolio = models["multi_farm"].portfolio_summary()
    print_section("Portfolio Summary", "  ")
    print_kv("Total Farms", str(portfolio.get("n_farms", 0)))
    print_kv("Total Area", f"{portfolio.get('total_area_ha', 0)} ha")
    print_kv("Total Profit", f"INR {portfolio.get('total_profit_inr', 0):,}")
    print_kv("Portfolio ROI", f"{portfolio.get('portfolio_roi_pct', 0)}%")


def cmd_autotuner(models):
    """ML AutoTuner command."""
    print_section("ML AutoTuner - Model Comparison", ">>")

    print(f"\n  Running classification benchmark...")
    spinner_task("Tuning disease detection models", 1.0)
    class_result = models["autotuner"].tune_classification("disease_detection", 200)
    print_kv("Task", class_result["task"])
    print_kv("Best Model", class_result["best_model"]["name"])
    print_kv("Best Accuracy", f"{class_result['best_model']['accuracy']:.4f}")
    headers = ["Model", "Accuracy", "Train Acc", "Overfit", "Time (ms)"]
    rows = [[r["model"], f"{r['accuracy']:.4f}", f"{r['train_accuracy']:.4f}",
             f"{r['overfit_gap']:.4f}", str(r["train_time_ms"])]
            for r in class_result["results"]]
    print_table(headers, rows, title="Classification Results")

    print(f"\n  Running regression benchmark...")
    spinner_task("Tuning yield prediction models", 1.0)
    reg_result = models["autotuner"].tune_regression("yield_prediction", 200)
    print_kv("Best Model", reg_result["best_model"]["name"])
    print_kv("Best R2", f"{reg_result['best_model']['r2_score']:.4f}")
    headers = ["Model", "R2", "RMSE", "MAE", "Overfit", "Time (ms)"]
    rows = [[r["model"], f"{r['r2_score']:.4f}", f"{r['rmse']:.0f}",
             f"{r['mae']:.0f}", f"{r['overfit_gap']:.4f}", str(r["train_time_ms"])]
            for r in reg_result["results"]]
    print_table(headers, rows, title="Regression Results")

    print_section("Recommendations", "  ")
    for rec in class_result.get("recommendations", []):
        print(f"    - {rec}")
    for rec in reg_result.get("recommendations", []):
        print(f"    - {rec}")


def cmd_train(models):
    """Train all ML models."""
    print_section("Training All ML Models", ">>")
    trainable = [
        ("Disease Detector", models["disease"]),
        ("Yield Predictor", models["yield"]),
        ("Soil Analyzer", models["soil"]),
        ("Pest Manager", models["pest"]),
    ]
    results = []
    for name, model in trainable:
        spinner_task(f"Training {name}", 0.5)
        info = model.train()
        metric = info.get("accuracy", info.get("r2_score", "N/A"))
        if isinstance(metric, float):
            metric = f"{metric:.3f}"
        results.append([name, str(metric), "PASS"])
        print(f"    {success('*')} {name}: {metric}")

    print_table(["Model", "Score", "Status"], results, title="Training Summary")


def cmd_dashboard(models):
    """Show dashboard with all available commands."""
    print_banner("AgriAI CLI v3.0", "AI-Powered Crop Disease Detection & Farm Intelligence")

    headers = ["Command", "Description", "Sprint"]
    rows = [
        ["demo", "Full feature demonstration", "All"],
        ["detect", "Disease detection from symptoms", "1"],
        ["predict-yield", "Crop yield prediction", "1"],
        ["analyze-soil", "Soil analysis & crop recommendation", "1"],
        ["weather", "Weather risk assessment", "1"],
        ["pest", "Pest identification & IPM strategy", "2"],
        ["irrigate", "Irrigation scheduling (FAO-56)", "2"],
        ["rotate", "Crop rotation planning", "3"],
        ["market", "Market price analysis & forecast", "3"],
        ["fertilizer", "Fertilizer calculator (ICAR)", "5"],
        ["carbon", "Carbon footprint analysis", "6"],
        ["finance", "Farm financial ledger", "6"],
        ["ndvi", "Satellite NDVI crop health", "6"],
        ["insurance", "Crop insurance calculator", "7"],
        ["multi-farm", "Multi-farm management", "7"],
        ["autotuner", "ML model comparison & tuning", "7"],
        ["train", "Train/retrain all ML models", "1"],
        ["dashboard", "Show this command list", "-"],
    ]
    print_table(headers, rows, title="Available Commands (18)")

    print(f"\n  Usage: python main.py <command>")
    print(f"  Example: python main.py demo\n")


# ── Main Entry Point ────────────────────────────────────────────────

COMMANDS = {
    "demo": cmd_demo,
    "detect": cmd_detect,
    "predict-yield": cmd_predict_yield,
    "analyze-soil": cmd_soil,
    "weather": cmd_weather,
    "pest": cmd_pest,
    "irrigate": cmd_irrigate,
    "rotate": cmd_rotate,
    "market": cmd_market,
    "fertilizer": cmd_fertilizer,
    "carbon": cmd_carbon,
    "insurance": cmd_insurance,
    "ndvi": cmd_ndvi,
    "finance": cmd_finance,
    "multi-farm": cmd_multi_farm,
    "autotuner": cmd_autotuner,
    "train": cmd_train,
    "dashboard": cmd_dashboard,
}


def main():
    args = sys.argv[1:]
    command = args[0] if args else "dashboard"

    if command in ("-h", "--help", "help"):
        command = "dashboard"

    if command not in COMMANDS:
        print(error(f"  Unknown command: {command}"))
        print(f"  Run 'python main.py dashboard' to see available commands.\n")
        sys.exit(1)

    models = get_models()
    handler = COMMANDS[command]

    if command == "detect" and len(args) > 1:
        handler(models, " ".join(args[1:]))
    else:
        handler(models)


if __name__ == "__main__":
    main()
