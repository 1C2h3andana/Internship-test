"""
Sprint 6 - Carbon Footprint Analyzer.
Calculates greenhouse gas emissions from farming activities and suggests reduction strategies.
"""

import math
from agri_ai_cli.utils.data_generator import CARBON_FACTORS


class CarbonFootprintAnalyzer:
    """Agricultural carbon footprint analysis and reduction advisory."""

    EMISSION_BENCHMARKS = {
        "rice": {"avg_kg_co2_per_ha": 5500, "best_kg_co2_per_ha": 3000},
        "wheat": {"avg_kg_co2_per_ha": 2500, "best_kg_co2_per_ha": 1500},
        "corn": {"avg_kg_co2_per_ha": 3000, "best_kg_co2_per_ha": 1800},
        "cotton": {"avg_kg_co2_per_ha": 3500, "best_kg_co2_per_ha": 2000},
        "soybean": {"avg_kg_co2_per_ha": 1800, "best_kg_co2_per_ha": 1000},
        "tomato": {"avg_kg_co2_per_ha": 4000, "best_kg_co2_per_ha": 2500},
        "potato": {"avg_kg_co2_per_ha": 3200, "best_kg_co2_per_ha": 1800},
    }

    REDUCTION_STRATEGIES = {
        "fertilizer": [
            {"strategy": "Neem-coated urea instead of regular urea", "reduction_pct": 15,
             "difficulty": "Easy", "cost_impact": "Minimal"},
            {"strategy": "Split nitrogen application (3-4 doses)", "reduction_pct": 10,
             "difficulty": "Easy", "cost_impact": "Labor increase"},
            {"strategy": "Leaf color chart for N management", "reduction_pct": 20,
             "difficulty": "Medium", "cost_impact": "Saves fertilizer cost"},
        ],
        "energy": [
            {"strategy": "Solar-powered irrigation pumps", "reduction_pct": 80,
             "difficulty": "High", "cost_impact": "High initial, zero running"},
            {"strategy": "Drip irrigation (vs flood)", "reduction_pct": 30,
             "difficulty": "Medium", "cost_impact": "Moderate initial investment"},
            {"strategy": "Laser land leveling", "reduction_pct": 15,
             "difficulty": "Medium", "cost_impact": "One-time cost"},
        ],
        "methane": [
            {"strategy": "Alternate Wetting and Drying (AWD) in rice", "reduction_pct": 30,
             "difficulty": "Medium", "cost_impact": "Saves water cost"},
            {"strategy": "Direct Seeded Rice (DSR)", "reduction_pct": 40,
             "difficulty": "High", "cost_impact": "Saves labor, may need herbicide"},
            {"strategy": "System of Rice Intensification (SRI)", "reduction_pct": 25,
             "difficulty": "High", "cost_impact": "Lower seed cost, more labor"},
        ],
        "residue": [
            {"strategy": "In-situ residue incorporation", "reduction_pct": 90,
             "difficulty": "Medium", "cost_impact": "Needs Happy Seeder machine"},
            {"strategy": "Composting crop residues", "reduction_pct": 70,
             "difficulty": "Medium", "cost_impact": "Labor intensive"},
            {"strategy": "Biochar production from residues", "reduction_pct": 85,
             "difficulty": "High", "cost_impact": "Needs pyrolysis unit"},
        ],
    }

    def analyze(self, crop, area_ha, diesel_liters, electricity_kwh,
                urea_kg, dap_kg, mop_kg, pesticide_kg,
                irrigation_hours, transport_km=50, yield_tons=5):
        """Calculate comprehensive carbon footprint for farming activity."""

        # Calculate emissions by source
        emissions = {
            "diesel": round(diesel_liters * CARBON_FACTORS["diesel_per_liter"], 1),
            "electricity": round(electricity_kwh * CARBON_FACTORS["electricity_per_kwh"], 1),
            "urea": round(urea_kg * CARBON_FACTORS["urea_per_kg"], 1),
            "dap": round(dap_kg * CARBON_FACTORS["dap_per_kg"], 1),
            "mop": round(mop_kg * CARBON_FACTORS["mop_per_kg"], 1),
            "pesticide": round(pesticide_kg * CARBON_FACTORS["pesticide_per_kg"], 1),
            "irrigation": round(irrigation_hours * CARBON_FACTORS["irrigation_pump_per_hour"], 1),
            "transport": round(transport_km * yield_tons * CARBON_FACTORS["transport_per_km_ton"], 1),
        }

        # Methane from rice paddies
        if crop == "rice":
            emissions["methane_rice"] = round(CARBON_FACTORS["rice_methane_per_ha"] * area_ha, 1)

        total = sum(emissions.values())
        per_ha = total / max(area_ha, 0.01)
        per_ton = total / max(yield_tons, 0.01)

        # Benchmark comparison
        benchmark = self.EMISSION_BENCHMARKS.get(crop, {"avg_kg_co2_per_ha": 3000, "best_kg_co2_per_ha": 1800})
        performance_ratio = per_ha / benchmark["avg_kg_co2_per_ha"]

        if performance_ratio <= 0.7:
            rating = "EXCELLENT"
        elif performance_ratio <= 1.0:
            rating = "GOOD"
        elif performance_ratio <= 1.3:
            rating = "AVERAGE"
        else:
            rating = "HIGH EMISSIONS"

        # Emission breakdown by category
        categories = self._categorize_emissions(emissions)

        # Reduction opportunities
        reductions = self._identify_reductions(emissions, crop, categories)

        return {
            "crop": crop,
            "area_ha": area_ha,
            "total_emissions_kg_co2": round(total, 1),
            "emissions_per_ha": round(per_ha, 1),
            "emissions_per_ton": round(per_ton, 1),
            "emission_sources": emissions,
            "categories": categories,
            "benchmark": {
                "national_avg": benchmark["avg_kg_co2_per_ha"],
                "best_practice": benchmark["best_kg_co2_per_ha"],
                "your_value": round(per_ha, 1),
                "performance": rating,
                "vs_average_pct": round((performance_ratio - 1) * 100, 1),
            },
            "reduction_plan": reductions,
            "carbon_credits": self._carbon_credit_potential(total, benchmark, area_ha),
            "equivalents": {
                "cars_equivalent_days": round(total / 12.0, 0),  # avg car = 12 kg CO2/day
                "trees_to_offset": round(total / 22.0, 0),  # avg tree absorbs 22 kg CO2/year
                "flights_equivalent": round(total / 255.0, 1),  # Delhi-Mumbai flight ~255 kg CO2
            },
        }

    def _categorize_emissions(self, emissions):
        categories = {
            "fertilizer_production": emissions.get("urea", 0) + emissions.get("dap", 0) + emissions.get("mop", 0),
            "field_operations": emissions.get("diesel", 0),
            "irrigation_energy": emissions.get("electricity", 0) + emissions.get("irrigation", 0),
            "crop_protection": emissions.get("pesticide", 0),
            "methane": emissions.get("methane_rice", 0),
            "transport": emissions.get("transport", 0),
        }
        total = sum(categories.values())
        return {k: {"kg_co2": round(v, 1), "pct": round(v / max(total, 1) * 100, 1)}
                for k, v in categories.items()}

    def _identify_reductions(self, emissions, crop, categories):
        reductions = []
        # Prioritize largest emission sources
        sorted_cats = sorted(categories.items(), key=lambda x: x[1]["kg_co2"], reverse=True)

        for cat_name, cat_data in sorted_cats[:3]:
            if cat_name == "fertilizer_production" and cat_data["kg_co2"] > 0:
                reductions.extend(self.REDUCTION_STRATEGIES["fertilizer"][:2])
            elif cat_name == "irrigation_energy" and cat_data["kg_co2"] > 0:
                reductions.extend(self.REDUCTION_STRATEGIES["energy"][:2])
            elif cat_name == "methane" and cat_data["kg_co2"] > 0:
                reductions.extend(self.REDUCTION_STRATEGIES["methane"][:2])

        potential_reduction = sum(r["reduction_pct"] for r in reductions) / max(len(reductions), 1)
        return {
            "strategies": reductions,
            "potential_reduction_pct": round(potential_reduction, 1),
        }

    def _carbon_credit_potential(self, total, benchmark, area_ha):
        best = benchmark["best_kg_co2_per_ha"] * area_ha
        if total > best:
            potential_reduction = total - best
            credit_value = potential_reduction / 1000 * 15  # ~$15/ton CO2 in voluntary market
            return {
                "eligible": True,
                "potential_reduction_kg": round(potential_reduction, 0),
                "credit_tons_co2": round(potential_reduction / 1000, 2),
                "estimated_value_usd": round(credit_value, 2),
                "estimated_value_inr": round(credit_value * 83, 0),
            }
        return {"eligible": False, "message": "Already at best-practice emission levels"}

    def get_model_info(self):
        return {
            "name": "Carbon Footprint Analyzer",
            "algorithm": "IPCC Emission Factor Method",
            "emission_sources": len(CARBON_FACTORS),
            "reduction_strategies": sum(len(v) for v in self.REDUCTION_STRATEGIES.values()),
        }
