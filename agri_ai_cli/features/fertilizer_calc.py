"""
Sprint 5 - Fertilizer Calculator.
ICAR target yield method for NPK fertilizer recommendation.
"""

import math


class FertilizerCalculator:
    """Precision fertilizer calculation using ICAR target yield approach."""

    # Nutrient content of common fertilizers (%)
    FERTILIZERS = {
        "urea": {"N": 46, "P": 0, "K": 0, "cost_per_kg": 5.36},
        "dap": {"N": 18, "P": 46, "K": 0, "cost_per_kg": 27.0},
        "mop": {"N": 0, "P": 0, "K": 60, "cost_per_kg": 16.5},
        "ssp": {"N": 0, "P": 16, "K": 0, "cost_per_kg": 8.0},
        "npk_10_26_26": {"N": 10, "P": 26, "K": 26, "cost_per_kg": 24.0},
        "ammonium_sulphate": {"N": 20.5, "P": 0, "K": 0, "cost_per_kg": 9.0},
        "potash_sulphate": {"N": 0, "P": 0, "K": 50, "cost_per_kg": 22.0},
        "neem_coated_urea": {"N": 46, "P": 0, "K": 0, "cost_per_kg": 5.63},
        "vermicompost": {"N": 1.5, "P": 0.8, "K": 1.0, "cost_per_kg": 8.0},
        "fym": {"N": 0.5, "P": 0.3, "K": 0.5, "cost_per_kg": 2.0},
    }

    # Crop nutrient requirements (kg/ha for target yield in tons)
    CROP_REQUIREMENTS = {
        "rice": {"N": 80, "P": 40, "K": 40, "target_yield_tons": 5.0},
        "wheat": {"N": 120, "P": 60, "K": 40, "target_yield_tons": 4.5},
        "corn": {"N": 150, "P": 70, "K": 60, "target_yield_tons": 8.0},
        "tomato": {"N": 120, "P": 80, "K": 100, "target_yield_tons": 35.0},
        "cotton": {"N": 100, "P": 50, "K": 50, "target_yield_tons": 2.5},
        "soybean": {"N": 30, "P": 60, "K": 40, "target_yield_tons": 3.0},
        "potato": {"N": 180, "P": 80, "K": 150, "target_yield_tons": 30.0},
        "sugarcane": {"N": 250, "P": 85, "K": 120, "target_yield_tons": 80.0},
        "chickpea": {"N": 20, "P": 40, "K": 20, "target_yield_tons": 2.0},
        "mustard": {"N": 80, "P": 40, "K": 20, "target_yield_tons": 1.8},
    }

    # Nutrient use efficiency factors
    EFFICIENCY = {
        "soil_N": 0.5,   # 50% of soil N is available
        "soil_P": 0.3,   # 30% of soil P is available
        "soil_K": 0.5,   # 50% of soil K is available
        "fert_N": 0.50,  # 50% fertilizer N efficiency
        "fert_P": 0.25,  # 25% fertilizer P efficiency
        "fert_K": 0.70,  # 70% fertilizer K efficiency
    }

    def calculate(self, crop, area_ha, soil_n, soil_p, soil_k, soil_ph=6.5,
                  organic_carbon=0.5, target_yield_factor=1.0):
        """Calculate fertilizer requirements using target yield method."""
        reqs = self.CROP_REQUIREMENTS.get(crop, self.CROP_REQUIREMENTS["rice"])

        # Adjust target based on factor
        target_n = reqs["N"] * target_yield_factor
        target_p = reqs["P"] * target_yield_factor
        target_k = reqs["K"] * target_yield_factor

        # Subtract soil available nutrients
        available_n = soil_n * self.EFFICIENCY["soil_N"]
        available_p = soil_p * self.EFFICIENCY["soil_P"]
        available_k = soil_k * self.EFFICIENCY["soil_K"]

        # Net nutrient requirement
        net_n = max(0, target_n - available_n)
        net_p = max(0, target_p - available_p)
        net_k = max(0, target_k - available_k)

        # Adjust for fertilizer efficiency
        fert_n = net_n / self.EFFICIENCY["fert_N"]
        fert_p = net_p / self.EFFICIENCY["fert_P"]
        fert_k = net_k / self.EFFICIENCY["fert_K"]

        # pH adjustment
        ph_factor = self._ph_adjustment(soil_ph)
        fert_p *= ph_factor["p_factor"]

        # Calculate fertilizer quantities
        fert_plan = self._optimize_fertilizer_mix(fert_n, fert_p, fert_k)

        # Scale to area
        for item in fert_plan:
            item["quantity_total_kg"] = round(item["quantity_kg_ha"] * area_ha, 1)
            item["total_cost_inr"] = round(item["cost_per_ha"] * area_ha, 0)

        total_cost = sum(item["total_cost_inr"] for item in fert_plan)

        return {
            "crop": crop,
            "area_ha": area_ha,
            "target_yield": round(reqs["target_yield_tons"] * target_yield_factor, 1),
            "nutrient_requirement": {
                "nitrogen_kg_ha": round(target_n, 1),
                "phosphorus_kg_ha": round(target_p, 1),
                "potassium_kg_ha": round(target_k, 1),
            },
            "soil_available": {
                "nitrogen_kg_ha": round(available_n, 1),
                "phosphorus_kg_ha": round(available_p, 1),
                "potassium_kg_ha": round(available_k, 1),
            },
            "net_requirement": {
                "nitrogen_kg_ha": round(fert_n, 1),
                "phosphorus_kg_ha": round(fert_p, 1),
                "potassium_kg_ha": round(fert_k, 1),
            },
            "fertilizer_plan": fert_plan,
            "total_cost_inr": round(total_cost),
            "cost_per_ha_inr": round(total_cost / max(area_ha, 0.01)),
            "ph_adjustment": ph_factor,
            "application_schedule": self._application_schedule(crop, fert_plan),
            "organic_recommendations": self._organic_recs(organic_carbon),
        }

    def _ph_adjustment(self, ph):
        if ph < 5.5:
            return {
                "status": "ACIDIC",
                "p_factor": 1.3,
                "recommendation": f"Apply 2-4 tons/ha lime. pH {ph} reduces P availability by 30%.",
                "lime_kg_ha": round((6.5 - ph) * 1000, 0),
            }
        elif ph > 8.0:
            return {
                "status": "ALKALINE",
                "p_factor": 1.2,
                "recommendation": f"Apply 1-2 tons/ha gypsum. pH {ph} reduces micronutrient availability.",
                "gypsum_kg_ha": round((ph - 7.0) * 800, 0),
            }
        return {
            "status": "OPTIMAL",
            "p_factor": 1.0,
            "recommendation": f"pH {ph} is in the optimal range for most crops.",
        }

    def _optimize_fertilizer_mix(self, n_need, p_need, k_need):
        """Calculate optimal fertilizer product quantities."""
        plan = []

        # Use DAP for phosphorus (also provides some N)
        if p_need > 0:
            dap_kg = p_need / (self.FERTILIZERS["dap"]["P"] / 100)
            dap_n = dap_kg * self.FERTILIZERS["dap"]["N"] / 100
            n_need = max(0, n_need - dap_n)
            plan.append({
                "fertilizer": "DAP (Di-Ammonium Phosphate)",
                "quantity_kg_ha": round(dap_kg, 1),
                "provides": f"P: {round(p_need, 1)} kg, N: {round(dap_n, 1)} kg",
                "cost_per_ha": round(dap_kg * self.FERTILIZERS["dap"]["cost_per_kg"]),
            })

        # Use Urea for remaining nitrogen
        if n_need > 0:
            urea_kg = n_need / (self.FERTILIZERS["urea"]["N"] / 100)
            plan.append({
                "fertilizer": "Urea",
                "quantity_kg_ha": round(urea_kg, 1),
                "provides": f"N: {round(n_need, 1)} kg",
                "cost_per_ha": round(urea_kg * self.FERTILIZERS["urea"]["cost_per_kg"]),
            })

        # Use MOP for potassium
        if k_need > 0:
            mop_kg = k_need / (self.FERTILIZERS["mop"]["K"] / 100)
            plan.append({
                "fertilizer": "MOP (Muriate of Potash)",
                "quantity_kg_ha": round(mop_kg, 1),
                "provides": f"K: {round(k_need, 1)} kg",
                "cost_per_ha": round(mop_kg * self.FERTILIZERS["mop"]["cost_per_kg"]),
            })

        return plan

    def _application_schedule(self, crop, plan):
        """Generate split application schedule."""
        return [
            {
                "timing": "Basal (at sowing)",
                "application": "Full DAP + Full MOP + 50% Urea",
                "note": "Mix with soil before sowing/transplanting",
            },
            {
                "timing": "First top dressing (25-30 DAS)",
                "application": "25% Urea",
                "note": "Apply near plant base and irrigate",
            },
            {
                "timing": "Second top dressing (50-60 DAS)",
                "application": "25% Urea",
                "note": "Apply at active tillering/vegetative growth",
            },
        ]

    def _organic_recs(self, organic_carbon):
        recs = []
        if organic_carbon < 0.4:
            recs.append("CRITICAL: Very low organic carbon. Apply FYM at 10-15 tons/ha")
            recs.append("Start vermicomposting for regular organic matter supply")
        elif organic_carbon < 0.75:
            recs.append("Apply FYM or compost at 5-8 tons/ha to improve soil organic matter")
        else:
            recs.append("Organic carbon levels are adequate. Maintain through residue incorporation.")
        recs.append("Consider green manuring with dhaincha/sunhemp before main crop")
        return recs

    def get_model_info(self):
        return {
            "name": "Fertilizer Calculator",
            "algorithm": "ICAR Target Yield Method",
            "fertilizers_db": len(self.FERTILIZERS),
            "crops_supported": len(self.CROP_REQUIREMENTS),
        }
