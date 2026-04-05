"""
Sprint 7 - Crop Insurance Advisor.
PMFBY-based crop insurance premium calculation and claim advisory.
"""

import math


class CropInsuranceAdvisor:
    """Crop insurance premium calculator and claim advisory system."""

    # PMFBY premium rates (% of sum insured)
    PREMIUM_RATES = {
        "kharif": {
            "rice": 2.0, "corn": 2.0, "soybean": 2.0, "cotton": 5.0,
            "groundnut": 2.0, "sugarcane": 5.0, "greengram": 2.0,
        },
        "rabi": {
            "wheat": 1.5, "chickpea": 1.5, "lentil": 1.5,
            "mustard": 1.5, "potato": 5.0, "onion": 5.0,
        },
    }

    # Sum insured per hectare (INR)
    SUM_INSURED = {
        "rice": 120000, "wheat": 105000, "corn": 95000, "cotton": 140000,
        "soybean": 80000, "chickpea": 75000, "lentil": 70000, "groundnut": 85000,
        "potato": 150000, "tomato": 180000, "mustard": 72000, "sugarcane": 200000,
        "onion": 120000, "greengram": 55000,
    }

    # Historical loss ratios (simulated)
    LOSS_RATIOS = {
        "rice": 0.35, "wheat": 0.25, "corn": 0.30, "cotton": 0.45,
        "soybean": 0.28, "chickpea": 0.22, "potato": 0.38, "tomato": 0.42,
    }

    COVERED_RISKS = [
        "Natural fire and lightning",
        "Storm, hailstorm, cyclone, typhoon, tempest, hurricane, tornado",
        "Flood, inundation, and landslide",
        "Drought, dry spells",
        "Pests and diseases",
        "Prevented sowing/planting (up to 25% sum insured)",
        "Post-harvest losses (up to 14 days, for specified crops)",
        "Localized calamities (hailstorm, landslide, inundation)",
    ]

    NOT_COVERED = [
        "War and nuclear risks",
        "Malicious damage",
        "Theft or act of enmity",
        "Grazing by domestic/wild animals",
        "Damage by preventable causes",
    ]

    def calculate_premium(self, crop, area_ha, season="kharif"):
        """Calculate insurance premium under PMFBY."""
        season_rates = self.PREMIUM_RATES.get(season, self.PREMIUM_RATES["kharif"])
        premium_rate = season_rates.get(crop, 2.0)
        sum_insured_per_ha = self.SUM_INSURED.get(crop, 100000)
        total_sum_insured = sum_insured_per_ha * area_ha
        farmer_premium = total_sum_insured * premium_rate / 100

        # Actuarial premium (what it would cost without subsidy)
        loss_ratio = self.LOSS_RATIOS.get(crop, 0.30)
        actuarial_rate = max(premium_rate, loss_ratio * 100 * 0.3)
        actuarial_premium = total_sum_insured * actuarial_rate / 100
        subsidy = actuarial_premium - farmer_premium
        subsidy_pct = subsidy / max(actuarial_premium, 1) * 100

        return {
            "crop": crop,
            "season": season,
            "area_ha": area_ha,
            "sum_insured_per_ha": sum_insured_per_ha,
            "total_sum_insured": round(total_sum_insured),
            "premium_rate_pct": premium_rate,
            "farmer_premium_inr": round(farmer_premium),
            "premium_per_ha_inr": round(farmer_premium / max(area_ha, 0.01)),
            "actuarial_premium_inr": round(actuarial_premium),
            "government_subsidy_inr": round(subsidy),
            "subsidy_percentage": round(subsidy_pct, 1),
            "coverage_details": {
                "covered_risks": self.COVERED_RISKS,
                "not_covered": self.NOT_COVERED,
            },
            "claim_process": self._claim_process(),
            "recommendation": self._insurance_recommendation(crop, premium_rate, loss_ratio),
            "benefit_analysis": self._benefit_analysis(farmer_premium, total_sum_insured, loss_ratio),
        }

    def assess_claim(self, crop, area_ha, damage_pct, cause, yield_actual_kg, yield_threshold_kg):
        """Assess potential insurance claim."""
        sum_insured_per_ha = self.SUM_INSURED.get(crop, 100000)
        total_sum_insured = sum_insured_per_ha * area_ha

        # Claim calculation
        if yield_actual_kg >= yield_threshold_kg:
            claim_eligible = False
            claim_amount = 0
            reason = "Actual yield meets or exceeds threshold yield - no claim applicable"
        else:
            claim_eligible = True
            shortfall_pct = (yield_threshold_kg - yield_actual_kg) / max(yield_threshold_kg, 1)
            claim_amount = total_sum_insured * shortfall_pct
            reason = f"Yield shortfall of {round(shortfall_pct * 100, 1)}% from threshold"

        # Documentation needed
        docs = self._required_documents(cause)

        return {
            "crop": crop,
            "area_ha": area_ha,
            "damage_percentage": damage_pct,
            "cause_of_loss": cause,
            "yield_actual_kg": yield_actual_kg,
            "yield_threshold_kg": yield_threshold_kg,
            "claim_eligible": claim_eligible,
            "estimated_claim_inr": round(claim_amount),
            "claim_per_ha_inr": round(claim_amount / max(area_ha, 0.01)),
            "reason": reason,
            "required_documents": docs,
            "claim_timeline": self._claim_timeline(),
            "tips": self._claim_tips(),
        }

    def _claim_process(self):
        return [
            {"step": 1, "action": "Intimate loss within 72 hours to insurance company or bank"},
            {"step": 2, "action": "File claim through crop insurance app or nearest CSC center"},
            {"step": 3, "action": "Loss assessment by joint survey team within 10 days"},
            {"step": 4, "action": "Yield estimation through Crop Cutting Experiments (CCE)"},
            {"step": 5, "action": "Claim settlement within 2 months of CCE data submission"},
        ]

    def _required_documents(self, cause):
        base_docs = [
            "Crop insurance policy document",
            "Land ownership/tenancy proof",
            "Sowing certificate from village officer",
            "Bank passbook copy (for direct benefit transfer)",
            "Aadhaar card copy",
        ]
        if "flood" in cause.lower() or "rain" in cause.lower():
            base_docs.append("Photos of flood/waterlogging damage with date stamp")
            base_docs.append("Revenue department damage certificate")
        if "drought" in cause.lower():
            base_docs.append("District drought declaration notification")
        if "pest" in cause.lower() or "disease" in cause.lower():
            base_docs.append("Plant protection officer's inspection report")
        return base_docs

    def _claim_timeline(self):
        return {
            "loss_intimation": "Within 72 hours of occurrence",
            "document_submission": "Within 15 days of loss intimation",
            "survey_assessment": "Within 10 days of intimation",
            "claim_processing": "Within 30 days of assessment",
            "payment": "Within 60 days of yield data submission",
        }

    def _claim_tips(self):
        return [
            "Report crop loss within 72 hours - delayed reports may be rejected",
            "Take photos with GPS location and timestamp for evidence",
            "Keep all purchase receipts for seeds and inputs as proof",
            "Contact the toll-free helpline 1800-180-1551 for claim assistance",
            "Check claim status on the PMFBY portal or mobile app",
        ]

    def _insurance_recommendation(self, crop, premium_rate, loss_ratio):
        expected_payout = loss_ratio * 100
        if expected_payout > premium_rate * 1.5:
            return {
                "verdict": "HIGHLY RECOMMENDED",
                "reason": f"Historical loss ratio ({round(loss_ratio * 100, 1)}%) significantly "
                         f"exceeds premium rate ({premium_rate}%). Insurance provides strong protection.",
            }
        elif expected_payout > premium_rate:
            return {
                "verdict": "RECOMMENDED",
                "reason": f"Expected losses ({round(loss_ratio * 100, 1)}%) exceed premium cost. "
                         "Good value for risk-averse farmers.",
            }
        return {
            "verdict": "OPTIONAL",
            "reason": "Low historical loss rates make insurance less critical, but still "
                     "provides safety net against catastrophic events.",
        }

    def _benefit_analysis(self, premium, sum_insured, loss_ratio):
        expected_claim = sum_insured * loss_ratio
        net_benefit = expected_claim - premium
        benefit_ratio = expected_claim / max(premium, 1)
        return {
            "premium_paid_inr": round(premium),
            "expected_claim_inr": round(expected_claim),
            "net_expected_benefit_inr": round(net_benefit),
            "benefit_ratio": round(benefit_ratio, 2),
            "max_possible_claim_inr": round(sum_insured),
            "protection_multiple": round(sum_insured / max(premium, 1), 1),
        }

    def get_model_info(self):
        return {
            "name": "Crop Insurance Advisor",
            "scheme": "PMFBY (Pradhan Mantri Fasal Bima Yojana)",
            "crops_supported": len(self.SUM_INSURED),
            "risk_categories": len(self.COVERED_RISKS),
        }
