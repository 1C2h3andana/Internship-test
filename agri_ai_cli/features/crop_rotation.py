"""
Sprint 3 - Crop Rotation Planner.
Multi-objective optimization for sustainable crop rotation planning.
"""

import random


class CropRotationPlanner:
    """Intelligent crop rotation planning based on soil health and economics."""

    CROP_FAMILIES = {
        "cereals": ["rice", "wheat", "corn", "barley", "millet"],
        "legumes": ["soybean", "chickpea", "lentil", "groundnut", "greengram"],
        "vegetables": ["tomato", "potato", "onion", "cabbage", "brinjal"],
        "oilseeds": ["mustard", "sunflower", "sesame"],
        "fiber": ["cotton", "jute"],
        "cash": ["sugarcane", "tobacco"],
    }

    CROP_NUTRIENTS = {
        "rice": {"n_demand": "high", "fixes_n": False, "residue": "medium"},
        "wheat": {"n_demand": "high", "fixes_n": False, "residue": "high"},
        "corn": {"n_demand": "very_high", "fixes_n": False, "residue": "high"},
        "soybean": {"n_demand": "low", "fixes_n": True, "residue": "medium"},
        "chickpea": {"n_demand": "low", "fixes_n": True, "residue": "low"},
        "lentil": {"n_demand": "low", "fixes_n": True, "residue": "low"},
        "groundnut": {"n_demand": "low", "fixes_n": True, "residue": "medium"},
        "greengram": {"n_demand": "low", "fixes_n": True, "residue": "low"},
        "tomato": {"n_demand": "medium", "fixes_n": False, "residue": "low"},
        "potato": {"n_demand": "medium", "fixes_n": False, "residue": "low"},
        "cotton": {"n_demand": "high", "fixes_n": False, "residue": "medium"},
        "mustard": {"n_demand": "medium", "fixes_n": False, "residue": "medium"},
        "sugarcane": {"n_demand": "very_high", "fixes_n": False, "residue": "high"},
        "onion": {"n_demand": "medium", "fixes_n": False, "residue": "low"},
        "sunflower": {"n_demand": "medium", "fixes_n": False, "residue": "medium"},
    }

    SEASON_CROPS = {
        "kharif": ["rice", "corn", "soybean", "cotton", "groundnut", "greengram", "sugarcane"],
        "rabi": ["wheat", "chickpea", "lentil", "mustard", "potato", "onion"],
        "zaid": ["greengram", "sunflower", "tomato", "cucumber"],
    }

    COMPATIBILITY = {
        ("rice", "wheat"): 0.9,
        ("rice", "chickpea"): 0.85,
        ("corn", "soybean"): 0.95,
        ("wheat", "soybean"): 0.9,
        ("cotton", "wheat"): 0.8,
        ("rice", "lentil"): 0.85,
        ("corn", "chickpea"): 0.85,
        ("potato", "wheat"): 0.75,
        ("sugarcane", "wheat"): 0.5,  # sugarcane is long duration
        ("tomato", "wheat"): 0.7,
    }

    def plan(self, current_crop, soil_type, seasons=4, region="north_india"):
        """Generate optimal crop rotation plan."""
        rotation = [current_crop]
        scores = []
        total_score = 0

        for i in range(seasons - 1):
            season_idx = (i + 1) % 3
            season = ["kharif", "rabi", "zaid"][season_idx]
            prev_crop = rotation[-1]

            candidates = self._get_candidates(prev_crop, season, soil_type)
            if not candidates:
                # Fallback: pick any crop from the season
                season_crops = self.SEASON_CROPS.get(season, self.SEASON_CROPS["kharif"])
                candidates = [(c, 0.5) for c in season_crops if c != prev_crop]
                if not candidates:
                    candidates = [(season_crops[0], 0.4)]

            best_crop, best_score = max(candidates, key=lambda x: x[1])
            rotation.append(best_crop)
            scores.append(round(best_score, 2))
            total_score += best_score

        avg_score = total_score / max(len(scores), 1)

        return {
            "rotation": rotation,
            "seasons": self._assign_seasons(rotation),
            "scores": scores,
            "average_score": round(avg_score, 2),
            "soil_health_impact": self._soil_health_analysis(rotation),
            "economic_analysis": self._economic_analysis(rotation),
            "nutrient_balance": self._nutrient_balance(rotation),
            "recommendations": self._rotation_recommendations(rotation, soil_type),
        }

    def _get_candidates(self, prev_crop, season, soil_type):
        """Get candidate crops with scores."""
        season_crops = self.SEASON_CROPS.get(season, [])
        prev_family = self._get_family(prev_crop)
        candidates = []

        for crop in season_crops:
            if crop == prev_crop:
                continue

            score = 0.5  # base score

            # Bonus for different family (disease break)
            crop_family = self._get_family(crop)
            if crop_family != prev_family:
                score += 0.2

            # Bonus for legume after cereal (N fixation)
            prev_info = self.CROP_NUTRIENTS.get(prev_crop, {})
            crop_info = self.CROP_NUTRIENTS.get(crop, {})
            if crop_info.get("fixes_n") and prev_info.get("n_demand") in ("high", "very_high"):
                score += 0.2

            # Bonus for cereal after legume
            if not crop_info.get("fixes_n") and prev_info.get("fixes_n"):
                score += 0.15

            # Compatibility bonus
            pair = (prev_crop, crop)
            reverse_pair = (crop, prev_crop)
            compat = self.COMPATIBILITY.get(pair, self.COMPATIBILITY.get(reverse_pair, 0))
            score += compat * 0.2

            # Soil type suitability
            score += self._soil_suitability(crop, soil_type) * 0.15

            candidates.append((crop, min(1.0, score)))

        return candidates

    def _get_family(self, crop):
        for family, crops in self.CROP_FAMILIES.items():
            if crop in crops:
                return family
        return "other"

    def _soil_suitability(self, crop, soil_type):
        suitability = {
            "alluvial": {"rice": 0.9, "wheat": 0.9, "sugarcane": 0.8, "corn": 0.7},
            "black": {"cotton": 0.9, "soybean": 0.8, "wheat": 0.7, "chickpea": 0.8},
            "red": {"groundnut": 0.9, "potato": 0.7, "corn": 0.7, "tomato": 0.6},
            "laterite": {"cashew": 0.9, "rice": 0.5, "groundnut": 0.6},
            "sandy": {"groundnut": 0.8, "potato": 0.7, "tomato": 0.6},
            "clay": {"rice": 0.9, "wheat": 0.7, "cotton": 0.7},
        }
        return suitability.get(soil_type, {}).get(crop, 0.5)

    def _assign_seasons(self, rotation):
        seasons = []
        for i, crop in enumerate(rotation):
            season_idx = i % 3
            season = ["kharif", "rabi", "zaid"][season_idx]
            months = {"kharif": "Jun-Oct", "rabi": "Nov-Mar", "zaid": "Apr-May"}
            seasons.append({"crop": crop, "season": season, "months": months[season]})
        return seasons

    def _soil_health_analysis(self, rotation):
        n_fixers = sum(1 for c in rotation if self.CROP_NUTRIENTS.get(c, {}).get("fixes_n"))
        heavy_feeders = sum(1 for c in rotation
                          if self.CROP_NUTRIENTS.get(c, {}).get("n_demand") in ("high", "very_high"))
        families_used = len(set(self._get_family(c) for c in rotation))
        diversity_score = min(1.0, families_used / 3)
        balance_score = min(1.0, (n_fixers + 1) / max(heavy_feeders, 1))

        return {
            "n_fixing_crops": n_fixers,
            "heavy_feeders": heavy_feeders,
            "crop_families_used": families_used,
            "diversity_score": round(diversity_score, 2),
            "nutrient_balance_score": round(balance_score, 2),
            "overall_health": round((diversity_score + balance_score) / 2, 2),
        }

    def _economic_analysis(self, rotation):
        prices = {
            "rice": 22, "wheat": 25, "corn": 18, "soybean": 38,
            "cotton": 55, "chickpea": 45, "lentil": 50, "groundnut": 42,
            "potato": 15, "tomato": 30, "mustard": 48, "sugarcane": 3.5,
            "onion": 20, "sunflower": 40, "greengram": 55,
        }
        yields_kg = {
            "rice": 4500, "wheat": 3800, "corn": 8000, "soybean": 2800,
            "cotton": 2000, "chickpea": 1500, "lentil": 1200, "groundnut": 2000,
            "potato": 25000, "tomato": 35000, "mustard": 1500, "sugarcane": 70000,
            "onion": 15000, "sunflower": 1200, "greengram": 800,
        }
        analysis = []
        total_revenue = 0
        for crop in rotation:
            price = prices.get(crop, 20)
            yld = yields_kg.get(crop, 2000)
            revenue = price * yld
            total_revenue += revenue
            analysis.append({"crop": crop, "yield_kg": yld, "price_per_kg": price,
                           "revenue_inr": round(revenue)})
        return {"per_crop": analysis, "total_revenue_inr": round(total_revenue),
                "avg_revenue_per_season": round(total_revenue / len(rotation))}

    def _nutrient_balance(self, rotation):
        n_added = sum(30 for c in rotation if self.CROP_NUTRIENTS.get(c, {}).get("fixes_n"))
        n_removed = sum({"low": 10, "medium": 25, "high": 40, "very_high": 55}.get(
            self.CROP_NUTRIENTS.get(c, {}).get("n_demand", "medium"), 25) for c in rotation)
        return {
            "nitrogen_fixed_kg": n_added,
            "nitrogen_consumed_kg": n_removed,
            "net_nitrogen_balance": n_added - n_removed,
            "recommendation": "Balanced" if n_added >= n_removed * 0.6 else "Add supplemental nitrogen",
        }

    def _rotation_recommendations(self, rotation, soil_type):
        recs = []
        families = [self._get_family(c) for c in rotation]
        if families.count("cereals") > len(rotation) * 0.6:
            recs.append("Too many cereals in sequence. Add more legumes for nitrogen fixation.")
        if "legumes" not in families:
            recs.append("Include at least one legume crop for biological nitrogen fixation.")
        if len(set(rotation)) < len(rotation) * 0.5:
            recs.append("Low crop diversity. Diversify to break pest and disease cycles.")
        if not recs:
            recs.append("Rotation plan looks well-balanced for soil health and economics.")
        return recs

    def get_model_info(self):
        return {
            "name": "Crop Rotation Planner",
            "algorithm": "Multi-objective Graph Scoring",
            "supported_crops": len(self.CROP_NUTRIENTS),
            "crop_families": len(self.CROP_FAMILIES),
        }
