"""
Sprint 1 - Soil Analysis Model.
Uses KNN classifier to identify soil type and recommend crops from nutrient profiles.
"""

import random
from agri_ai_cli.utils.ml_algorithms import KNNClassifier, train_test_split
from agri_ai_cli.utils.data_generator import generate_soil_data, SOIL_TYPES


class SoilAnalyzer:
    """AI-powered soil analysis and crop recommendation engine."""

    def __init__(self):
        self.model = KNNClassifier(k=5)
        self.is_trained = False
        self.accuracy = 0.0
        self.soil_types = list(SOIL_TYPES.keys())
        self.feature_names = [
            "nitrogen_kg_ha", "phosphorus_kg_ha", "potassium_kg_ha",
            "ph", "organic_carbon_pct", "moisture_pct", "temperature_c"
        ]

    def train(self, n_samples=250):
        """Train soil classification model."""
        random.seed(42)
        X, y, soil_types = generate_soil_data(n_samples)
        self.soil_types = soil_types

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2)
        self.model.fit(X_train, y_train)
        self.accuracy = self.model.score(X_test, y_test)
        self.is_trained = True

        return {
            "accuracy": round(self.accuracy, 3),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "n_soil_types": len(soil_types),
        }

    def analyze(self, nitrogen, phosphorus, potassium, ph, organic_carbon, moisture, temperature):
        """Analyze soil from nutrient profile."""
        if not self.is_trained:
            self.train()

        features = [nitrogen, phosphorus, potassium, ph, organic_carbon, moisture, temperature]
        soil_type = self.model.predict([features])[0]
        probas = self.model.predict_proba([features])[0]

        sorted_probs = sorted(probas.items(), key=lambda x: x[1], reverse=True)
        soil_info = SOIL_TYPES.get(soil_type, {})

        # Nutrient scoring
        nutrient_scores = self._score_nutrients(nitrogen, phosphorus, potassium, ph, organic_carbon)

        # Overall health
        health_score = sum(nutrient_scores.values()) / len(nutrient_scores)

        return {
            "soil_type": soil_type,
            "soil_type_name": soil_type.title(),
            "confidence": sorted_probs[0][1] if sorted_probs else 0.5,
            "health_score": round(health_score, 2),
            "health_label": self._health_label(health_score),
            "nutrient_scores": nutrient_scores,
            "recommended_crops": soil_info.get("best_crops", []),
            "fertilizer_plan": self._fertilizer_plan(nitrogen, phosphorus, potassium, ph),
            "soil_improvement": self._soil_improvement(soil_type, nutrient_scores),
            "type_probabilities": [
                {"type": t.title(), "probability": round(p, 3)}
                for t, p in sorted_probs[:4]
            ],
        }

    def _score_nutrients(self, n, p, k, ph, oc):
        """Score individual nutrients on 0-1 scale."""
        def clamp_score(val, low, high):
            if val < low:
                return val / low
            elif val > high:
                return max(0, 1 - (val - high) / high)
            return 1.0

        return {
            "nitrogen": round(clamp_score(n, 120, 280), 2),
            "phosphorus": round(clamp_score(p, 15, 50), 2),
            "potassium": round(clamp_score(k, 100, 300), 2),
            "ph_balance": round(1 - abs(ph - 6.5) / 3, 2),
            "organic_carbon": round(min(1, oc / 1.0), 2),
        }

    def _health_label(self, score):
        if score >= 0.8:
            return "EXCELLENT"
        elif score >= 0.6:
            return "GOOD"
        elif score >= 0.4:
            return "FAIR"
        else:
            return "POOR"

    def _fertilizer_plan(self, n, p, k, ph):
        """Generate fertilizer recommendation."""
        plan = []
        if n < 150:
            deficit = 150 - n
            plan.append(f"Apply {round(deficit * 2.17, 1)} kg/ha Urea (N deficit: {round(deficit, 1)} kg)")
        if p < 25:
            deficit = 25 - p
            plan.append(f"Apply {round(deficit * 5.43, 1)} kg/ha DAP (P deficit: {round(deficit, 1)} kg)")
        if k < 120:
            deficit = 120 - k
            plan.append(f"Apply {round(deficit * 1.67, 1)} kg/ha MOP (K deficit: {round(deficit, 1)} kg)")
        if ph < 5.5:
            plan.append(f"Apply agricultural lime at 2-4 tons/ha to raise pH from {round(ph, 1)}")
        elif ph > 8.0:
            plan.append(f"Apply gypsum at 1-2 tons/ha to lower pH from {round(ph, 1)}")

        if not plan:
            plan.append("Nutrient levels are adequate. Maintain current fertilization schedule.")

        return plan

    def _soil_improvement(self, soil_type, scores):
        """Provide soil-specific improvement suggestions."""
        improvements = {
            "sandy": [
                "Add organic matter (compost/manure) to improve water retention",
                "Use cover crops to prevent erosion",
                "Apply mulch to reduce moisture evaporation",
            ],
            "clay": [
                "Add gypsum to improve soil structure",
                "Incorporate sand and organic matter for better drainage",
                "Avoid working soil when wet to prevent compaction",
            ],
            "laterite": [
                "Heavy organic matter application needed",
                "Lime application to correct acidity",
                "Green manuring with leguminous crops",
            ],
            "red": [
                "Regular organic matter additions",
                "Contour farming to prevent erosion",
                "Phosphorus supplementation usually needed",
            ],
            "alluvial": [
                "Maintain organic carbon through crop residue incorporation",
                "Practice balanced fertilization",
                "Good drainage management in flood-prone areas",
            ],
            "black": [
                "Avoid overwatering as black soil retains moisture",
                "Deep plowing to improve aeration",
                "Use raised beds for better drainage",
            ],
        }
        return improvements.get(soil_type, [
            "Add organic matter regularly",
            "Test soil annually for nutrient status",
            "Practice crop rotation for soil health",
        ])

    def get_model_info(self):
        return {
            "name": "Soil Analyzer",
            "algorithm": "K-Nearest Neighbors (k=5)",
            "n_soil_types": len(self.soil_types),
            "trained": self.is_trained,
            "accuracy": round(self.accuracy, 3) if self.is_trained else None,
        }
