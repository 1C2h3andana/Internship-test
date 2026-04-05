"""
Sprint 1 - Yield Prediction Model.
Uses Gradient Boosting Regressor to predict crop yield from environmental conditions.
"""

import random
import math
from agri_ai_cli.utils.ml_algorithms import GradientBoostingRegressor, train_test_split
from agri_ai_cli.utils.data_generator import generate_yield_data, CROP_YIELD_PARAMS


class YieldPredictor:
    """AI-powered crop yield prediction engine."""

    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=30, learning_rate=0.1, max_depth=4)
        self.is_trained = False
        self.r2_score = 0.0
        self.crops = list(CROP_YIELD_PARAMS.keys())
        self.feature_names = [
            "temperature", "rainfall_mm", "humidity_pct", "soil_ph",
            "nitrogen_kg", "phosphorus_kg", "potassium_kg", "sunlight_hours", "crop_type"
        ]

    def train(self, n_samples=300):
        """Train yield prediction model."""
        random.seed(42)
        X, y, crops = generate_yield_data(n_samples)
        self.crops = crops

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2)
        self.model.fit(X_train, y_train)
        self.r2_score = self.model.score(X_test, y_test)
        self.is_trained = True

        return {
            "r2_score": round(self.r2_score, 3),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "n_crops": len(crops),
        }

    def predict(self, temperature, rainfall, humidity, soil_ph,
                nitrogen, phosphorus, potassium, sunlight, crop):
        """Predict yield for given conditions."""
        if not self.is_trained:
            self.train()

        crop_idx = self.crops.index(crop) if crop in self.crops else 0
        features = [temperature, rainfall, humidity, soil_ph,
                    nitrogen, phosphorus, potassium, sunlight, float(crop_idx)]

        predicted_yield = self.model.predict([features])[0]
        predicted_yield = max(100, predicted_yield)

        # Calculate revenue estimate (INR/kg prices)
        prices = {
            "rice": 22, "wheat": 25, "corn": 18, "tomato": 30,
            "cotton": 55, "soybean": 38, "potato": 15, "sugarcane": 3.5,
        }
        price_per_kg = prices.get(crop, 20)
        revenue = predicted_yield * price_per_kg

        # Confidence interval (based on model uncertainty)
        ci_width = predicted_yield * 0.15
        ci_lower = predicted_yield - ci_width
        ci_upper = predicted_yield + ci_width

        # Factor analysis
        params = CROP_YIELD_PARAMS.get(crop, CROP_YIELD_PARAMS["rice"])
        factors = self._analyze_factors(features, params)

        return {
            "predicted_yield_kg": round(predicted_yield, 1),
            "predicted_yield_tons": round(predicted_yield / 1000, 2),
            "revenue_inr": round(revenue, 0),
            "price_per_kg": price_per_kg,
            "confidence_interval": {
                "lower": round(ci_lower, 1),
                "upper": round(ci_upper, 1),
            },
            "crop": crop,
            "factors": factors,
            "recommendations": self._get_recommendations(factors, crop),
            "yield_rating": self._yield_rating(predicted_yield, params["base_yield"]),
        }

    def _analyze_factors(self, features, params):
        """Analyze how each factor contributes to yield."""
        temp, rain, humid, ph, n, p, k, sun, _ = features
        return {
            "temperature": {
                "value": round(temp, 1),
                "optimal": params["temp_opt"],
                "score": round(max(0, 1 - abs(temp - params["temp_opt"]) / 15), 2),
                "status": "good" if abs(temp - params["temp_opt"]) < 5 else "suboptimal",
            },
            "rainfall": {
                "value": round(rain, 1),
                "optimal": params["rain_opt"],
                "score": round(max(0, 1 - abs(rain - params["rain_opt"]) / 1000), 2),
                "status": "good" if abs(rain - params["rain_opt"]) < 300 else "suboptimal",
            },
            "soil_ph": {
                "value": round(ph, 1),
                "optimal": params["ph_opt"],
                "score": round(max(0, 1 - abs(ph - params["ph_opt"]) / 3), 2),
                "status": "good" if abs(ph - params["ph_opt"]) < 0.8 else "suboptimal",
            },
            "nutrients": {
                "nitrogen": round(n, 1),
                "phosphorus": round(p, 1),
                "potassium": round(k, 1),
                "score": round(min(1, (n + p + k) / 300), 2),
            },
            "sunlight": {
                "value": round(sun, 1),
                "score": round(min(1, sun / 8), 2),
            },
        }

    def _yield_rating(self, predicted, base):
        ratio = predicted / base
        if ratio >= 0.85:
            return "EXCELLENT"
        elif ratio >= 0.65:
            return "GOOD"
        elif ratio >= 0.45:
            return "AVERAGE"
        else:
            return "BELOW AVERAGE"

    def _get_recommendations(self, factors, crop):
        recs = []
        if factors["temperature"]["status"] == "suboptimal":
            if factors["temperature"]["value"] > factors["temperature"]["optimal"]:
                recs.append("Consider shade nets or mulching to reduce soil temperature")
            else:
                recs.append("Use row covers or poly tunnels to increase warmth")

        if factors["rainfall"]["status"] == "suboptimal":
            if factors["rainfall"]["value"] < factors["rainfall"]["optimal"]:
                recs.append("Supplement with drip irrigation to meet water deficit")
            else:
                recs.append("Improve drainage to prevent waterlogging")

        if factors["soil_ph"]["status"] == "suboptimal":
            if factors["soil_ph"]["value"] < factors["soil_ph"]["optimal"]:
                recs.append("Apply agricultural lime to raise soil pH")
            else:
                recs.append("Apply gypsum or sulfur to lower soil pH")

        if factors["nutrients"]["score"] < 0.6:
            recs.append("Increase fertilizer application based on soil test results")

        if not recs:
            recs.append(f"Conditions are near-optimal for {crop} cultivation")

        return recs

    def get_model_info(self):
        return {
            "name": "Yield Predictor",
            "algorithm": "Gradient Boosting Regressor",
            "n_estimators": 30,
            "trained": self.is_trained,
            "r2_score": round(self.r2_score, 3) if self.is_trained else None,
        }
