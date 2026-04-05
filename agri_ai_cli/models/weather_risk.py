"""
Sprint 1 - Weather Risk Assessment Model.
Multi-factor weighted scoring system for agricultural weather risk analysis.
"""

import math
from agri_ai_cli.utils.data_generator import generate_weather_scenarios


class WeatherRiskAssessor:
    """Weather risk assessment engine for agricultural planning."""

    def __init__(self):
        self.risk_weights = {
            "heat": 0.25,
            "flood": 0.25,
            "drought": 0.20,
            "wind": 0.20,
            "frost": 0.10,
        }
        self.crop_sensitivity = {
            "rice": {"heat": 0.8, "flood": 0.3, "drought": 1.0, "wind": 0.5, "frost": 0.9},
            "wheat": {"heat": 0.9, "flood": 0.8, "drought": 0.7, "wind": 0.6, "frost": 0.4},
            "corn": {"heat": 0.7, "flood": 0.7, "drought": 0.8, "wind": 0.8, "frost": 0.9},
            "tomato": {"heat": 0.8, "flood": 0.9, "drought": 0.6, "wind": 0.9, "frost": 1.0},
            "cotton": {"heat": 0.4, "flood": 0.8, "drought": 0.5, "wind": 0.7, "frost": 1.0},
            "soybean": {"heat": 0.7, "flood": 0.7, "drought": 0.7, "wind": 0.6, "frost": 0.8},
            "potato": {"heat": 0.9, "flood": 0.8, "drought": 0.6, "wind": 0.4, "frost": 0.7},
        }

    def assess(self, temperature, humidity, rainfall, wind_speed, cloud_cover, crop="rice"):
        """Perform comprehensive weather risk assessment."""
        # Calculate individual risk dimensions
        risks = {
            "heat": self._heat_risk(temperature),
            "flood": self._flood_risk(rainfall, humidity),
            "drought": self._drought_risk(rainfall, humidity, temperature),
            "wind": self._wind_risk(wind_speed),
            "frost": self._frost_risk(temperature),
        }

        # Apply crop sensitivity
        sensitivity = self.crop_sensitivity.get(crop, self.crop_sensitivity["rice"])
        adjusted_risks = {k: min(1.0, v * sensitivity.get(k, 1.0)) for k, v in risks.items()}

        # Weighted overall score
        overall = sum(adjusted_risks[k] * self.risk_weights[k] for k in risks)
        overall = min(1.0, overall)

        risk_level = self._risk_level(overall)

        return {
            "overall_score": round(overall, 3),
            "risk_level": risk_level,
            "individual_risks": {k: round(v, 3) for k, v in adjusted_risks.items()},
            "raw_risks": {k: round(v, 3) for k, v in risks.items()},
            "crop": crop,
            "weather": {
                "temperature": temperature,
                "humidity": humidity,
                "rainfall": rainfall,
                "wind_speed": wind_speed,
                "cloud_cover": cloud_cover,
            },
            "advisories": self._generate_advisories(adjusted_risks, crop, temperature, rainfall),
            "forecast_action": self._action_plan(risk_level, adjusted_risks),
        }

    def _heat_risk(self, temp):
        if temp > 42:
            return 1.0
        elif temp > 35:
            return (temp - 35) / 7
        elif temp < 5:
            return (5 - temp) / 10
        return 0.0

    def _flood_risk(self, rainfall, humidity):
        rain_risk = max(0, (rainfall - 100) / 200) if rainfall > 100 else 0
        humid_factor = max(0, (humidity - 85) / 15) if humidity > 85 else 0
        return min(1.0, rain_risk + humid_factor * 0.3)

    def _drought_risk(self, rainfall, humidity, temp):
        if rainfall < 10:
            rain_factor = 1.0
        elif rainfall < 50:
            rain_factor = 1 - rainfall / 50
        else:
            rain_factor = 0
        heat_factor = max(0, (temp - 35) / 10)
        dry_factor = max(0, (40 - humidity) / 40) if humidity < 40 else 0
        return min(1.0, rain_factor * 0.5 + heat_factor * 0.25 + dry_factor * 0.25)

    def _wind_risk(self, wind_speed):
        if wind_speed > 80:
            return 1.0
        elif wind_speed > 40:
            return (wind_speed - 40) / 40
        return 0.0

    def _frost_risk(self, temp):
        if temp < -5:
            return 1.0
        elif temp < 2:
            return (2 - temp) / 7
        return 0.0

    def _risk_level(self, score):
        if score > 0.7:
            return "CRITICAL"
        elif score > 0.4:
            return "HIGH"
        elif score > 0.2:
            return "MEDIUM"
        return "LOW"

    def _generate_advisories(self, risks, crop, temp, rainfall):
        advisories = []
        if risks["heat"] > 0.5:
            advisories.append(f"HEAT WARNING: Temperature {temp}C poses risk to {crop}. "
                            "Increase irrigation frequency and apply mulch.")
        if risks["flood"] > 0.5:
            advisories.append(f"FLOOD ALERT: Heavy rainfall ({rainfall}mm) expected. "
                            "Ensure drainage channels are clear.")
        if risks["drought"] > 0.5:
            advisories.append("DROUGHT CONCERN: Low moisture conditions detected. "
                            "Activate supplemental irrigation immediately.")
        if risks["wind"] > 0.5:
            advisories.append("WIND WARNING: High winds may damage standing crops. "
                            "Consider windbreaks and support structures.")
        if risks["frost"] > 0.3:
            advisories.append(f"FROST ALERT: Temperature {temp}C may cause frost damage. "
                            "Cover sensitive crops and use anti-frost sprinklers.")
        if not advisories:
            advisories.append(f"Weather conditions are favorable for {crop} cultivation.")
        return advisories

    def _action_plan(self, level, risks):
        if level == "CRITICAL":
            return {
                "urgency": "IMMEDIATE ACTION REQUIRED",
                "actions": [
                    "Postpone any planned field operations",
                    "Activate emergency crop protection measures",
                    "Contact insurance provider if damage occurs",
                    "Document field conditions for claim support",
                ],
            }
        elif level == "HIGH":
            return {
                "urgency": "ACTION RECOMMENDED WITHIN 24 HOURS",
                "actions": [
                    "Monitor weather updates hourly",
                    "Prepare protective measures for standing crops",
                    "Delay harvesting if possible until risk subsides",
                ],
            }
        elif level == "MEDIUM":
            return {
                "urgency": "MONITOR CLOSELY",
                "actions": [
                    "Check weather forecasts twice daily",
                    "Ensure irrigation systems are operational",
                    "Have protective materials on standby",
                ],
            }
        return {
            "urgency": "ROUTINE MONITORING",
            "actions": [
                "Continue normal farming operations",
                "Weekly weather check is sufficient",
            ],
        }

    def batch_assess(self, n_scenarios=20, crop="rice"):
        """Generate and assess multiple weather scenarios."""
        scenarios = generate_weather_scenarios(n_scenarios)
        results = []
        for s in scenarios:
            result = self.assess(
                s["temperature"], s["humidity"], s["rainfall"],
                s["wind_speed"], s["cloud_cover"], crop
            )
            results.append(result)
        return results

    def get_model_info(self):
        return {
            "name": "Weather Risk Assessor",
            "algorithm": "Weighted Multi-Factor Scoring",
            "risk_dimensions": 5,
            "supported_crops": list(self.crop_sensitivity.keys()),
        }
