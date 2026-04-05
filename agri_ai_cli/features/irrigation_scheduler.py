"""
Sprint 2 - Irrigation Scheduler.
FAO-56 Penman-Monteith based evapotranspiration calculation and scheduling.
"""

import math


class IrrigationScheduler:
    """Smart irrigation scheduling using FAO-56 methodology."""

    # Crop coefficients (Kc) by growth stage
    CROP_KC = {
        "rice": {"initial": 1.05, "mid": 1.20, "late": 0.90},
        "wheat": {"initial": 0.40, "mid": 1.15, "late": 0.25},
        "corn": {"initial": 0.30, "mid": 1.20, "late": 0.35},
        "tomato": {"initial": 0.60, "mid": 1.15, "late": 0.80},
        "cotton": {"initial": 0.35, "mid": 1.20, "late": 0.50},
        "soybean": {"initial": 0.40, "mid": 1.15, "late": 0.50},
        "potato": {"initial": 0.50, "mid": 1.15, "late": 0.75},
    }

    # Root zone depth (meters)
    ROOT_DEPTH = {
        "rice": 0.3, "wheat": 1.0, "corn": 1.2, "tomato": 0.7,
        "cotton": 1.3, "soybean": 0.8, "potato": 0.4,
    }

    # Allowable depletion fraction
    DEPLETION_FRACTION = {
        "rice": 0.20, "wheat": 0.55, "corn": 0.55, "tomato": 0.40,
        "cotton": 0.65, "soybean": 0.50, "potato": 0.35,
    }

    def schedule(self, crop, temperature, humidity, wind_speed, sunlight_hours,
                 soil_moisture_pct, field_area_ha=1.0, growth_stage="mid",
                 latitude=20.0, day_of_year=180):
        """Calculate irrigation schedule using FAO-56 method."""

        # Step 1: Reference evapotranspiration (ET0) via Penman-Monteith
        et0 = self._penman_monteith(temperature, humidity, wind_speed,
                                      sunlight_hours, latitude, day_of_year)

        # Step 2: Crop evapotranspiration (ETc)
        kc = self.CROP_KC.get(crop, self.CROP_KC["rice"]).get(growth_stage, 1.0)
        etc = et0 * kc

        # Step 3: Net irrigation requirement
        effective_rain = 0  # assume no rain for scheduling
        net_irrigation = max(0, etc - effective_rain)

        # Step 4: Soil water balance
        root_depth = self.ROOT_DEPTH.get(crop, 0.6)
        taw = 1000 * root_depth * 0.15  # total available water (mm) - simplified
        depletion_frac = self.DEPLETION_FRACTION.get(crop, 0.5)
        raw = taw * depletion_frac  # readily available water

        current_depletion = taw * (1 - soil_moisture_pct / 100)
        needs_irrigation = current_depletion > raw

        # Step 5: Irrigation amount
        if needs_irrigation:
            irrigation_depth_mm = current_depletion - (taw - raw) * 0.5
        else:
            irrigation_depth_mm = 0

        irrigation_volume_liters = irrigation_depth_mm * field_area_ha * 10000  # mm to L/ha

        # Step 6: Frequency estimation
        if etc > 0:
            days_between = max(1, int(raw / etc))
        else:
            days_between = 7

        # Step 7: Duration for drip/sprinkler
        drip_rate = 4.0  # liters per hour per emitter
        emitters_per_ha = 5000
        drip_hours = irrigation_volume_liters / (drip_rate * emitters_per_ha) if irrigation_volume_liters > 0 else 0

        sprinkler_rate = 10.0  # mm per hour
        sprinkler_hours = irrigation_depth_mm / sprinkler_rate if irrigation_depth_mm > 0 else 0

        return {
            "crop": crop,
            "growth_stage": growth_stage,
            "et0_mm_day": round(et0, 2),
            "kc": kc,
            "etc_mm_day": round(etc, 2),
            "net_irrigation_mm": round(net_irrigation, 2),
            "soil_status": {
                "current_moisture_pct": soil_moisture_pct,
                "total_available_water_mm": round(taw, 1),
                "readily_available_water_mm": round(raw, 1),
                "current_depletion_mm": round(current_depletion, 1),
                "needs_irrigation": needs_irrigation,
            },
            "recommendation": {
                "irrigation_depth_mm": round(max(0, irrigation_depth_mm), 1),
                "volume_liters_per_ha": round(max(0, irrigation_volume_liters), 0),
                "frequency_days": days_between,
                "next_irrigation": "TODAY" if needs_irrigation else f"In ~{days_between} days",
            },
            "method_comparison": {
                "drip": {
                    "duration_hours": round(drip_hours, 1),
                    "efficiency_pct": 90,
                    "water_saved_pct": 30,
                },
                "sprinkler": {
                    "duration_hours": round(sprinkler_hours, 1),
                    "efficiency_pct": 75,
                    "water_saved_pct": 15,
                },
                "flood": {
                    "depth_mm": round(max(0, irrigation_depth_mm) * 1.3, 1),
                    "efficiency_pct": 60,
                    "water_saved_pct": 0,
                },
            },
            "water_saving_tips": self._water_tips(crop, growth_stage),
        }

    def _penman_monteith(self, temp, humidity, wind, sun_hours, lat, doy):
        """Simplified FAO-56 Penman-Monteith reference ET0."""
        # Saturation vapor pressure
        es = 0.6108 * math.exp(17.27 * temp / (temp + 237.3))
        # Actual vapor pressure
        ea = es * humidity / 100
        # Vapor pressure deficit
        vpd = es - ea

        # Slope of vapor pressure curve
        delta = 4098 * es / (temp + 237.3) ** 2

        # Psychrometric constant (approx for elevation=0)
        gamma = 0.0665

        # Solar radiation estimate (simplified)
        lat_rad = lat * math.pi / 180
        dr = 1 + 0.033 * math.cos(2 * math.pi * doy / 365)
        solar_decl = 0.409 * math.sin(2 * math.pi * doy / 365 - 1.39)
        ws = math.acos(-math.tan(lat_rad) * math.tan(solar_decl))
        ra = (24 * 60 / math.pi) * 0.0820 * dr * (
            ws * math.sin(lat_rad) * math.sin(solar_decl) +
            math.cos(lat_rad) * math.cos(solar_decl) * math.sin(ws)
        )
        # Estimated solar radiation
        n_max = 24 * ws / math.pi
        rs = (0.25 + 0.50 * sun_hours / max(n_max, 1)) * ra
        # Net radiation (simplified)
        rn = 0.77 * rs - 2.0  # very simplified net radiation

        # FAO-56 equation
        numerator = 0.408 * delta * rn + gamma * (900 / (temp + 273)) * wind * vpd
        denominator = delta + gamma * (1 + 0.34 * wind)

        et0 = max(0, numerator / denominator)
        return et0

    def _water_tips(self, crop, stage):
        tips = [
            "Irrigate during early morning or late evening to reduce evaporation",
            "Use mulching to conserve soil moisture (reduces ET by 10-25%)",
            "Monitor soil moisture with tensiometers for precision scheduling",
        ]
        if crop == "rice":
            tips.append("Alternate Wetting and Drying (AWD) saves 15-30% water in rice")
        if stage == "initial":
            tips.append("Young plants need frequent but shallow irrigation")
        elif stage == "mid":
            tips.append("Peak water demand phase - do not skip irrigations")
        elif stage == "late":
            tips.append("Reduce irrigation gradually as crop approaches maturity")
        return tips

    def get_model_info(self):
        return {
            "name": "Irrigation Scheduler",
            "algorithm": "FAO-56 Penman-Monteith",
            "supported_crops": list(self.CROP_KC.keys()),
            "methods": ["drip", "sprinkler", "flood"],
        }
