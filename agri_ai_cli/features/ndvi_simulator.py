"""
Sprint 6 - Satellite NDVI Simulator.
Simulates Normalized Difference Vegetation Index analysis for crop health monitoring.
"""

import math
import random


class NDVISimulator:
    """Simulated satellite NDVI analysis for crop health assessment."""

    # Typical NDVI ranges by growth stage
    NDVI_PROFILES = {
        "rice": {
            "initial": (0.15, 0.30),
            "vegetative": (0.40, 0.65),
            "reproductive": (0.60, 0.85),
            "maturity": (0.30, 0.50),
            "harvest": (0.10, 0.25),
        },
        "wheat": {
            "initial": (0.10, 0.25),
            "vegetative": (0.35, 0.60),
            "reproductive": (0.55, 0.80),
            "maturity": (0.25, 0.45),
            "harvest": (0.08, 0.20),
        },
        "corn": {
            "initial": (0.12, 0.28),
            "vegetative": (0.45, 0.70),
            "reproductive": (0.65, 0.90),
            "maturity": (0.35, 0.55),
            "harvest": (0.10, 0.22),
        },
    }

    HEALTH_THRESHOLDS = {
        "bare_soil": (0.0, 0.1),
        "sparse_vegetation": (0.1, 0.2),
        "moderate_vegetation": (0.2, 0.4),
        "dense_vegetation": (0.4, 0.6),
        "very_dense_vegetation": (0.6, 0.8),
        "peak_vegetation": (0.8, 1.0),
    }

    def analyze(self, crop="rice", growth_stage="reproductive", field_size_ha=5.0,
                stress_factor=0.0, seed=None):
        """Simulate NDVI satellite analysis for a crop field."""
        if seed is not None:
            random.seed(seed)

        profile = self.NDVI_PROFILES.get(crop, self.NDVI_PROFILES["rice"])
        stage_range = profile.get(growth_stage, (0.3, 0.6))

        # Generate pixel-level NDVI data (simulated 10m resolution)
        pixels_per_ha = 100  # 10m x 10m = 100 pixels per hectare
        n_pixels = int(field_size_ha * pixels_per_ha)
        n_pixels = min(n_pixels, 2000)  # cap for performance

        base_ndvi = random.uniform(*stage_range)
        # Apply stress
        base_ndvi *= (1 - stress_factor * 0.4)

        ndvi_values = []
        for _ in range(n_pixels):
            noise = random.gauss(0, 0.05)
            val = max(0, min(1, base_ndvi + noise))
            ndvi_values.append(val)

        # Statistics
        mean_ndvi = sum(ndvi_values) / len(ndvi_values)
        sorted_vals = sorted(ndvi_values)
        median_ndvi = sorted_vals[len(sorted_vals) // 2]
        min_ndvi = sorted_vals[0]
        max_ndvi = sorted_vals[-1]
        std_ndvi = math.sqrt(sum((v - mean_ndvi) ** 2 for v in ndvi_values) / len(ndvi_values))

        # Classify pixels
        classification = self._classify_pixels(ndvi_values)

        # Health assessment
        health = self._assess_health(mean_ndvi, std_ndvi, growth_stage, crop)

        # Anomaly detection
        anomalies = self._detect_anomalies(ndvi_values, mean_ndvi, std_ndvi)

        # Generate field map (simplified grid)
        field_map = self._generate_field_map(ndvi_values, field_size_ha)

        return {
            "crop": crop,
            "growth_stage": growth_stage,
            "field_size_ha": field_size_ha,
            "statistics": {
                "mean_ndvi": round(mean_ndvi, 4),
                "median_ndvi": round(median_ndvi, 4),
                "min_ndvi": round(min_ndvi, 4),
                "max_ndvi": round(max_ndvi, 4),
                "std_dev": round(std_ndvi, 4),
                "n_pixels": n_pixels,
            },
            "classification": classification,
            "health_assessment": health,
            "anomalies": anomalies,
            "field_map": field_map,
            "recommendations": self._ndvi_recommendations(health, anomalies, crop),
            "temporal_analysis": self._temporal_profile(crop),
        }

    def _classify_pixels(self, ndvi_values):
        """Classify NDVI pixels into vegetation categories."""
        counts = {k: 0 for k in self.HEALTH_THRESHOLDS}
        for v in ndvi_values:
            for cat, (low, high) in self.HEALTH_THRESHOLDS.items():
                if low <= v < high:
                    counts[cat] += 1
                    break

        total = len(ndvi_values)
        return {k: {"count": v, "percentage": round(v / total * 100, 1)}
                for k, v in counts.items() if v > 0}

    def _assess_health(self, mean_ndvi, std_ndvi, stage, crop):
        """Assess overall crop health from NDVI."""
        profile = self.NDVI_PROFILES.get(crop, self.NDVI_PROFILES["rice"])
        expected = profile.get(stage, (0.3, 0.6))
        expected_mid = (expected[0] + expected[1]) / 2

        deviation = (mean_ndvi - expected_mid) / max(expected_mid, 0.01)

        if deviation >= -0.1 and std_ndvi < 0.1:
            status = "HEALTHY"
            score = min(1.0, 0.8 + deviation)
        elif deviation >= -0.2:
            status = "MODERATE STRESS"
            score = max(0.3, 0.6 + deviation)
        else:
            status = "SEVERE STRESS"
            score = max(0.1, 0.4 + deviation)

        uniformity = max(0, 1 - std_ndvi / 0.15)

        return {
            "status": status,
            "health_score": round(score, 2),
            "uniformity_score": round(uniformity, 2),
            "expected_ndvi_range": f"{expected[0]:.2f} - {expected[1]:.2f}",
            "actual_mean": round(mean_ndvi, 3),
            "deviation_from_expected": round(deviation * 100, 1),
        }

    def _detect_anomalies(self, ndvi_values, mean, std):
        """Detect spatial anomalies in NDVI data."""
        threshold = mean - 2 * std
        anomaly_pixels = sum(1 for v in ndvi_values if v < threshold)
        anomaly_pct = anomaly_pixels / len(ndvi_values) * 100

        if anomaly_pct > 20:
            severity = "HIGH"
            interpretation = "Large area showing stress - possible disease outbreak or water deficit"
        elif anomaly_pct > 10:
            severity = "MEDIUM"
            interpretation = "Moderate stress patches - investigate specific areas"
        elif anomaly_pct > 5:
            severity = "LOW"
            interpretation = "Minor stress spots - likely edge effects or small issues"
        else:
            severity = "NONE"
            interpretation = "Field appears uniformly healthy"

        return {
            "anomaly_pixels": anomaly_pixels,
            "anomaly_percentage": round(anomaly_pct, 1),
            "severity": severity,
            "threshold_ndvi": round(threshold, 3),
            "interpretation": interpretation,
        }

    def _generate_field_map(self, ndvi_values, field_size_ha):
        """Generate a simplified text-based field map."""
        # Create a grid representation
        grid_size = min(10, max(3, int(math.sqrt(field_size_ha) * 3)))
        pixels_per_cell = max(1, len(ndvi_values) // (grid_size * grid_size))

        grid = []
        idx = 0
        for row in range(grid_size):
            row_data = []
            for col in range(grid_size):
                chunk = ndvi_values[idx:idx + pixels_per_cell]
                if chunk:
                    cell_val = sum(chunk) / len(chunk)
                else:
                    cell_val = 0.5
                idx += pixels_per_cell

                # Map to visual character
                if cell_val >= 0.7:
                    symbol = "#"  # very healthy
                elif cell_val >= 0.5:
                    symbol = "+"  # healthy
                elif cell_val >= 0.3:
                    symbol = "."  # moderate
                elif cell_val >= 0.1:
                    symbol = "-"  # sparse
                else:
                    symbol = " "  # bare
                row_data.append({"value": round(cell_val, 2), "symbol": symbol})
            grid.append(row_data)

        return {
            "grid": grid,
            "legend": {"#": "NDVI>0.7 (Dense)", "+": "0.5-0.7 (Good)",
                       ".": "0.3-0.5 (Moderate)", "-": "0.1-0.3 (Sparse)", " ": "<0.1 (Bare)"},
            "grid_size": f"{grid_size}x{grid_size}",
        }

    def _temporal_profile(self, crop):
        """Expected NDVI temporal profile for the crop."""
        profile = self.NDVI_PROFILES.get(crop, self.NDVI_PROFILES["rice"])
        return [
            {"stage": stage, "expected_ndvi": f"{r[0]:.2f}-{r[1]:.2f}",
             "mid_value": round((r[0] + r[1]) / 2, 2)}
            for stage, r in profile.items()
        ]

    def _ndvi_recommendations(self, health, anomalies, crop):
        recs = []
        if health["status"] == "SEVERE STRESS":
            recs.append("URGENT: Significant crop stress detected. Immediate field inspection needed.")
            recs.append("Check for water stress, nutrient deficiency, or disease symptoms.")
        elif health["status"] == "MODERATE STRESS":
            recs.append("Some stress detected. Schedule field visit within 2-3 days.")
            recs.append("Compare with soil moisture and recent weather data.")

        if anomalies["severity"] in ("HIGH", "MEDIUM"):
            recs.append(f"Anomalous patches cover {anomalies['anomaly_percentage']}% of field.")
            recs.append("Use GPS coordinates to locate and investigate affected zones.")

        if health["uniformity_score"] < 0.5:
            recs.append("High variability in crop growth. Consider variable-rate fertilization.")

        if not recs:
            recs.append(f"Crop health looks good for {crop} at current growth stage.")
            recs.append("Continue regular monitoring with bi-weekly satellite passes.")

        return recs

    def get_model_info(self):
        return {
            "name": "NDVI Simulator",
            "algorithm": "Simulated Satellite Remote Sensing",
            "resolution": "10m (simulated Sentinel-2)",
            "crops_profiled": len(self.NDVI_PROFILES),
        }
