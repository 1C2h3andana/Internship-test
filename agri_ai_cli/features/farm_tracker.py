"""
Sprint 4 - Farm History Tracker.
JSON-based persistent database for farm activity logging and trend analysis.
"""

import json
import os
import uuid
from datetime import datetime


class FarmTracker:
    """Persistent farm activity tracking with JSON storage."""

    def __init__(self, db_path="agri_ai_cli/data/farm_history.json"):
        self.db_path = db_path
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {"records": [], "metadata": {"created": datetime.now().isoformat(), "version": "3.0"}}

    def _save(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "w") as f:
            json.dump(self.data, f, indent=2, default=str)

    def add_record(self, crop, area_ha, yield_kg, season, cost_inr=0, revenue_inr=0, notes=""):
        """Add a farming activity record."""
        record = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now().isoformat(),
            "crop": crop,
            "area_ha": area_ha,
            "yield_kg": yield_kg,
            "yield_per_ha": round(yield_kg / max(area_ha, 0.01), 1),
            "season": season,
            "cost_inr": cost_inr,
            "revenue_inr": revenue_inr,
            "profit_inr": revenue_inr - cost_inr,
            "notes": notes,
        }
        self.data["records"].append(record)
        self._save()
        return record

    def get_records(self, crop=None, season=None, limit=20):
        """Retrieve records with optional filtering."""
        records = self.data["records"]
        if crop:
            records = [r for r in records if r.get("crop") == crop]
        if season:
            records = [r for r in records if r.get("season") == season]
        return records[-limit:]

    def get_summary(self):
        """Get overall farm performance summary."""
        records = self.data["records"]
        if not records:
            return {"message": "No records found. Start by adding farming activities."}

        total_area = sum(r.get("area_ha", 0) for r in records)
        total_yield = sum(r.get("yield_kg", 0) for r in records)
        total_cost = sum(r.get("cost_inr", 0) for r in records)
        total_revenue = sum(r.get("revenue_inr", 0) for r in records)
        crops_grown = list(set(r.get("crop", "") for r in records))

        return {
            "total_records": len(records),
            "total_area_ha": round(total_area, 2),
            "total_yield_kg": round(total_yield, 1),
            "total_cost_inr": round(total_cost),
            "total_revenue_inr": round(total_revenue),
            "total_profit_inr": round(total_revenue - total_cost),
            "avg_yield_per_ha": round(total_yield / max(total_area, 0.01), 1),
            "roi_pct": round((total_revenue - total_cost) / max(total_cost, 1) * 100, 1),
            "crops_grown": crops_grown,
            "n_seasons": len(set(r.get("season", "") for r in records)),
        }

    def yield_trend(self, crop=None):
        """Analyze yield trends over time."""
        records = self.data["records"]
        if crop:
            records = [r for r in records if r.get("crop") == crop]
        if len(records) < 2:
            return {"message": "Need at least 2 records for trend analysis"}

        yields = [r.get("yield_per_ha", 0) for r in records]
        n = len(yields)
        mean_y = sum(yields) / n

        # Simple trend
        x_mean = (n - 1) / 2
        num = sum((i - x_mean) * (yields[i] - mean_y) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))
        slope = num / max(den, 1e-10)

        if slope > 0:
            direction = "IMPROVING"
        elif slope < -0.5:
            direction = "DECLINING"
        else:
            direction = "STABLE"

        return {
            "trend_direction": direction,
            "slope_per_season": round(slope, 2),
            "average_yield": round(mean_y, 1),
            "best_yield": round(max(yields), 1),
            "worst_yield": round(min(yields), 1),
            "n_records": n,
        }

    def add_demo_data(self):
        """Add demonstration records."""
        demos = [
            ("rice", 2.5, 11000, "kharif", 45000, 242000, "Good monsoon season"),
            ("wheat", 2.5, 9500, "rabi", 38000, 237500, "Normal winter crop"),
            ("rice", 2.5, 12000, "kharif", 48000, 264000, "Improved variety used"),
            ("soybean", 1.5, 3800, "kharif", 25000, 144400, "First time growing"),
            ("wheat", 3.0, 12000, "rabi", 45000, 300000, "Expanded area"),
            ("chickpea", 1.0, 1600, "rabi", 15000, 86400, "Short season crop"),
        ]
        for crop, area, yld, season, cost, rev, notes in demos:
            self.add_record(crop, area, yld, season, cost, rev, notes)
        return len(demos)

    def get_model_info(self):
        return {
            "name": "Farm Tracker",
            "storage": "JSON persistent database",
            "records": len(self.data["records"]),
            "db_path": self.db_path,
        }
