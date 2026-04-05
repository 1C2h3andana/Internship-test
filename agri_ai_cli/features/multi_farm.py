"""
Sprint 7 - Multi-Farm Management System.
Manage and compare multiple farm units with aggregated analytics.
"""

import json
import os
import uuid
from datetime import datetime


class MultiFarmManager:
    """Multi-farm portfolio management and comparison system."""

    def __init__(self, db_path="agri_ai_cli/data/multi_farm.json"):
        self.db_path = db_path
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {"farms": {}, "metadata": {"created": datetime.now().isoformat()}}

    def _save(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "w") as f:
            json.dump(self.data, f, indent=2, default=str)

    def add_farm(self, name, area_ha, location, soil_type, primary_crop,
                 water_source="canal", irrigation_type="flood"):
        """Register a new farm unit."""
        farm_id = str(uuid.uuid4())[:8]
        farm = {
            "id": farm_id,
            "name": name,
            "area_ha": area_ha,
            "location": location,
            "soil_type": soil_type,
            "primary_crop": primary_crop,
            "water_source": water_source,
            "irrigation_type": irrigation_type,
            "created": datetime.now().isoformat(),
            "seasons": [],
        }
        self.data["farms"][farm_id] = farm
        self._save()
        return farm

    def add_season_record(self, farm_id, crop, season, yield_kg, cost_inr, revenue_inr):
        """Add a seasonal performance record to a farm."""
        if farm_id not in self.data["farms"]:
            return {"error": f"Farm {farm_id} not found"}

        record = {
            "season": season,
            "crop": crop,
            "yield_kg": yield_kg,
            "cost_inr": cost_inr,
            "revenue_inr": revenue_inr,
            "profit_inr": revenue_inr - cost_inr,
            "timestamp": datetime.now().isoformat(),
        }
        self.data["farms"][farm_id]["seasons"].append(record)
        self._save()
        return record

    def get_farm(self, farm_id):
        """Get details of a specific farm."""
        return self.data["farms"].get(farm_id)

    def list_farms(self):
        """List all registered farms."""
        farms = []
        for fid, farm in self.data["farms"].items():
            total_profit = sum(s.get("profit_inr", 0) for s in farm.get("seasons", []))
            farms.append({
                "id": fid,
                "name": farm["name"],
                "area_ha": farm["area_ha"],
                "location": farm["location"],
                "primary_crop": farm["primary_crop"],
                "n_seasons": len(farm.get("seasons", [])),
                "total_profit_inr": total_profit,
            })
        return farms

    def compare_farms(self):
        """Compare performance across all farms."""
        comparisons = []
        for fid, farm in self.data["farms"].items():
            seasons = farm.get("seasons", [])
            if not seasons:
                continue

            total_yield = sum(s.get("yield_kg", 0) for s in seasons)
            total_cost = sum(s.get("cost_inr", 0) for s in seasons)
            total_revenue = sum(s.get("revenue_inr", 0) for s in seasons)
            area = farm.get("area_ha", 1)

            comparisons.append({
                "farm_name": farm["name"],
                "farm_id": fid,
                "area_ha": area,
                "avg_yield_per_ha": round(total_yield / max(area * len(seasons), 1), 1),
                "avg_profit_per_ha": round((total_revenue - total_cost) / max(area * len(seasons), 1), 0),
                "total_profit": round(total_revenue - total_cost),
                "roi_pct": round((total_revenue - total_cost) / max(total_cost, 1) * 100, 1),
                "n_seasons": len(seasons),
                "soil_type": farm.get("soil_type", "unknown"),
                "irrigation": farm.get("irrigation_type", "unknown"),
            })

        return sorted(comparisons, key=lambda x: x["avg_profit_per_ha"], reverse=True)

    def portfolio_summary(self):
        """Get aggregated portfolio summary."""
        farms = self.data["farms"]
        if not farms:
            return {"message": "No farms registered. Add farms to see portfolio summary."}

        total_area = sum(f.get("area_ha", 0) for f in farms.values())
        all_seasons = []
        for f in farms.values():
            all_seasons.extend(f.get("seasons", []))

        total_cost = sum(s.get("cost_inr", 0) for s in all_seasons)
        total_revenue = sum(s.get("revenue_inr", 0) for s in all_seasons)
        total_yield = sum(s.get("yield_kg", 0) for s in all_seasons)

        crops = list(set(s.get("crop", "") for s in all_seasons))
        soil_types = list(set(f.get("soil_type", "") for f in farms.values()))

        return {
            "n_farms": len(farms),
            "total_area_ha": round(total_area, 2),
            "crops_grown": crops,
            "soil_types": soil_types,
            "total_seasons_recorded": len(all_seasons),
            "total_yield_kg": round(total_yield, 1),
            "total_cost_inr": round(total_cost),
            "total_revenue_inr": round(total_revenue),
            "total_profit_inr": round(total_revenue - total_cost),
            "portfolio_roi_pct": round((total_revenue - total_cost) / max(total_cost, 1) * 100, 1),
            "avg_yield_per_ha": round(total_yield / max(total_area, 0.01), 1),
        }

    def add_demo_data(self):
        """Add demo multi-farm data."""
        farms = [
            ("North Field", 3.5, "Village Rampur", "alluvial", "rice", "canal", "flood"),
            ("South Plot", 2.0, "Village Rampur", "clay", "wheat", "borewell", "sprinkler"),
            ("East Farm", 5.0, "Town Bhiwani", "black", "cotton", "canal", "drip"),
            ("Kitchen Garden", 0.5, "Village Rampur", "alluvial", "tomato", "borewell", "drip"),
        ]
        farm_ids = []
        for name, area, loc, soil, crop, water, irrig in farms:
            f = self.add_farm(name, area, loc, soil, crop, water, irrig)
            farm_ids.append(f["id"])

        # Add season records
        season_data = [
            (farm_ids[0], "rice", "kharif_2024", 15000, 55000, 330000),
            (farm_ids[0], "wheat", "rabi_2024", 12000, 42000, 300000),
            (farm_ids[1], "wheat", "rabi_2024", 7600, 32000, 190000),
            (farm_ids[1], "chickpea", "rabi_2025", 2800, 18000, 126000),
            (farm_ids[2], "cotton", "kharif_2024", 9000, 85000, 495000),
            (farm_ids[2], "wheat", "rabi_2024", 15000, 55000, 375000),
            (farm_ids[3], "tomato", "zaid_2024", 15000, 12000, 450000),
        ]
        for fid, crop, season, yld, cost, rev in season_data:
            self.add_season_record(fid, crop, season, yld, cost, rev)

        return len(farms)

    def get_model_info(self):
        return {
            "name": "Multi-Farm Manager",
            "storage": "JSON persistent database",
            "n_farms": len(self.data["farms"]),
        }
