"""
Sprint 4 - Seasonal Calendar.
Agricultural calendar with crop scheduling and government scheme information.
"""


class SeasonalCalendar:
    """Agricultural seasonal calendar and activity planner."""

    SEASONS = {
        "kharif": {
            "months": "June - October",
            "start_month": 6,
            "end_month": 10,
            "description": "Monsoon season crops",
            "crops": {
                "rice": {"sow": "Jun", "transplant": "Jul", "harvest": "Oct-Nov"},
                "corn": {"sow": "Jun-Jul", "harvest": "Sep-Oct"},
                "soybean": {"sow": "Jun-Jul", "harvest": "Oct"},
                "cotton": {"sow": "May-Jun", "harvest": "Nov-Dec"},
                "groundnut": {"sow": "Jun-Jul", "harvest": "Oct"},
                "greengram": {"sow": "Jul", "harvest": "Sep"},
                "sugarcane": {"sow": "Feb-Mar", "harvest": "Jan-Feb (next year)"},
            },
        },
        "rabi": {
            "months": "November - March",
            "start_month": 11,
            "end_month": 3,
            "description": "Winter season crops",
            "crops": {
                "wheat": {"sow": "Nov", "harvest": "Mar-Apr"},
                "chickpea": {"sow": "Oct-Nov", "harvest": "Mar"},
                "lentil": {"sow": "Oct-Nov", "harvest": "Feb-Mar"},
                "mustard": {"sow": "Oct", "harvest": "Feb"},
                "potato": {"sow": "Oct-Nov", "harvest": "Feb"},
                "onion": {"sow": "Nov-Dec", "harvest": "Apr-May"},
            },
        },
        "zaid": {
            "months": "April - May",
            "start_month": 4,
            "end_month": 5,
            "description": "Summer season crops",
            "crops": {
                "greengram": {"sow": "Mar-Apr", "harvest": "May-Jun"},
                "sunflower": {"sow": "Mar", "harvest": "Jun"},
                "tomato": {"sow": "Mar-Apr", "harvest": "Jun-Jul"},
                "watermelon": {"sow": "Feb-Mar", "harvest": "May-Jun"},
                "cucumber": {"sow": "Mar-Apr", "harvest": "Jun"},
            },
        },
    }

    GOV_SCHEMES = [
        {
            "name": "PM-KISAN",
            "full_name": "Pradhan Mantri Kisan Samman Nidhi",
            "benefit": "INR 6,000 per year in 3 installments",
            "eligibility": "All landholding farmer families",
            "deadline": "Rolling enrollment",
        },
        {
            "name": "PMFBY",
            "full_name": "Pradhan Mantri Fasal Bima Yojana",
            "benefit": "Crop insurance at 1.5-5% premium",
            "eligibility": "All farmers growing notified crops",
            "deadline": "Before sowing season cutoff",
        },
        {
            "name": "KCC",
            "full_name": "Kisan Credit Card",
            "benefit": "Short-term credit at 4% interest (with subvention)",
            "eligibility": "All farmers, sharecroppers, tenant farmers",
            "deadline": "Apply anytime at bank",
        },
        {
            "name": "Soil Health Card",
            "full_name": "Soil Health Card Scheme",
            "benefit": "Free soil testing and fertilizer recommendations",
            "eligibility": "All farmers",
            "deadline": "Apply at local agriculture office",
        },
        {
            "name": "eNAM",
            "full_name": "National Agriculture Market",
            "benefit": "Online trading platform for better prices",
            "eligibility": "All farmers with produce to sell",
            "deadline": "Register at nearest APMC mandi",
        },
        {
            "name": "PKVY",
            "full_name": "Paramparagat Krishi Vikas Yojana",
            "benefit": "INR 50,000/ha over 3 years for organic farming",
            "eligibility": "Farmer groups (min 50 farmers, 50 acres)",
            "deadline": "Apply through state agriculture dept",
        },
    ]

    def get_calendar(self, month=None):
        """Get seasonal calendar for current or specified month."""
        if month is None:
            from datetime import datetime
            month = datetime.now().month

        current_season = self._get_season(month)
        activities = self._get_activities(month)
        upcoming = self._upcoming_activities(month)

        return {
            "current_month": month,
            "month_name": self._month_name(month),
            "current_season": current_season,
            "season_info": self.SEASONS.get(current_season, {}),
            "current_activities": activities,
            "upcoming_activities": upcoming,
            "schemes_available": self._relevant_schemes(current_season),
        }

    def get_crop_calendar(self, crop):
        """Get full calendar for a specific crop."""
        for season_name, season_data in self.SEASONS.items():
            if crop in season_data["crops"]:
                crop_schedule = season_data["crops"][crop]
                return {
                    "crop": crop,
                    "season": season_name,
                    "season_months": season_data["months"],
                    "schedule": crop_schedule,
                    "activities": self._crop_activities(crop, season_name),
                    "input_calendar": self._input_schedule(crop),
                }
        return {"crop": crop, "message": f"Crop '{crop}' not found in calendar database"}

    def _get_season(self, month):
        if month in (6, 7, 8, 9, 10):
            return "kharif"
        elif month in (11, 12, 1, 2, 3):
            return "rabi"
        return "zaid"

    def _month_name(self, month):
        names = ["", "January", "February", "March", "April", "May", "June",
                 "July", "August", "September", "October", "November", "December"]
        return names[month] if 1 <= month <= 12 else "Unknown"

    def _get_activities(self, month):
        activities = []
        month_abbr = self._month_name(month)[:3]

        for season_name, season_data in self.SEASONS.items():
            for crop, schedule in season_data["crops"].items():
                for activity, timing in schedule.items():
                    if month_abbr in timing:
                        activities.append({
                            "crop": crop,
                            "activity": activity,
                            "timing": timing,
                            "season": season_name,
                        })
        return activities

    def _upcoming_activities(self, month):
        next_month = (month % 12) + 1
        next_next = (next_month % 12) + 1
        activities = []
        for m in [next_month, next_next]:
            acts = self._get_activities(m)
            for a in acts:
                a["month"] = self._month_name(m)
                activities.append(a)
        return activities

    def _relevant_schemes(self, season):
        # All schemes are generally available, but some are more relevant
        return self.GOV_SCHEMES

    def _crop_activities(self, crop, season):
        """Detailed activity timeline for a crop."""
        timelines = {
            "rice": [
                {"week": "W1-2", "activity": "Nursery preparation and seed soaking"},
                {"week": "W3-4", "activity": "Transplanting seedlings to main field"},
                {"week": "W5-8", "activity": "Tillering and vegetative growth management"},
                {"week": "W9-12", "activity": "Panicle initiation and flowering"},
                {"week": "W13-16", "activity": "Grain filling and maturity"},
                {"week": "W17-18", "activity": "Harvesting and post-harvest drying"},
            ],
            "wheat": [
                {"week": "W1-2", "activity": "Field preparation and sowing"},
                {"week": "W3-4", "activity": "Germination and crown root initiation"},
                {"week": "W5-8", "activity": "Tillering phase - first irrigation"},
                {"week": "W9-12", "activity": "Jointing and booting stage"},
                {"week": "W13-16", "activity": "Heading and flowering"},
                {"week": "W17-20", "activity": "Grain filling, maturity, harvest"},
            ],
        }
        return timelines.get(crop, [
            {"week": "W1-4", "activity": "Land preparation and sowing"},
            {"week": "W5-8", "activity": "Vegetative growth management"},
            {"week": "W9-12", "activity": "Reproductive phase"},
            {"week": "W13+", "activity": "Maturity and harvest"},
        ])

    def _input_schedule(self, crop):
        """Recommended input application schedule."""
        return {
            "fertilizer": [
                {"timing": "Basal", "application": "50% N + full P + full K at sowing"},
                {"timing": "30 DAS", "application": "25% N as top dressing"},
                {"timing": "60 DAS", "application": "25% N as top dressing"},
            ],
            "irrigation": "As per soil moisture monitoring (see Irrigation Scheduler)",
            "pest_monitoring": "Weekly scouting from 2 weeks after emergence",
            "weed_management": "Pre-emergence herbicide at sowing + manual weeding at 30 DAS",
        }

    def get_model_info(self):
        return {
            "name": "Seasonal Calendar",
            "seasons": 3,
            "crops_tracked": sum(len(s["crops"]) for s in self.SEASONS.values()),
            "gov_schemes": len(self.GOV_SCHEMES),
        }
