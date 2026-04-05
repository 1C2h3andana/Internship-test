"""
Sprint 6 - Farm Financial Ledger.
Complete farm financial tracking with income, expenses, profit analysis, and ROI.
"""

import json
import os
import uuid
from datetime import datetime


class FinancialLedger:
    """Farm financial management and analysis system."""

    EXPENSE_CATEGORIES = [
        "seeds", "fertilizer", "pesticide", "irrigation", "labor",
        "machinery", "transport", "land_rent", "insurance", "miscellaneous",
    ]

    INCOME_CATEGORIES = [
        "crop_sale", "subsidy", "insurance_claim", "byproduct_sale", "other",
    ]

    def __init__(self, db_path="agri_ai_cli/data/financial_ledger.json"):
        self.db_path = db_path
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {
            "expenses": [],
            "income": [],
            "metadata": {"created": datetime.now().isoformat(), "version": "3.0"},
        }

    def _save(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "w") as f:
            json.dump(self.data, f, indent=2, default=str)

    def add_expense(self, amount, category, description="", crop="general", season=""):
        """Record a farm expense."""
        if category not in self.EXPENSE_CATEGORIES:
            category = "miscellaneous"
        entry = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now().isoformat(),
            "type": "expense",
            "amount": amount,
            "category": category,
            "description": description,
            "crop": crop,
            "season": season,
        }
        self.data["expenses"].append(entry)
        self._save()
        return entry

    def add_income(self, amount, category, description="", crop="general", season=""):
        """Record farm income."""
        if category not in self.INCOME_CATEGORIES:
            category = "other"
        entry = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now().isoformat(),
            "type": "income",
            "amount": amount,
            "category": category,
            "description": description,
            "crop": crop,
            "season": season,
        }
        self.data["income"].append(entry)
        self._save()
        return entry

    def get_summary(self, crop=None, season=None):
        """Get financial summary with filtering."""
        expenses = self.data["expenses"]
        income = self.data["income"]

        if crop:
            expenses = [e for e in expenses if e.get("crop") == crop]
            income = [i for i in income if i.get("crop") == crop]
        if season:
            expenses = [e for e in expenses if e.get("season") == season]
            income = [i for i in income if i.get("season") == season]

        total_expenses = sum(e["amount"] for e in expenses)
        total_income = sum(i["amount"] for i in income)
        profit = total_income - total_expenses
        roi = (profit / max(total_expenses, 1)) * 100

        # Expense breakdown
        expense_breakdown = {}
        for e in expenses:
            cat = e["category"]
            expense_breakdown[cat] = expense_breakdown.get(cat, 0) + e["amount"]

        # Income breakdown
        income_breakdown = {}
        for i in income:
            cat = i["category"]
            income_breakdown[cat] = income_breakdown.get(cat, 0) + i["amount"]

        return {
            "total_income_inr": round(total_income),
            "total_expenses_inr": round(total_expenses),
            "net_profit_inr": round(profit),
            "roi_pct": round(roi, 1),
            "profit_status": "PROFIT" if profit > 0 else "LOSS",
            "expense_breakdown": expense_breakdown,
            "income_breakdown": income_breakdown,
            "n_transactions": len(expenses) + len(income),
            "cost_per_rupee_earned": round(total_expenses / max(total_income, 1), 2),
            "expense_analysis": self._expense_analysis(expense_breakdown, total_expenses),
        }

    def _expense_analysis(self, breakdown, total):
        """Analyze expense patterns and suggest optimizations."""
        analysis = []
        for cat, amount in sorted(breakdown.items(), key=lambda x: x[1], reverse=True):
            pct = amount / max(total, 1) * 100
            analysis.append({
                "category": cat,
                "amount_inr": round(amount),
                "percentage": round(pct, 1),
                "assessment": self._assess_expense(cat, pct),
            })
        return analysis

    def _assess_expense(self, category, pct):
        thresholds = {
            "fertilizer": (15, 25),
            "labor": (20, 35),
            "seeds": (5, 15),
            "pesticide": (5, 15),
            "irrigation": (10, 20),
            "machinery": (10, 20),
            "land_rent": (15, 30),
        }
        low, high = thresholds.get(category, (5, 20))
        if pct > high:
            return f"HIGH - {category} cost above typical range ({high}%). Review for optimization."
        elif pct < low:
            return f"LOW - Check if {category} investment is adequate."
        return "NORMAL - Within expected range."

    def get_crop_profitability(self):
        """Compare profitability across crops."""
        crops = set()
        for e in self.data["expenses"]:
            crops.add(e.get("crop", "general"))
        for i in self.data["income"]:
            crops.add(i.get("crop", "general"))

        results = []
        for crop in sorted(crops):
            if crop == "general":
                continue
            summary = self.get_summary(crop=crop)
            results.append({
                "crop": crop,
                "income": summary["total_income_inr"],
                "expenses": summary["total_expenses_inr"],
                "profit": summary["net_profit_inr"],
                "roi_pct": summary["roi_pct"],
            })

        return sorted(results, key=lambda x: x["roi_pct"], reverse=True)

    def add_demo_data(self):
        """Add demonstration financial data."""
        # Rice season expenses
        demos_expense = [
            (12000, "seeds", "Hybrid rice seeds 25kg", "rice", "kharif"),
            (18000, "fertilizer", "Urea 200kg + DAP 100kg + MOP 50kg", "rice", "kharif"),
            (5000, "pesticide", "Fungicide + insecticide sprays", "rice", "kharif"),
            (8000, "irrigation", "Diesel for pump 3 months", "rice", "kharif"),
            (25000, "labor", "Transplanting + weeding + harvest labor", "rice", "kharif"),
            (6000, "machinery", "Tractor tillage + thresher hire", "rice", "kharif"),
            (3000, "transport", "Market transport 2 trips", "rice", "kharif"),
            (8000, "seeds", "Wheat seeds 40kg", "wheat", "rabi"),
            (15000, "fertilizer", "Urea + DAP for wheat", "wheat", "rabi"),
            (3000, "pesticide", "Herbicide + fungicide", "wheat", "rabi"),
            (6000, "irrigation", "4 irrigations", "wheat", "rabi"),
            (15000, "labor", "Sowing + harvest labor", "wheat", "rabi"),
            (5000, "machinery", "Combine harvester hire", "wheat", "rabi"),
        ]
        demos_income = [
            (220000, "crop_sale", "Rice 10 tons at 22/kg", "rice", "kharif"),
            (6000, "subsidy", "PM-KISAN installment", "general", "kharif"),
            (190000, "crop_sale", "Wheat 7.6 tons at 25/kg", "wheat", "rabi"),
            (6000, "subsidy", "PM-KISAN installment", "general", "rabi"),
            (12000, "byproduct_sale", "Straw sale", "wheat", "rabi"),
        ]

        for amount, cat, desc, crop, season in demos_expense:
            self.add_expense(amount, cat, desc, crop, season)
        for amount, cat, desc, crop, season in demos_income:
            self.add_income(amount, cat, desc, crop, season)

        return len(demos_expense) + len(demos_income)

    def get_model_info(self):
        return {
            "name": "Financial Ledger",
            "storage": "JSON persistent database",
            "expense_categories": len(self.EXPENSE_CATEGORIES),
            "income_categories": len(self.INCOME_CATEGORIES),
            "transactions": len(self.data["expenses"]) + len(self.data["income"]),
        }
