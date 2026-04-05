"""
Sprint 3 - Market Price Analyzer.
Time-series analysis and price forecasting for agricultural commodities.
"""

import math
import random
from agri_ai_cli.utils.data_generator import generate_market_data


class MarketAnalyzer:
    """Agricultural commodity price analysis and forecasting engine."""

    MSP_PRICES = {
        "rice": 2183, "wheat": 2275, "corn": 2090, "cotton": 6620,
        "soybean": 4600, "groundnut": 6377, "lentil": 6425,
        "mustard": 5650, "sugarcane": 315, "chickpea": 5440,
    }

    def analyze(self, crop="rice", months_history=24):
        """Perform comprehensive market analysis for a crop."""
        prices = generate_market_data(crop, months_history)

        # Basic statistics
        stats = self._compute_stats(prices)

        # Trend analysis
        trend = self._trend_analysis(prices)

        # Seasonality detection
        seasonality = self._seasonality(prices)

        # Price forecast (next 6 months)
        forecast = self._forecast(prices, months_ahead=6)

        # Volatility
        volatility = self._volatility(prices)

        # MSP comparison
        msp = self.MSP_PRICES.get(crop, 0)

        return {
            "crop": crop,
            "statistics": stats,
            "trend": trend,
            "seasonality": seasonality,
            "forecast": forecast,
            "volatility": volatility,
            "msp_inr_quintal": msp,
            "current_vs_msp": round(stats["current"] / max(msp / 100, 1) * 100, 1) if msp else None,
            "trading_signal": self._trading_signal(trend, forecast, stats),
            "market_recommendations": self._recommendations(crop, trend, forecast, stats),
        }

    def _compute_stats(self, prices):
        n = len(prices)
        mean = sum(prices) / n
        sorted_p = sorted(prices)
        median = sorted_p[n // 2]
        min_p = min(prices)
        max_p = max(prices)
        std = math.sqrt(sum((p - mean) ** 2 for p in prices) / max(n - 1, 1))

        return {
            "current": round(prices[-1], 2),
            "mean": round(mean, 2),
            "median": round(median, 2),
            "min": round(min_p, 2),
            "max": round(max_p, 2),
            "std_dev": round(std, 2),
            "range": round(max_p - min_p, 2),
            "n_months": n,
        }

    def _trend_analysis(self, prices):
        """Simple linear trend using least squares."""
        n = len(prices)
        x_mean = (n - 1) / 2
        y_mean = sum(prices) / n

        numerator = sum((i - x_mean) * (prices[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        slope = numerator / max(denominator, 1e-10)
        intercept = y_mean - slope * x_mean

        # Monthly change rate
        monthly_change_pct = (slope / max(abs(y_mean), 1e-10)) * 100

        if monthly_change_pct > 1:
            direction = "RISING"
        elif monthly_change_pct < -1:
            direction = "FALLING"
        else:
            direction = "STABLE"

        return {
            "direction": direction,
            "slope_per_month": round(slope, 3),
            "monthly_change_pct": round(monthly_change_pct, 2),
            "intercept": round(intercept, 2),
            "trend_strength": round(abs(monthly_change_pct), 2),
        }

    def _seasonality(self, prices):
        """Detect seasonal patterns (monthly averages)."""
        monthly_avgs = {}
        for i, p in enumerate(prices):
            month = i % 12
            if month not in monthly_avgs:
                monthly_avgs[month] = []
            monthly_avgs[month].append(p)

        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        seasonal_pattern = {}
        overall_avg = sum(prices) / len(prices)

        for month_idx in sorted(monthly_avgs.keys()):
            avg = sum(monthly_avgs[month_idx]) / len(monthly_avgs[month_idx])
            seasonal_pattern[month_names[month_idx]] = {
                "average": round(avg, 2),
                "index": round(avg / overall_avg * 100, 1),
            }

        # Find best and worst months
        best_month = max(seasonal_pattern.items(), key=lambda x: x[1]["average"])
        worst_month = min(seasonal_pattern.items(), key=lambda x: x[1]["average"])

        return {
            "monthly_pattern": seasonal_pattern,
            "best_month": {"month": best_month[0], "avg_price": best_month[1]["average"]},
            "worst_month": {"month": worst_month[0], "avg_price": worst_month[1]["average"]},
            "seasonal_amplitude_pct": round(
                (best_month[1]["average"] - worst_month[1]["average"]) / overall_avg * 100, 1),
        }

    def _forecast(self, prices, months_ahead=6):
        """Simple exponential smoothing forecast."""
        alpha = 0.3  # smoothing factor
        n = len(prices)

        # Exponential smoothing
        smoothed = [prices[0]]
        for i in range(1, n):
            smoothed.append(alpha * prices[i] + (1 - alpha) * smoothed[-1])

        # Trend component
        trend_val = (smoothed[-1] - smoothed[max(0, -6)]) / min(6, n)

        forecasted = []
        last = smoothed[-1]
        for i in range(1, months_ahead + 1):
            pred = last + trend_val * i
            # Add seasonal component
            seasonal_idx = (n + i) % 12
            seasonal_factor = math.sin(2 * math.pi * seasonal_idx / 12) * sum(prices) / n * 0.05
            pred += seasonal_factor
            confidence = max(0.5, 1 - i * 0.08)
            forecasted.append({
                "month_ahead": i,
                "predicted_price": round(pred, 2),
                "confidence": round(confidence, 2),
                "range_low": round(pred * 0.9, 2),
                "range_high": round(pred * 1.1, 2),
            })

        return forecasted

    def _volatility(self, prices):
        """Calculate price volatility metrics."""
        if len(prices) < 2:
            return {"daily_volatility": 0, "annualized": 0, "risk_level": "LOW"}

        returns = [(prices[i] - prices[i - 1]) / max(abs(prices[i - 1]), 1e-10)
                   for i in range(1, len(prices))]
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / max(len(returns) - 1, 1)
        monthly_vol = math.sqrt(variance)
        annualized = monthly_vol * math.sqrt(12)

        if annualized > 0.3:
            risk = "HIGH"
        elif annualized > 0.15:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        return {
            "monthly_volatility": round(monthly_vol * 100, 2),
            "annualized_volatility": round(annualized * 100, 2),
            "risk_level": risk,
            "max_drawdown_pct": round(self._max_drawdown(prices), 2),
        }

    def _max_drawdown(self, prices):
        peak = prices[0]
        max_dd = 0
        for p in prices:
            if p > peak:
                peak = p
            dd = (peak - p) / peak * 100
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def _trading_signal(self, trend, forecast, stats):
        if trend["direction"] == "RISING" and forecast[0]["predicted_price"] > stats["current"]:
            return {"signal": "HOLD", "reason": "Prices trending upward - wait for peak"}
        elif trend["direction"] == "FALLING":
            return {"signal": "SELL", "reason": "Prices declining - sell current stock"}
        else:
            return {"signal": "HOLD", "reason": "Market stable - sell at convenience"}

    def _recommendations(self, crop, trend, forecast, stats):
        recs = []
        if trend["direction"] == "RISING":
            recs.append(f"Prices for {crop} are trending upward at {trend['monthly_change_pct']}%/month")
            recs.append("Consider holding produce for 1-2 months for better returns")
        elif trend["direction"] == "FALLING":
            recs.append(f"Prices for {crop} are declining at {abs(trend['monthly_change_pct'])}%/month")
            recs.append("Sell current stock soon to minimize losses")
        else:
            recs.append(f"Market for {crop} is relatively stable")

        if forecast[0]["predicted_price"] > stats["current"] * 1.05:
            recs.append(f"Short-term forecast suggests prices may rise to {forecast[0]['predicted_price']}")
        elif forecast[0]["predicted_price"] < stats["current"] * 0.95:
            recs.append(f"Short-term forecast suggests prices may drop to {forecast[0]['predicted_price']}")

        return recs

    def get_model_info(self):
        return {
            "name": "Market Analyzer",
            "algorithm": "Exponential Smoothing + Linear Trend",
            "supported_crops": list(self.MSP_PRICES.keys()),
            "forecast_horizon": "6 months",
        }
