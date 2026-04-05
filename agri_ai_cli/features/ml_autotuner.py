"""
Sprint 7 - ML AutoTuner.
Automated hyperparameter tuning and model comparison for AgriAI models.
"""

import random
import time
import math
from agri_ai_cli.utils.ml_algorithms import (
    NaiveBayesClassifier, KNNClassifier, DecisionTreeClassifier,
    RandomForestClassifier, LinearRegression, GradientBoostingRegressor,
    train_test_split
)
from agri_ai_cli.utils.data_generator import generate_disease_data, generate_yield_data, generate_soil_data


class MLAutoTuner:
    """Automated ML model selection and hyperparameter tuning."""

    def __init__(self):
        self.results = []

    def tune_classification(self, task="disease_detection", n_samples=300):
        """Compare and tune classifiers for a given task."""
        random.seed(42)

        # Generate data based on task
        if task == "disease_detection":
            from agri_ai_cli.utils.ml_algorithms import TFIDFVectorizer
            texts, labels = generate_disease_data(n_samples)
            vectorizer = TFIDFVectorizer(max_features=150)
            X = vectorizer.fit_transform(texts)
            y = labels
            task_name = "Disease Detection"
        elif task == "soil_classification":
            X, y, _ = generate_soil_data(n_samples)
            task_name = "Soil Classification"
        else:
            X, y, _ = generate_soil_data(n_samples)
            task_name = "Classification"

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2)

        # Define model configurations to try
        configs = [
            {
                "name": "Naive Bayes",
                "model_class": NaiveBayesClassifier,
                "params": {},
            },
            {
                "name": "KNN (k=3)",
                "model_class": KNNClassifier,
                "params": {"k": 3},
            },
            {
                "name": "KNN (k=5)",
                "model_class": KNNClassifier,
                "params": {"k": 5},
            },
            {
                "name": "KNN (k=7)",
                "model_class": KNNClassifier,
                "params": {"k": 7},
            },
            {
                "name": "Decision Tree (depth=5)",
                "model_class": DecisionTreeClassifier,
                "params": {"max_depth": 5},
            },
            {
                "name": "Decision Tree (depth=10)",
                "model_class": DecisionTreeClassifier,
                "params": {"max_depth": 10},
            },
            {
                "name": "Random Forest (5 trees)",
                "model_class": RandomForestClassifier,
                "params": {"n_estimators": 5, "max_depth": 8},
            },
            {
                "name": "Random Forest (10 trees)",
                "model_class": RandomForestClassifier,
                "params": {"n_estimators": 10, "max_depth": 8},
            },
        ]

        results = []
        for config in configs:
            start = time.time()
            model = config["model_class"](**config["params"])
            model.fit(X_train, y_train)
            train_time = time.time() - start

            start = time.time()
            accuracy = model.score(X_test, y_test)
            predict_time = time.time() - start

            train_accuracy = model.score(X_train, y_train)
            overfit = train_accuracy - accuracy

            results.append({
                "model": config["name"],
                "accuracy": round(accuracy, 4),
                "train_accuracy": round(train_accuracy, 4),
                "overfit_gap": round(overfit, 4),
                "train_time_ms": round(train_time * 1000, 1),
                "predict_time_ms": round(predict_time * 1000, 1),
                "params": config["params"],
            })

        # Sort by accuracy
        results.sort(key=lambda x: x["accuracy"], reverse=True)
        best = results[0]

        return {
            "task": task_name,
            "n_samples": n_samples,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "n_models_tested": len(results),
            "results": results,
            "best_model": {
                "name": best["model"],
                "accuracy": best["accuracy"],
                "params": best["params"],
            },
            "recommendations": self._classification_recommendations(results),
        }

    def tune_regression(self, task="yield_prediction", n_samples=250):
        """Compare and tune regressors for a given task."""
        random.seed(42)

        X, y, _ = generate_yield_data(n_samples)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2)

        configs = [
            {
                "name": "Linear Regression (lr=0.01)",
                "model_class": LinearRegression,
                "params": {"learning_rate": 0.01, "epochs": 500},
            },
            {
                "name": "Linear Regression (lr=0.05)",
                "model_class": LinearRegression,
                "params": {"learning_rate": 0.05, "epochs": 500},
            },
            {
                "name": "GBR (20 trees, depth=3)",
                "model_class": GradientBoostingRegressor,
                "params": {"n_estimators": 20, "learning_rate": 0.1, "max_depth": 3},
            },
            {
                "name": "GBR (30 trees, depth=4)",
                "model_class": GradientBoostingRegressor,
                "params": {"n_estimators": 30, "learning_rate": 0.1, "max_depth": 4},
            },
            {
                "name": "GBR (50 trees, depth=3)",
                "model_class": GradientBoostingRegressor,
                "params": {"n_estimators": 50, "learning_rate": 0.05, "max_depth": 3},
            },
            {
                "name": "GBR (20 trees, depth=5)",
                "model_class": GradientBoostingRegressor,
                "params": {"n_estimators": 20, "learning_rate": 0.1, "max_depth": 5},
            },
        ]

        results = []
        for config in configs:
            start = time.time()
            model = config["model_class"](**config["params"])
            model.fit(X_train, y_train)
            train_time = time.time() - start

            start = time.time()
            r2 = model.score(X_test, y_test)
            predict_time = time.time() - start

            train_r2 = model.score(X_train, y_train)

            # Calculate RMSE
            preds = model.predict(X_test)
            rmse = math.sqrt(sum((p - a) ** 2 for p, a in zip(preds, y_test)) / len(y_test))
            mae = sum(abs(p - a) for p, a in zip(preds, y_test)) / len(y_test)

            results.append({
                "model": config["name"],
                "r2_score": round(r2, 4),
                "train_r2": round(train_r2, 4),
                "rmse": round(rmse, 1),
                "mae": round(mae, 1),
                "overfit_gap": round(train_r2 - r2, 4),
                "train_time_ms": round(train_time * 1000, 1),
                "predict_time_ms": round(predict_time * 1000, 1),
                "params": config["params"],
            })

        results.sort(key=lambda x: x["r2_score"], reverse=True)
        best = results[0]

        return {
            "task": "Yield Prediction",
            "n_samples": n_samples,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "n_models_tested": len(results),
            "results": results,
            "best_model": {
                "name": best["model"],
                "r2_score": best["r2_score"],
                "rmse": best["rmse"],
                "params": best["params"],
            },
            "recommendations": self._regression_recommendations(results),
        }

    def full_audit(self):
        """Run complete model audit across all tasks."""
        classification_result = self.tune_classification("disease_detection", 200)
        soil_result = self.tune_classification("soil_classification", 200)
        regression_result = self.tune_regression("yield_prediction", 200)

        return {
            "disease_detection": classification_result,
            "soil_classification": soil_result,
            "yield_prediction": regression_result,
            "summary": {
                "total_models_tested": (
                    classification_result["n_models_tested"] +
                    soil_result["n_models_tested"] +
                    regression_result["n_models_tested"]
                ),
                "best_disease_model": classification_result["best_model"]["name"],
                "best_disease_accuracy": classification_result["best_model"]["accuracy"],
                "best_soil_model": soil_result["best_model"]["name"],
                "best_soil_accuracy": soil_result["best_model"]["accuracy"],
                "best_yield_model": regression_result["best_model"]["name"],
                "best_yield_r2": regression_result["best_model"]["r2_score"],
            },
        }

    def _classification_recommendations(self, results):
        recs = []
        best = results[0]
        if best["overfit_gap"] > 0.15:
            recs.append(f"Warning: {best['model']} shows overfitting (gap: {best['overfit_gap']}). "
                       "Consider reducing model complexity or adding more data.")
        if best["accuracy"] < 0.7:
            recs.append("Accuracy below 70%. Consider feature engineering or collecting more training data.")
        if best["accuracy"] >= 0.85:
            recs.append(f"Good accuracy ({best['accuracy']}). Model is production-ready.")

        # Check if ensemble is better
        ensemble_results = [r for r in results if "Forest" in r["model"]]
        simple_results = [r for r in results if "Forest" not in r["model"]]
        if ensemble_results and simple_results:
            best_ensemble = max(ensemble_results, key=lambda x: x["accuracy"])
            best_simple = max(simple_results, key=lambda x: x["accuracy"])
            if best_ensemble["accuracy"] > best_simple["accuracy"]:
                recs.append(f"Ensemble methods ({best_ensemble['model']}) outperform simpler models. "
                           "Worth the extra computation time.")
            else:
                recs.append(f"Simple model ({best_simple['model']}) performs comparably. "
                           "Prefer for faster inference.")
        return recs

    def _regression_recommendations(self, results):
        recs = []
        best = results[0]
        if best["r2_score"] < 0.5:
            recs.append("Low R2 score. Feature engineering or non-linear models may help.")
        elif best["r2_score"] >= 0.8:
            recs.append(f"Strong R2 score ({best['r2_score']}). Model captures yield patterns well.")

        if best["overfit_gap"] > 0.2:
            recs.append("Significant overfitting detected. Reduce tree depth or add regularization.")

        # Speed vs accuracy tradeoff
        fastest = min(results, key=lambda x: x["train_time_ms"])
        if fastest["model"] != best["model"]:
            speed_ratio = best["train_time_ms"] / max(fastest["train_time_ms"], 0.1)
            acc_diff = best["r2_score"] - fastest["r2_score"]
            if acc_diff < 0.05 and speed_ratio > 3:
                recs.append(f"Consider {fastest['model']} -- similar accuracy but {speed_ratio:.1f}x faster.")
        return recs

    def get_model_info(self):
        return {
            "name": "ML AutoTuner",
            "algorithm": "Grid Search + Model Comparison",
            "classifiers_available": 4,
            "regressors_available": 2,
        }
