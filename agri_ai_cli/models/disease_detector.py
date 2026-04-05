"""
Sprint 1 - Disease Detection Model.
Uses Random Forest + TF-IDF to classify crop diseases from symptom descriptions.
"""

import random
from agri_ai_cli.utils.ml_algorithms import (
    RandomForestClassifier, TFIDFVectorizer, train_test_split,
    classification_report
)
from agri_ai_cli.utils.data_generator import generate_disease_data, DISEASE_KNOWLEDGE


class DiseaseDetector:
    """AI-powered crop disease detection from symptom text."""

    def __init__(self):
        self.vectorizer = TFIDFVectorizer(max_features=200)
        self.model = RandomForestClassifier(n_estimators=8, max_depth=10)
        self.is_trained = False
        self.accuracy = 0.0
        self.diseases = list(DISEASE_KNOWLEDGE.keys())

    def train(self, n_samples=400):
        """Train the disease detection model on synthetic data."""
        random.seed(42)
        texts, labels = generate_disease_data(n_samples)

        # Vectorize text
        X = self.vectorizer.fit_transform(texts)

        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_ratio=0.2)
        self.model.fit(X_train, y_train)
        self.accuracy = self.model.score(X_test, y_test)
        self.is_trained = True

        return {
            "accuracy": self.accuracy,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "n_diseases": len(set(labels)),
            "vocabulary_size": len(self.vectorizer.vocabulary),
        }

    def predict(self, symptom_text):
        """Predict disease from symptom description."""
        if not self.is_trained:
            self.train()

        X = self.vectorizer.transform([symptom_text])
        prediction = self.model.predict(X)[0]
        probas = self.model.predict_proba(X)[0]

        # Sort by probability
        sorted_probs = sorted(probas.items(), key=lambda x: x[1], reverse=True)

        # Get disease info
        info = DISEASE_KNOWLEDGE.get(prediction, {})
        crop = info.get("crop", "unknown")
        severity = info.get("severity_range", (1, 5))
        avg_severity = sum(severity) / 2

        return {
            "disease": prediction,
            "disease_name": prediction.replace("_", " ").title(),
            "crop": crop,
            "confidence": sorted_probs[0][1] if sorted_probs else 0.5,
            "severity": avg_severity,
            "severity_label": self._severity_label(avg_severity),
            "top_predictions": [
                {"disease": d.replace("_", " ").title(), "probability": round(p, 3)}
                for d, p in sorted_probs[:5]
            ],
            "treatment": self._get_treatment(prediction),
            "prevention": self._get_prevention(prediction),
        }

    def _severity_label(self, score):
        if score >= 8:
            return "CRITICAL"
        elif score >= 6:
            return "HIGH"
        elif score >= 4:
            return "MODERATE"
        else:
            return "LOW"

    def _get_treatment(self, disease):
        treatments = {
            "rice_blast": [
                "Apply tricyclazole fungicide at 0.6g/L",
                "Use resistant varieties (Tetep, Tadukan)",
                "Drain fields and reduce nitrogen application",
            ],
            "rice_brown_spot": [
                "Seed treatment with carbendazim",
                "Balanced NPK fertilization",
                "Apply mancozeb at first symptom appearance",
            ],
            "wheat_rust": [
                "Apply propiconazole fungicide immediately",
                "Use rust-resistant wheat varieties",
                "Early sowing to escape rust window",
            ],
            "wheat_powdery_mildew": [
                "Sulfur dust application at 25 kg/ha",
                "Triadimefon spray at 0.1%",
                "Avoid excess nitrogen fertilization",
            ],
            "corn_leaf_blight": [
                "Apply azoxystrobin + propiconazole",
                "Use tolerant hybrids",
                "Practice crop rotation with non-grass crops",
            ],
            "corn_common_rust": [
                "Foliar fungicide at first pustule appearance",
                "Plant resistant hybrids",
                "Scout fields weekly during warm humid weather",
            ],
            "tomato_early_blight": [
                "Apply chlorothalonil or copper fungicide",
                "Mulch to prevent soil splash",
                "Remove lower infected leaves promptly",
            ],
            "tomato_late_blight": [
                "Apply metalaxyl + mancozeb immediately",
                "Destroy all infected plant material",
                "Improve air circulation between plants",
            ],
            "potato_late_blight": [
                "Prophylactic mancozeb sprays in wet weather",
                "Use certified disease-free seed potatoes",
                "Hill potatoes to protect tubers",
            ],
            "cotton_bacterial_blight": [
                "Seed treatment with streptocycline",
                "Use disease-free certified seeds",
                "Copper oxychloride spray at 3g/L",
            ],
            "soybean_rust": [
                "Apply triazole fungicides at R3-R4 stage",
                "Plant early-maturing varieties",
                "Scout regularly from flowering onward",
            ],
        }
        return treatments.get(disease, ["Consult local agricultural extension officer"])

    def _get_prevention(self, disease):
        preventions = {
            "rice_blast": ["Avoid excess nitrogen", "Maintain proper water management",
                          "Use silicon-based fertilizers"],
            "wheat_rust": ["Early sowing", "Grow resistant varieties", "Remove volunteer wheat"],
            "tomato_late_blight": ["Adequate spacing", "Avoid overhead irrigation",
                                   "Monitor weather forecasts"],
        }
        return preventions.get(disease, [
            "Practice crop rotation",
            "Use disease-free seeds",
            "Maintain field hygiene",
        ])

    def get_model_info(self):
        return {
            "name": "Disease Detector",
            "algorithm": "Random Forest + TF-IDF",
            "n_diseases": len(self.diseases),
            "trained": self.is_trained,
            "accuracy": round(self.accuracy, 3) if self.is_trained else None,
        }
