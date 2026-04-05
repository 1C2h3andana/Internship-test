"""
Sprint 2 - Pest Management Engine.
Decision Tree classifier with ETL pipeline for Integrated Pest Management (IPM).
"""

import random
from agri_ai_cli.utils.ml_algorithms import DecisionTreeClassifier, train_test_split
from agri_ai_cli.utils.data_generator import generate_pest_data, PEST_KNOWLEDGE


class PestManager:
    """AI-powered pest identification and IPM recommendation engine."""

    def __init__(self):
        self.model = DecisionTreeClassifier(max_depth=8, min_samples_split=3)
        self.is_trained = False
        self.accuracy = 0.0
        self.pests = list(PEST_KNOWLEDGE.keys())
        self.crops = []

    def train(self, n_samples=250):
        """Train pest detection model."""
        random.seed(42)
        X, y, pests, crops = generate_pest_data(n_samples)
        self.pests = pests
        self.crops = crops

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2)
        self.model.fit(X_train, y_train)
        self.accuracy = self.model.score(X_test, y_test)
        self.is_trained = True

        return {
            "accuracy": round(self.accuracy, 3),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "n_pests": len(pests),
        }

    def identify(self, temperature, humidity, crop, month=6, rainfall=50):
        """Identify likely pest threat and recommend IPM strategy."""
        if not self.is_trained:
            self.train()

        crop_idx = self.crops.index(crop) if crop in self.crops else 0
        features = [temperature, humidity, float(crop_idx), float(month), rainfall]
        pest = self.model.predict([features])[0]
        probas = self.model.predict_proba([features])[0]
        sorted_probs = sorted(probas.items(), key=lambda x: x[1], reverse=True)

        info = PEST_KNOWLEDGE.get(pest, {})
        severity = info.get("severity", "medium")

        return {
            "pest": pest,
            "pest_name": pest.replace("_", " ").title(),
            "confidence": sorted_probs[0][1] if sorted_probs else 0.5,
            "severity": severity,
            "affected_crops": info.get("crops", []),
            "optimal_conditions": info.get("conditions", {}),
            "ipm_strategy": self._ipm_strategy(pest, severity),
            "treatment_options": info.get("treatment", []),
            "economic_threshold": self._economic_threshold(pest),
            "top_threats": [
                {"pest": p.replace("_", " ").title(), "probability": round(pr, 3)}
                for p, pr in sorted_probs[:5]
            ],
        }

    def _ipm_strategy(self, pest, severity):
        """Generate Integrated Pest Management strategy."""
        strategies = {
            "cultural": self._cultural_controls(pest),
            "biological": self._biological_controls(pest),
            "mechanical": self._mechanical_controls(pest),
            "chemical": self._chemical_controls(pest, severity),
        }
        return strategies

    def _cultural_controls(self, pest):
        controls = {
            "aphids": ["Intercrop with marigold or coriander", "Avoid excess nitrogen fertilization"],
            "stem_borer": ["Collect and destroy stubble after harvest", "Clip egg masses during transplanting"],
            "whitefly": ["Use reflective silver mulch", "Avoid planting near infected fields"],
            "bollworm": ["Early and uniform sowing dates", "Destroy crop residues immediately"],
            "brown_planthopper": ["Avoid close spacing", "Alternate wetting and drying irrigation"],
            "leaf_miner": ["Remove and destroy infested leaves", "Use trap crops like castor"],
            "thrips": ["Avoid planting near onion/garlic fields", "Maintain field hygiene"],
            "army_worm": ["Deep plowing to expose pupae", "Trap cropping with napier grass"],
        }
        return controls.get(pest, ["Practice crop rotation", "Maintain field sanitation"])

    def _biological_controls(self, pest):
        controls = {
            "aphids": ["Release ladybird beetles (Coccinellidae)", "Encourage lacewing populations"],
            "stem_borer": ["Release Trichogramma parasitoid wasps", "Conserve spiders in the field"],
            "whitefly": ["Release Encarsia formosa parasitoids", "Apply Beauveria bassiana fungus"],
            "bollworm": ["Release Trichogramma egg parasitoids", "Apply Bt (Bacillus thuringiensis)"],
            "brown_planthopper": ["Conserve spider populations", "Avoid broad-spectrum insecticides"],
            "leaf_miner": ["Release Diglyphus isaea parasitoids", "Apply neem-based products"],
            "thrips": ["Release predatory mites (Amblyseius)", "Apply Metarhizium anisopliae"],
            "army_worm": ["Release Telenomus remus parasitoids", "Apply NPV (Nuclear Polyhedrosis Virus)"],
        }
        return controls.get(pest, ["Use neem-based biopesticides", "Conserve natural enemies"])

    def _mechanical_controls(self, pest):
        controls = {
            "aphids": ["Strong water spray to dislodge", "Yellow sticky traps for monitoring"],
            "stem_borer": ["Light traps (mercury vapor) for moths", "Pheromone traps for monitoring"],
            "whitefly": ["Yellow sticky traps at canopy level", "Vacuum removal in small fields"],
            "bollworm": ["Pheromone traps (5/ha)", "Hand picking of larvae"],
            "brown_planthopper": ["Light traps at field borders", "Sweepnet monitoring"],
            "leaf_miner": ["Remove and destroy mined leaves", "Blue sticky traps"],
            "thrips": ["Blue sticky traps at crop height", "Overhead irrigation to disturb"],
            "army_worm": ["Dig trenches around affected areas", "Bird perches for predation"],
        }
        return controls.get(pest, ["Use appropriate traps", "Manual removal where feasible"])

    def _chemical_controls(self, pest, severity):
        if severity == "low":
            return ["Chemical control generally not needed at this severity level"]
        controls = {
            "aphids": ["Imidacloprid 17.8% SL at 0.3ml/L (last resort)"],
            "stem_borer": ["Carbofuran 3G at 25kg/ha in leaf whorl"],
            "whitefly": ["Spiromesifen 22.9% SC at 0.5ml/L"],
            "bollworm": ["Emamectin benzoate 5% SG at 0.2g/L"],
            "brown_planthopper": ["Pymetrozine 50% WG at 0.3g/L"],
            "leaf_miner": ["Cyromazine 75% WP at 0.3g/L"],
            "thrips": ["Spinosad 45% SC at 0.15ml/L"],
            "army_worm": ["Chlorantraniliprole 18.5% SC at 0.3ml/L"],
        }
        note = " (Use only when ETL is exceeded and biocontrols fail)"
        result = controls.get(pest, ["Consult local extension officer"])
        return [r + note if i == 0 else r for i, r in enumerate(result)]

    def _economic_threshold(self, pest):
        thresholds = {
            "aphids": {"threshold": "10-15 aphids per plant", "action": "Spray when exceeded"},
            "stem_borer": {"threshold": "5% dead hearts or 2% white ears", "action": "Apply granules"},
            "whitefly": {"threshold": "5-10 adults per leaf", "action": "Start control measures"},
            "bollworm": {"threshold": "1 larva per plant on 10% plants", "action": "Begin Bt spray"},
            "brown_planthopper": {"threshold": "5-10 hoppers per hill", "action": "Drain field + spray"},
            "leaf_miner": {"threshold": "3-5 mines per leaf", "action": "Remove leaves + spray"},
            "thrips": {"threshold": "10-15 per flower", "action": "Apply spinosad"},
            "army_worm": {"threshold": "2-3 larvae per sq meter", "action": "Immediate intervention"},
        }
        return thresholds.get(pest, {"threshold": "Consult local ETL", "action": "Monitor regularly"})

    def get_model_info(self):
        return {
            "name": "Pest Manager",
            "algorithm": "Decision Tree + IPM Knowledge Base",
            "n_pests": len(self.pests),
            "trained": self.is_trained,
            "accuracy": round(self.accuracy, 3) if self.is_trained else None,
        }
