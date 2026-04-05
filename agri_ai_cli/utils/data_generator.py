"""
Synthetic Data Generator for AgriAI CLI.
Generates realistic agricultural training data for all ML models.
"""

import random
import math


# ── Disease Detection Data ─────────────────────────────────────────

DISEASE_KNOWLEDGE = {
    "rice_blast": {
        "crop": "rice",
        "symptoms": [
            "diamond shaped lesions on leaves",
            "gray center spots with brown borders",
            "leaf tips turning brown and dying",
            "white to gray fungal growth on leaves",
            "neck rot causing panicle to break",
            "nodes turning black on the stem",
        ],
        "severity_range": (3, 9),
    },
    "rice_brown_spot": {
        "crop": "rice",
        "symptoms": [
            "oval brown spots on leaves",
            "small circular brown lesions",
            "seedling blight with brown discoloration",
            "spots with yellow halo around them",
            "grain discoloration during maturity",
        ],
        "severity_range": (2, 7),
    },
    "wheat_rust": {
        "crop": "wheat",
        "symptoms": [
            "orange rust pustules on leaves",
            "brown powdery spores on leaf surface",
            "yellow streaks running along leaf veins",
            "stem showing reddish brown rust spots",
            "premature leaf senescence and drying",
            "reduced grain filling with shriveled seeds",
        ],
        "severity_range": (4, 10),
    },
    "wheat_powdery_mildew": {
        "crop": "wheat",
        "symptoms": [
            "white powdery coating on leaf surface",
            "gray fungal patches on stems",
            "leaves curling and turning yellow",
            "reduced tillering and stunted growth",
            "chaffy grains with poor development",
        ],
        "severity_range": (2, 8),
    },
    "corn_leaf_blight": {
        "crop": "corn",
        "symptoms": [
            "long elliptical gray green lesions",
            "cigar shaped spots on lower leaves",
            "lesions merging to kill entire leaf",
            "tan colored necrotic areas spreading upward",
            "reduced ear size and kernel filling",
        ],
        "severity_range": (3, 9),
    },
    "corn_common_rust": {
        "crop": "corn",
        "symptoms": [
            "small reddish brown pustules on both leaf surfaces",
            "circular to elongate rust spots",
            "dark brown to black spores late season",
            "heavy infection causing premature death",
            "chlorotic halos around pustules",
        ],
        "severity_range": (2, 7),
    },
    "tomato_early_blight": {
        "crop": "tomato",
        "symptoms": [
            "dark concentric rings on lower leaves",
            "target like spots with bulls eye pattern",
            "yellowing around leaf lesions",
            "stem canker near soil line",
            "fruit rot with dark leathery spots",
            "progressive defoliation from bottom up",
        ],
        "severity_range": (3, 8),
    },
    "tomato_late_blight": {
        "crop": "tomato",
        "symptoms": [
            "water soaked dark spots on leaves",
            "white fuzzy mold on leaf undersides",
            "rapid browning and death of foliage",
            "brown firm rot on green fruit",
            "stems showing dark brown streaks",
            "entire plant wilting in humid conditions",
        ],
        "severity_range": (5, 10),
    },
    "potato_late_blight": {
        "crop": "potato",
        "symptoms": [
            "dark water soaked lesions on leaf margins",
            "white sporulation on leaf undersides in morning",
            "rapid foliage death and blackening",
            "tuber rot with reddish brown granular flesh",
            "foul smell from infected plant parts",
        ],
        "severity_range": (5, 10),
    },
    "cotton_bacterial_blight": {
        "crop": "cotton",
        "symptoms": [
            "angular water soaked spots on leaves",
            "dark brown to black lesions on veins",
            "boll rot with bacterial ooze",
            "blackarm symptoms on stems and branches",
            "seedling wilt and damping off",
        ],
        "severity_range": (3, 8),
    },
    "soybean_rust": {
        "crop": "soybean",
        "symptoms": [
            "small tan to dark brown lesions on leaves",
            "pustules on leaf undersides releasing spores",
            "premature defoliation and pod abortion",
            "reduced seed size and quality",
            "yellowing and browning of lower canopy",
        ],
        "severity_range": (4, 9),
    },
}


def generate_disease_data(n_samples=500):
    """Generate synthetic disease symptom descriptions and labels."""
    texts = []
    labels = []
    diseases = list(DISEASE_KNOWLEDGE.keys())

    for _ in range(n_samples):
        disease = random.choice(diseases)
        info = DISEASE_KNOWLEDGE[disease]

        # Combine 2-4 symptoms with some noise
        n_symptoms = random.randint(2, min(4, len(info["symptoms"])))
        selected = random.sample(info["symptoms"], n_symptoms)

        # Add crop context sometimes
        parts = []
        if random.random() > 0.3:
            parts.append(f"my {info['crop']} plant is showing")
        parts.extend(selected)

        # Add noise words
        noise = ["I noticed", "there are", "the plant has", "I can see", "found",
                 "observed", "the field shows", "crops are exhibiting"]
        if random.random() > 0.5:
            parts.insert(0, random.choice(noise))

        text = " ".join(parts)
        texts.append(text)
        labels.append(disease)

    return texts, labels


# ── Yield Prediction Data ─────────────────────────────────────────

CROP_YIELD_PARAMS = {
    "rice": {"base_yield": 4500, "temp_opt": 28, "rain_opt": 1200, "ph_opt": 6.0},
    "wheat": {"base_yield": 3800, "temp_opt": 22, "rain_opt": 600, "ph_opt": 6.5},
    "corn": {"base_yield": 8000, "temp_opt": 25, "rain_opt": 800, "ph_opt": 6.2},
    "tomato": {"base_yield": 35000, "temp_opt": 24, "rain_opt": 700, "ph_opt": 6.3},
    "soybean": {"base_yield": 2800, "temp_opt": 26, "rain_opt": 700, "ph_opt": 6.5},
    "cotton": {"base_yield": 2000, "temp_opt": 30, "rain_opt": 900, "ph_opt": 6.8},
    "potato": {"base_yield": 25000, "temp_opt": 18, "rain_opt": 500, "ph_opt": 5.5},
}


def generate_yield_data(n_samples=400):
    """Generate synthetic yield prediction data.
    Features: [temperature, rainfall_mm, humidity%, soil_ph, nitrogen_kg,
               phosphorus_kg, potassium_kg, sunlight_hours, crop_encoded]
    """
    crops = list(CROP_YIELD_PARAMS.keys())
    X = []
    y = []

    for _ in range(n_samples):
        crop = random.choice(crops)
        params = CROP_YIELD_PARAMS[crop]
        crop_idx = crops.index(crop)

        temp = random.uniform(15, 40)
        rainfall = random.uniform(200, 2000)
        humidity = random.uniform(30, 95)
        soil_ph = random.uniform(4.5, 8.5)
        nitrogen = random.uniform(20, 200)
        phosphorus = random.uniform(10, 100)
        potassium = random.uniform(15, 150)
        sunlight = random.uniform(4, 12)

        # Calculate yield based on proximity to optimal conditions
        temp_factor = max(0, 1 - abs(temp - params["temp_opt"]) / 15)
        rain_factor = max(0, 1 - abs(rainfall - params["rain_opt"]) / 1000)
        ph_factor = max(0, 1 - abs(soil_ph - params["ph_opt"]) / 3)
        nutrient_factor = min(1, (nitrogen + phosphorus + potassium) / 300)
        sun_factor = min(1, sunlight / 8)

        yield_val = params["base_yield"] * (
            0.3 * temp_factor +
            0.25 * rain_factor +
            0.15 * ph_factor +
            0.15 * nutrient_factor +
            0.15 * sun_factor
        )
        yield_val *= random.uniform(0.85, 1.15)  # noise
        yield_val = max(100, yield_val)

        features = [temp, rainfall, humidity, soil_ph, nitrogen,
                     phosphorus, potassium, sunlight, float(crop_idx)]
        X.append(features)
        y.append(yield_val)

    return X, y, crops


# ── Soil Analysis Data ─────────────────────────────────────────────

SOIL_TYPES = {
    "alluvial": {"n": (180, 280), "p": (20, 50), "k": (150, 250), "ph": (6.5, 7.5),
                 "organic": (0.5, 1.5), "best_crops": ["rice", "wheat", "sugarcane"]},
    "black": {"n": (100, 200), "p": (15, 40), "k": (200, 350), "ph": (7.0, 8.5),
              "organic": (0.3, 1.0), "best_crops": ["cotton", "soybean", "wheat"]},
    "red": {"n": (80, 150), "p": (10, 30), "k": (100, 200), "ph": (5.5, 6.8),
            "organic": (0.2, 0.8), "best_crops": ["groundnut", "potato", "corn"]},
    "laterite": {"n": (60, 120), "p": (5, 20), "k": (80, 180), "ph": (5.0, 6.5),
                 "organic": (0.1, 0.5), "best_crops": ["cashew", "tea", "coffee"]},
    "sandy": {"n": (40, 100), "p": (5, 25), "k": (50, 150), "ph": (6.0, 7.5),
              "organic": (0.1, 0.4), "best_crops": ["melon", "carrot", "potato"]},
    "clay": {"n": (120, 220), "p": (15, 45), "k": (180, 300), "ph": (6.0, 8.0),
             "organic": (0.4, 1.2), "best_crops": ["rice", "wheat", "cotton"]},
}


def generate_soil_data(n_samples=300):
    """Generate synthetic soil analysis data.
    Features: [nitrogen, phosphorus, potassium, ph, organic_carbon, moisture%, temperature]
    Labels: soil_type
    """
    soil_types = list(SOIL_TYPES.keys())
    X = []
    y = []

    for _ in range(n_samples):
        soil = random.choice(soil_types)
        params = SOIL_TYPES[soil]

        nitrogen = random.uniform(*params["n"])
        phosphorus = random.uniform(*params["p"])
        potassium = random.uniform(*params["k"])
        ph = random.uniform(*params["ph"])
        organic = random.uniform(*params["organic"])
        moisture = random.uniform(15, 80)
        temperature = random.uniform(15, 40)

        features = [nitrogen, phosphorus, potassium, ph, organic, moisture, temperature]
        X.append(features)
        y.append(soil)

    return X, y, soil_types


# ── Pest Data ──────────────────────────────────────────────────────

PEST_KNOWLEDGE = {
    "aphids": {
        "crops": ["wheat", "cotton", "tomato", "potato"],
        "conditions": {"temp": (15, 30), "humidity": (50, 90)},
        "severity": "medium",
        "treatment": ["neem oil spray", "ladybug release", "insecticidal soap"],
    },
    "stem_borer": {
        "crops": ["rice", "corn", "sugarcane"],
        "conditions": {"temp": (25, 35), "humidity": (60, 95)},
        "severity": "high",
        "treatment": ["trichogramma release", "light traps", "pheromone traps"],
    },
    "whitefly": {
        "crops": ["tomato", "cotton", "soybean"],
        "conditions": {"temp": (20, 35), "humidity": (40, 80)},
        "severity": "medium",
        "treatment": ["yellow sticky traps", "neem extract", "reflective mulch"],
    },
    "bollworm": {
        "crops": ["cotton", "corn", "tomato"],
        "conditions": {"temp": (22, 35), "humidity": (50, 85)},
        "severity": "high",
        "treatment": ["bt spray", "pheromone traps", "hand picking"],
    },
    "brown_planthopper": {
        "crops": ["rice"],
        "conditions": {"temp": (25, 32), "humidity": (70, 95)},
        "severity": "critical",
        "treatment": ["alternate wetting drying", "resistant varieties", "spider conservation"],
    },
    "leaf_miner": {
        "crops": ["tomato", "potato", "cotton"],
        "conditions": {"temp": (18, 30), "humidity": (40, 75)},
        "severity": "low",
        "treatment": ["parasitic wasp release", "neem spray", "remove affected leaves"],
    },
    "thrips": {
        "crops": ["cotton", "soybean", "wheat"],
        "conditions": {"temp": (20, 33), "humidity": (30, 70)},
        "severity": "medium",
        "treatment": ["blue sticky traps", "spinosad spray", "predatory mites"],
    },
    "army_worm": {
        "crops": ["corn", "rice", "wheat", "soybean"],
        "conditions": {"temp": (22, 35), "humidity": (60, 90)},
        "severity": "critical",
        "treatment": ["early detection scouting", "bt spray", "chemical control as last resort"],
    },
}


def generate_pest_data(n_samples=300):
    """Generate synthetic pest detection data.
    Features: [temperature, humidity, crop_encoded, month, rainfall_recent]
    Labels: pest_type
    """
    pests = list(PEST_KNOWLEDGE.keys())
    crops = ["rice", "wheat", "corn", "tomato", "cotton", "soybean", "potato", "sugarcane"]
    X = []
    y = []

    for _ in range(n_samples):
        pest = random.choice(pests)
        info = PEST_KNOWLEDGE[pest]
        crop = random.choice(info["crops"])
        crop_idx = crops.index(crop) if crop in crops else 0

        temp = random.uniform(*info["conditions"]["temp"])
        humidity = random.uniform(*info["conditions"]["humidity"])
        month = random.randint(1, 12)
        rainfall = random.uniform(0, 200)

        # Add some noise
        temp += random.gauss(0, 2)
        humidity += random.gauss(0, 5)

        features = [temp, humidity, float(crop_idx), float(month), rainfall]
        X.append(features)
        y.append(pest)

    return X, y, pests, crops


# ── Weather Risk Data ──────────────────────────────────────────────

def generate_weather_scenarios(n=50):
    """Generate weather scenario data for risk assessment."""
    scenarios = []
    for _ in range(n):
        temp = random.uniform(10, 45)
        humidity = random.uniform(20, 100)
        rainfall = random.uniform(0, 300)
        wind_speed = random.uniform(0, 80)
        cloud_cover = random.uniform(0, 100)

        # Calculate risk scores
        heat_risk = max(0, (temp - 35) / 10) if temp > 35 else max(0, (10 - temp) / 10)
        flood_risk = max(0, (rainfall - 150) / 150)
        drought_risk = max(0, 1 - rainfall / 50) if rainfall < 50 else 0
        wind_risk = max(0, (wind_speed - 40) / 40)
        frost_risk = max(0, (5 - temp) / 10) if temp < 5 else 0

        overall = 0.25 * heat_risk + 0.25 * flood_risk + 0.2 * drought_risk + 0.2 * wind_risk + 0.1 * frost_risk

        if overall > 0.7:
            level = "CRITICAL"
        elif overall > 0.4:
            level = "HIGH"
        elif overall > 0.2:
            level = "MEDIUM"
        else:
            level = "LOW"

        scenarios.append({
            "temperature": round(temp, 1),
            "humidity": round(humidity, 1),
            "rainfall": round(rainfall, 1),
            "wind_speed": round(wind_speed, 1),
            "cloud_cover": round(cloud_cover, 1),
            "risk_scores": {
                "heat": round(heat_risk, 3),
                "flood": round(flood_risk, 3),
                "drought": round(drought_risk, 3),
                "wind": round(wind_risk, 3),
                "frost": round(frost_risk, 3),
            },
            "overall_risk": round(overall, 3),
            "risk_level": level,
        })

    return scenarios


# ── Market Price Data ──────────────────────────────────────────────

def generate_market_data(crop="rice", months=24):
    """Generate synthetic historical price data with seasonal patterns."""
    base_prices = {
        "rice": 22.0, "wheat": 25.0, "corn": 18.0, "tomato": 30.0,
        "cotton": 55.0, "soybean": 38.0, "potato": 15.0, "sugarcane": 3.5,
    }
    base = base_prices.get(crop, 20.0)
    prices = []
    for m in range(months):
        seasonal = math.sin(2 * math.pi * m / 12) * base * 0.15
        trend = m * base * 0.005
        noise = random.gauss(0, base * 0.05)
        price = base + seasonal + trend + noise
        prices.append(max(base * 0.5, price))
    return prices


# ── Carbon Footprint Data ─────────────────────────────────────────

CARBON_FACTORS = {
    "diesel_per_liter": 2.68,       # kg CO2
    "electricity_per_kwh": 0.82,    # kg CO2 (India grid average)
    "urea_per_kg": 1.53,            # kg CO2 (production + application)
    "dap_per_kg": 1.10,             # kg CO2
    "mop_per_kg": 0.58,             # kg CO2
    "pesticide_per_kg": 6.30,       # kg CO2
    "rice_methane_per_ha": 1500,    # kg CO2 equivalent
    "transport_per_km_ton": 0.12,   # kg CO2
    "irrigation_pump_per_hour": 3.5,# kg CO2
}


def generate_carbon_data(n_samples=200):
    """Generate synthetic carbon footprint datasets for farms."""
    data = []
    for _ in range(n_samples):
        area_ha = random.uniform(0.5, 50)
        diesel = random.uniform(20, 200) * area_ha
        electricity = random.uniform(100, 2000) * area_ha
        urea = random.uniform(50, 300) * area_ha
        dap = random.uniform(20, 150) * area_ha
        mop = random.uniform(10, 100) * area_ha
        pesticide = random.uniform(1, 20) * area_ha
        is_rice = random.random() > 0.6
        transport_km = random.uniform(5, 200)
        yield_tons = random.uniform(1, 10) * area_ha
        irrigation_hours = random.uniform(50, 500) * area_ha

        total_carbon = (
            diesel * CARBON_FACTORS["diesel_per_liter"] +
            electricity * CARBON_FACTORS["electricity_per_kwh"] +
            urea * CARBON_FACTORS["urea_per_kg"] +
            dap * CARBON_FACTORS["dap_per_kg"] +
            mop * CARBON_FACTORS["mop_per_kg"] +
            pesticide * CARBON_FACTORS["pesticide_per_kg"] +
            (CARBON_FACTORS["rice_methane_per_ha"] * area_ha if is_rice else 0) +
            transport_km * yield_tons * CARBON_FACTORS["transport_per_km_ton"] +
            irrigation_hours * CARBON_FACTORS["irrigation_pump_per_hour"]
        )

        data.append({
            "area_ha": round(area_ha, 2),
            "diesel_liters": round(diesel, 1),
            "electricity_kwh": round(electricity, 1),
            "urea_kg": round(urea, 1),
            "dap_kg": round(dap, 1),
            "mop_kg": round(mop, 1),
            "pesticide_kg": round(pesticide, 2),
            "is_paddy": is_rice,
            "transport_km": round(transport_km, 1),
            "yield_tons": round(yield_tons, 2),
            "irrigation_hours": round(irrigation_hours, 1),
            "total_carbon_kg": round(total_carbon, 2),
            "carbon_per_ha": round(total_carbon / area_ha, 2),
            "carbon_per_ton": round(total_carbon / max(yield_tons, 0.1), 2),
        })

    return data
