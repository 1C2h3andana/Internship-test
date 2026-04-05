"""
Core - AI Advisory Engine.
NLP-based conversational assistant that routes queries to appropriate ML models.
"""

import re


class AIAdvisor:
    """Natural language query routing and advisory engine."""

    INTENT_PATTERNS = {
        "disease": [
            r"disease|sick|infect|lesion|spot|blight|rust|mildew|wilt|rot|fungus|yellow.*leaf",
            r"brown.*spot|white.*powder|water.*soak|dying.*leaf|curling",
        ],
        "yield": [
            r"yield|harvest|production|how\s+much|predict.*crop|crop.*output|tons?\s+per",
        ],
        "soil": [
            r"soil|nutrient|nitrogen|phosphorus|potassium|ph\s+level|organic.*carbon",
            r"soil.*type|soil.*health|fertile|deficien",
        ],
        "weather": [
            r"weather|rain|temperature|wind|frost|drought|flood|climate|forecast",
        ],
        "pest": [
            r"pest|insect|bug|worm|aphid|borer|whitefly|thrip|mite|caterpillar",
        ],
        "irrigation": [
            r"irrigat|water|drip|sprinkler|moisture|evapotranspiration|dry",
        ],
        "rotation": [
            r"rotation|next\s+crop|after.*harvest|sequence|what.*plant.*next",
        ],
        "market": [
            r"market|price|sell|msp|trade|cost|revenue|commodity|rate",
        ],
        "fertilizer": [
            r"fertiliz|urea|dap|mop|npk|nutrient.*need|how\s+much.*apply",
        ],
        "carbon": [
            r"carbon|emission|greenhouse|co2|footprint|climate.*impact|sustain",
        ],
        "insurance": [
            r"insur|pmfby|premium|claim|crop.*loss|damage|coverage|policy",
        ],
        "ndvi": [
            r"ndvi|satellite|remote\s+sens|vegetation.*index|field.*map|crop.*health.*map",
        ],
        "finance": [
            r"financ|profit|loss|expense|income|ledger|roi|budget|cost.*analysis",
        ],
    }

    GREETINGS = [
        r"^(hi|hello|hey|good\s+(morning|afternoon|evening)|namaste)",
    ]

    def __init__(self):
        self.context = {"last_crop": "rice", "last_intent": None}

    def process_query(self, query):
        """Process natural language query and return advice."""
        query_lower = query.lower().strip()

        # Check for greetings
        for pattern in self.GREETINGS:
            if re.search(pattern, query_lower):
                return self._greeting_response()

        # Detect intent
        intent = self._detect_intent(query_lower)
        self.context["last_intent"] = intent

        # Extract crop if mentioned
        crop = self._extract_crop(query_lower)
        if crop:
            self.context["last_crop"] = crop

        # Generate response based on intent
        return self._generate_response(intent, query_lower, crop)

    def _detect_intent(self, query):
        """Detect user intent from query text."""
        scores = {}
        for intent, patterns in self.INTENT_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, query)
                score += len(matches)
            if score > 0:
                scores[intent] = score

        if scores:
            return max(scores, key=scores.get)
        return "general"

    def _extract_crop(self, query):
        """Extract crop name from query."""
        crops = [
            "rice", "wheat", "corn", "tomato", "cotton", "soybean",
            "potato", "sugarcane", "chickpea", "lentil", "mustard",
            "groundnut", "onion", "sunflower",
        ]
        for crop in crops:
            if crop in query:
                return crop
        return None

    def _greeting_response(self):
        return {
            "intent": "greeting",
            "response": "Welcome to AgriAI! I can help you with:\n"
                       "  - Crop disease detection\n"
                       "  - Yield prediction\n"
                       "  - Soil analysis\n"
                       "  - Weather risk assessment\n"
                       "  - Pest management (IPM)\n"
                       "  - Irrigation scheduling\n"
                       "  - Crop rotation planning\n"
                       "  - Market price analysis\n"
                       "  - Fertilizer calculation\n"
                       "  - Carbon footprint analysis\n"
                       "  - Crop insurance advisory\n"
                       "  - Satellite NDVI analysis\n"
                       "  - Financial management\n\n"
                       "Just describe your farming question in plain language!",
            "suggested_commands": ["detect", "predict-yield", "analyze-soil", "weather"],
        }

    def _generate_response(self, intent, query, crop):
        """Generate contextual response based on detected intent."""
        crop = crop or self.context.get("last_crop", "rice")

        responses = {
            "disease": {
                "intent": "disease",
                "response": f"I can analyze disease symptoms for {crop}. "
                           "Describe what you see on the plant (leaf spots, discoloration, wilting, etc.) "
                           "and I'll identify the disease and recommend treatment.",
                "suggested_command": f"python main.py detect --symptoms '{query}'",
                "quick_tips": [
                    "Take close-up photos of affected plant parts",
                    "Note if symptoms appear on old or new leaves",
                    "Check nearby plants for similar symptoms",
                ],
            },
            "yield": {
                "intent": "yield",
                "response": f"I can predict yield for {crop} based on your field conditions. "
                           "I need: temperature, rainfall, humidity, soil pH, and nutrient levels.",
                "suggested_command": "python main.py predict-yield",
                "quick_tips": [
                    "Accurate soil test data improves prediction quality",
                    "Consider recent weather patterns for better estimates",
                ],
            },
            "soil": {
                "intent": "soil",
                "response": "I can analyze your soil health from nutrient data. "
                           "Provide N, P, K levels, pH, and organic carbon percentage.",
                "suggested_command": "python main.py analyze-soil",
                "quick_tips": [
                    "Get soil tested at nearest KVK or soil testing lab",
                    "Sample from 6-8 spots at 15cm depth for representative results",
                ],
            },
            "weather": {
                "intent": "weather",
                "response": f"I can assess weather risks for {crop} cultivation. "
                           "Provide current temperature, humidity, rainfall, and wind speed.",
                "suggested_command": "python main.py weather",
                "quick_tips": [
                    "Check IMD forecasts for upcoming week",
                    "Set up a rain gauge in your field for accurate measurement",
                ],
            },
            "pest": {
                "intent": "pest",
                "response": f"I can identify pest threats for {crop} and recommend IPM strategies. "
                           "What temperature and humidity conditions are you experiencing?",
                "suggested_command": "python main.py pest",
                "quick_tips": [
                    "Scout fields early morning when pests are visible",
                    "Check undersides of leaves for eggs and larvae",
                    "Use yellow/blue sticky traps for monitoring",
                ],
            },
            "irrigation": {
                "intent": "irrigation",
                "response": f"I can calculate an optimal irrigation schedule for {crop} "
                           "using the FAO-56 Penman-Monteith method.",
                "suggested_command": "python main.py irrigate",
                "quick_tips": [
                    "Measure soil moisture before irrigating",
                    "Irrigate in early morning to reduce evaporation",
                ],
            },
            "rotation": {
                "intent": "rotation",
                "response": f"I can plan a crop rotation starting from {crop}. "
                           "This considers soil health, nitrogen balance, and economics.",
                "suggested_command": "python main.py rotate",
            },
            "market": {
                "intent": "market",
                "response": f"I can analyze market prices for {crop} with trend forecasting.",
                "suggested_command": "python main.py market",
            },
            "fertilizer": {
                "intent": "fertilizer",
                "response": f"I can calculate precise fertilizer requirements for {crop} "
                           "using the ICAR target yield method.",
                "suggested_command": "python main.py fertilizer",
            },
            "carbon": {
                "intent": "carbon",
                "response": "I can calculate your farm's carbon footprint and suggest reduction strategies.",
                "suggested_command": "python main.py carbon",
            },
            "insurance": {
                "intent": "insurance",
                "response": f"I can calculate PMFBY insurance premiums for {crop} "
                           "and guide you through the claim process.",
                "suggested_command": "python main.py insurance",
            },
            "ndvi": {
                "intent": "ndvi",
                "response": f"I can simulate satellite NDVI analysis for your {crop} field "
                           "to assess crop health spatially.",
                "suggested_command": "python main.py ndvi",
            },
            "finance": {
                "intent": "finance",
                "response": "I can help track farm finances - income, expenses, and profitability analysis.",
                "suggested_command": "python main.py finance",
            },
            "general": {
                "intent": "general",
                "response": "I'm not sure what you're asking about. Try describing your farming "
                           "concern - for example:\n"
                           "  'My rice leaves have brown spots'\n"
                           "  'How much wheat will I harvest?'\n"
                           "  'What fertilizer for cotton?'\n"
                           "  'Is it safe to plant this week?'",
                "suggested_command": "python main.py demo",
            },
        }

        return responses.get(intent, responses["general"])

    def get_model_info(self):
        return {
            "name": "AI Advisor",
            "algorithm": "Regex-based NLP Intent Detection",
            "intents": len(self.INTENT_PATTERNS),
            "supported_crops": 14,
        }
