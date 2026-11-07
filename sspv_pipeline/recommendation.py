"""
MODULE 7: Hybrid Logic-and-ML Recommendation Engine.

Implements Slide 14 Hybrid Recommendation Engine:
- Tier 1: Rule-Based Logic for Determinate Agronomic Conditions
- Tier 2: Decision Tree Classifier for Indeterminate Conditions
"""

from typing import Dict, Any

import numpy as np
from sklearn.tree import DecisionTreeClassifier


class HybridRecommendationEngine:
    """
    Rule-Based Logic (determinate conditions):
        * Low NGRDI + Low ExG -> Irrigation Refill Alert
        * Persistently Elevated ExB -> Waterlogging / Seepage Excess Warning
        * High NGRDI + High VARI -> Optimal (No Intervention Needed)
        * Elevated Salt Patch Ratio -> SSPV Mound Salinity Leaching Advisory

    Decision Tree Classifier (indeterminate conditions):
        * Inputs: CNN Task-Head Class Probabilities (9 dims) + 4 RGB VIs = 13 features
        * Output: Ranked Irrigation Recommendations with Confidence Scores
    """

    RECOMMENDATION_LABELS = [
        "Optimal: Maintain Current SSPV Capillary Delivery (No Intervention)",
        "Moderate Refill: Replenish SSPV Vessels to 75% Capacity Within 24h",
        "Urgent Irrigation Alert: Immediate SSPV Refill + Check Porous Wall Clogging",
        "Waterlogging Warning: Suspend Refill & Inspect Subsurface Drainage",
        "Salinity Management: Scrape Outer Salt Crust on Mound Perimeter"
    ]

    def __init__(self):
        self.dt_classifier = DecisionTreeClassifier(max_depth=5, random_state=42)
        self._fit_domain_calibrated_tree()

    def _fit_domain_calibrated_tree(self):
        """
        Pre-calibrates the Decision Tree on agronomist domain knowledge patterns
        covering indeterminate multi-task probability & vegetation index combinations.
        """
        rng = np.random.default_rng(42)
        n_samples = 600
        X, y = [], []

        for _ in range(n_samples):
            # Sample synthetic CNN task probabilities and VI values
            p_stress = rng.dirichlet([1.0, 1.0, 1.0])
            p_moist = rng.dirichlet([1.0, 1.0, 1.0])
            p_yield = rng.dirichlet([1.0, 1.0, 1.0])
            ngrdi = rng.uniform(-0.15, 0.30)
            exg = rng.uniform(-0.05, 0.35)
            vari = rng.uniform(-0.20, 0.35)
            tgi = rng.uniform(-0.05, 0.15)

            feat = np.concatenate([p_stress, p_moist, p_yield, [ngrdi, exg, vari, tgi]])
            X.append(feat)

            # Agronomist target assignment
            if p_moist[2] > 0.55:
                y.append(3)  # Waterlogging
            elif p_stress[2] > 0.50 or (p_moist[0] > 0.55 and ngrdi < 0.0):
                y.append(2)  # Urgent Refill
            elif p_stress[1] > 0.45 or ngrdi < 0.06:
                y.append(1)  # Moderate Refill
            else:
                y.append(0)  # Optimal

        self.dt_classifier.fit(np.array(X), np.array(y))

    def recommend(
        self,
        ngrdi: float,
        exg: float,
        vari: float,
        tgi: float,
        exb: float,
        salt_ratio: float,
        stress_probs: np.ndarray,
        moisture_probs: np.ndarray,
        yield_probs: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluates Rule-Based Logic first; falls back to Decision Tree if indeterminate.
        Returns a complete list of Ranked Recommendations with associated confidence scores.
        """
        triggers = []

        # Always compute Decision Tree probability distribution across candidate actions
        feat = np.concatenate([
            stress_probs, moisture_probs, yield_probs,
            [ngrdi, exg, vari, tgi]
        ]).reshape(1, -1)
        dt_probs = self.dt_classifier.predict_proba(feat)[0]

        # Build score distribution across all 5 RECOMMENDATION_LABELS
        score_dist = np.zeros(len(self.RECOMMENDATION_LABELS), dtype=np.float32)
        for cls_idx, prob_val in zip(self.dt_classifier.classes_, dt_probs):
            score_dist[int(cls_idx)] = float(prob_val)
        score_dist[4] = min(0.85, float(salt_ratio * 4.0))

        # Determinate Rule 1: Urgent Irrigation Alert (Slide 14: Low NGRDI + Low ExG)
        if ngrdi < 0.01 and exg < 0.05:
            primary_action = self.RECOMMENDATION_LABELS[2]
            confidence = 0.94
            score_dist[2] = 0.94
            engine_mode = "Rule-Based Logic (Determinate: Low NGRDI + Low ExG)"
            triggers.append(f"Low NGRDI ({ngrdi:.3f}) & Low ExG ({exg:.3f})")

        # Determinate Rule 2: Waterlogging Warning (Slide 14: Elevated ExB)
        elif exb > 0.08 and moisture_probs[2] > 0.45:
            primary_action = self.RECOMMENDATION_LABELS[3]
            confidence = 0.91
            score_dist[3] = 0.91
            engine_mode = "Rule-Based Logic (Determinate: Elevated ExB)"
            triggers.append(f"Elevated ExB ({exb:.3f}) & High Wet Zone Prob ({moisture_probs[2]:.2f})")

        # Determinate Rule 3: Optimal Condition (Slide 14: High NGRDI + High VARI)
        elif ngrdi > 0.09 and vari > 0.07 and stress_probs[0] > 0.50:
            primary_action = self.RECOMMENDATION_LABELS[0]
            confidence = 0.95
            score_dist[0] = 0.95
            engine_mode = "Rule-Based Logic (Determinate: High NGRDI + High VARI)"
            triggers.append(f"High NGRDI ({ngrdi:.3f}) & High VARI ({vari:.3f})")

        # Indeterminate Conditions -> Decision Tree Classifier
        else:
            pred_idx = int(np.argmax(score_dist[:4]))
            primary_action = self.RECOMMENDATION_LABELS[pred_idx]
            confidence = float(score_dist[pred_idx])
            engine_mode = "Decision Tree Classifier (Indeterminate Multi-Factor State)"
            triggers.append("Hybrid fusion of CNN task probabilities + RGB indices")

        # Normalize & sort ranked recommendations (Slide 14)
        norm_scores = score_dist / (np.sum(score_dist) + 1e-8)
        ranked_indices = np.argsort(norm_scores)[::-1]
        ranked_recommendations = [
            {
                "rank": int(r_idx + 1),
                "action": self.RECOMMENDATION_LABELS[int(i)],
                "confidence": round(float(norm_scores[int(i)]), 3)
            }
            for r_idx, i in enumerate(ranked_indices)
        ]

        # Secondary check for Salinity Crust around SSPV mounds (Slide 7)
        secondary_advisory = None
        if salt_ratio > 0.05:
            secondary_advisory = (
                f"Salinity Advisory: White salt patch detected on {salt_ratio*100:.1f}% of plot area. "
                "SSPV outward percolation is actively pushing salts away from the root zone."
            )

        # RuTAG Human Ecology & Rural Farmer Practical Advisory (Bilingual: EN & HI)
        rutag_advisories = {
            self.RECOMMENDATION_LABELS[0]: {
                "en": "Capillary balance is optimal. Earthen vessel is auto-regulating moisture on plant suction. No refill required.",
                "hi": "मटके से नमी का संतुलन उत्तम है। पौधे अपनी आवश्यकता अनुसार जल ले रहे हैं। आज पानी भरने की आवश्यकता नहीं है।",
                "water_saved_l_day": 3.8,
                "urgency": "Normal"
            },
            self.RECOMMENDATION_LABELS[1]: {
                "en": "Water level inside porous vessels is depleting. Refill vessels to 75% capacity within 24 hours to sustain root hydration.",
                "hi": "मटकों में जल स्तर कम हो रहा है। पौधों में सूखा तनाव रोकने के लिए अगले 24 घंटे में मटकों को तीन-चौथाई भरें।",
                "water_saved_l_day": 3.2,
                "urgency": "Medium"
            },
            self.RECOMMENDATION_LABELS[2]: {
                "en": "Urgent refill needed. If vessels are already full, inspect clay walls for pore siltation and brush interior with clean water.",
                "hi": "अति आवश्यक: तुरंत मटके भरें। यदि मटके भरे हैं, तो उनकी दीवारों पर गाद जमने की जांच करें और भीतरी सतह साफ करें।",
                "water_saved_l_day": 2.5,
                "urgency": "High"
            },
            self.RECOMMENDATION_LABELS[3]: {
                "en": "Soil is saturated. Suspend refills immediately and inspect subsurface drainage channels to prevent root asphyxiation.",
                "hi": "खेत में अत्यधिक नमी है। तुरंत पानी भरना रोकें। मेड़ और जलनिकासी की जांच करें ताकि पौधों की जड़ें न सड़ें।",
                "water_saved_l_day": 0.0,
                "urgency": "High"
            },
            self.RECOMMENDATION_LABELS[4]: {
                "en": "Porous vessel has successfully pushed salts outward. Scrape and remove the white saline crust from mound perimeter.",
                "hi": "मटके ने लवण को मेड़ के किनारे धकेल दिया है। सिंचाई से पहले सफेद नमक की पपड़ी को खुरच कर खेत से बाहर निकालें।",
                "water_saved_l_day": 3.5,
                "urgency": "Medium"
            }
        }

        rutag_info = rutag_advisories.get(primary_action, rutag_advisories[self.RECOMMENDATION_LABELS[0]])

        return {
            "primary_recommendation": primary_action,
            "confidence": round(confidence, 3),
            "engine_mode": engine_mode,
            "triggers": triggers,
            "secondary_advisory": secondary_advisory,
            "ranked_recommendations": ranked_recommendations,
            "rutag_farmer_advisory": {
                "action_en": rutag_info["en"],
                "action_hi": rutag_info["hi"],
                "urgency": rutag_info["urgency"],
                "estimated_water_saved_liters_day": rutag_info["water_saved_l_day"]
            }
        }
