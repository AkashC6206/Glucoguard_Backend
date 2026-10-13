"""
GlucoGuard - N-of-1 Causal Engine & Counterfactual Action Ranker
Implements statistical significance testing (Wilcoxon signed-rank & Cohen's d)
and physiological counterfactual action simulation using scipy.stats and numpy.
"""

import numpy as np
from scipy import stats
from typing import Dict, Any, List, Optional

class NOf1CausalComparator:
    """Evaluates individualized N-of-1 7-day trials comparing baseline vs intervention."""

    @staticmethod
    def compare_trial(
        baseline_readings: List[float],
        intervention_readings: List[float],
        trial_name: str = "15-min Post-Dinner Walk"
    ) -> Dict[str, Any]:
        if len(baseline_readings) < 3 or len(intervention_readings) < 3:
            return {
                "error": "At least 3 pairs of baseline and intervention readings are required for statistical power."
            }

        # Truncate to matching sample sizes if uneven
        min_len = min(len(baseline_readings), len(intervention_readings))
        base = np.array(baseline_readings[:min_len])
        interv = np.array(intervention_readings[:min_len])

        diffs = base - interv
        mean_baseline = float(np.mean(base))
        mean_intervention = float(np.mean(interv))
        mean_reduction = float(np.mean(diffs))

        # Wilcoxon signed-rank test (non-parametric paired test)
        try:
            wilcoxon_stat, p_value = stats.wilcoxon(base, interv)
            p_val = float(p_value)
        except Exception:
            # Fallback to paired t-test if differences are uniform
            t_stat, p_val = stats.ttest_rel(base, interv)
            p_val = float(p_val)

        # Cohen's d Effect Size
        pooled_std = float(np.sqrt((np.var(base, ddof=1) + np.var(interv, ddof=1)) / 2))
        cohen_d = float(mean_reduction / pooled_std) if pooled_std > 0 else 0.0

        is_significant = p_val < 0.05
        effect_label = "Negligible"
        if abs(cohen_d) >= 0.8:
            effect_label = "Large"
        elif abs(cohen_d) >= 0.5:
            effect_label = "Moderate"
        elif abs(cohen_d) >= 0.2:
            effect_label = "Small"

        return {
            "trial_name": trial_name,
            "sample_pairs": min_len,
            "baseline_mean_mg_dl": round(mean_baseline, 1),
            "intervention_mean_mg_dl": round(mean_intervention, 1),
            "mean_glucose_reduction_mg_dl": round(mean_reduction, 1),
            "p_value": round(p_val, 4),
            "cohen_d_effect_size": round(cohen_d, 2),
            "effect_magnitude": effect_label,
            "is_statistically_significant": is_significant,
            "clinical_recommendation": (
                f"Adopting '{trial_name}' produces a {effect_label.lower()} statistically significant "
                f"reduction of {round(mean_reduction, 1)} mg/dL in post-meal glucose (p={round(p_val, 3)})."
                if is_significant else
                f"'{trial_name}' reduced glucose by {round(mean_reduction, 1)} mg/dL, but requires more data to confirm statistical significance."
            )
        }

class CounterfactualActionRanker:
    """Simulates post-prandial glycemic excursions and ranks alternative user actions."""

    @staticmethod
    def simulate_options(
        carbs_g: float,
        fiber_g: float = 0.0,
        baseline_glucose: float = 110.0
    ) -> List[Dict[str, Any]]:
        """Simulates glycemic excursion across multiple behavioral options."""
        # Unattenuated base excursion: ~3.0 mg/dL per gram of net carb
        net_carbs = max(0.0, carbs_g - (fiber_g * 0.5))
        base_spike = net_carbs * 2.8

        options = [
            {
                "id": "opt_sedentary",
                "label": "Sedentary Rest",
                "duration_minutes": 0,
                "walk_type": "None",
                "glucose_spike_delta": round(base_spike),
                "predicted_peak_glucose": round(baseline_glucose + base_spike),
                "is_recommended": False,
                "description": "Remaining seated post-meal allows full glycemic excursion."
            },
            {
                "id": "opt_walk_10",
                "label": "10-Minute Leisurely Walk",
                "duration_minutes": 10,
                "walk_type": "Leisurely",
                "glucose_spike_delta": round(max(10.0, base_spike - 16.0)),
                "predicted_peak_glucose": round(baseline_glucose + max(10.0, base_spike - 16.0)),
                "is_recommended": base_spike <= 45.0,
                "description": "Blunts immediate peak through GLUT4 non-insulin glucose disposal."
            },
            {
                "id": "opt_walk_15",
                "label": "15-Minute Brisk Walk",
                "duration_minutes": 15,
                "walk_type": "Brisk",
                "glucose_spike_delta": round(max(5.0, base_spike - 28.0)),
                "predicted_peak_glucose": round(baseline_glucose + max(5.0, base_spike - 28.0)),
                "is_recommended": base_spike > 45.0,
                "description": "Significantly flattens postprandial curve (recommended by ADA)."
            },
            {
                "id": "opt_walk_20",
                "label": "20-Minute Post-Meal Walk",
                "duration_minutes": 20,
                "walk_type": "Brisk",
                "glucose_spike_delta": round(max(0.0, base_spike - 38.0)),
                "predicted_peak_glucose": round(baseline_glucose + max(0.0, base_spike - 38.0)),
                "is_recommended": base_spike > 70.0,
                "description": "Maximizes muscular glycogen uptake and blunts spikes by ~35-40 mg/dL."
            }
        ]

        # Sort options by predicted peak glucose (lowest first)
        options.sort(key=lambda x: x["predicted_peak_glucose"])
        return options
