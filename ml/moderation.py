"""
GlucoGuard - Privacy Scrubber & Medical Misinformation Moderation
Strips Personal Health Information (PHI) and flags dangerous medical misinformation
prior to community forum publication.
"""

import re
import os
import json
import google.generativeai as genai
from typing import Dict, Any, Tuple, Optional

class PrivacyScrubber:
    """Zero-dependency regex + rule-based PII/PHI scrubber."""

    EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
    PHONE_PATTERN = re.compile(r'(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}')
    MRN_PATTERN = re.compile(r'\b(MRN|ID|Patient|Hospital)\s*[:#-]?\s*\d{4,10}\b', re.IGNORECASE)

    @classmethod
    def scrub_text(cls, text: str) -> Tuple[str, bool]:
        """Scrubs emails, phones, and MRNs from text. Returns (clean_text, had_phi)."""
        had_phi = False
        clean = text

        if cls.EMAIL_PATTERN.search(clean):
            clean = cls.EMAIL_PATTERN.sub("[REDACTED_EMAIL]", clean)
            had_phi = True

        if cls.PHONE_PATTERN.search(clean):
            clean = cls.PHONE_PATTERN.sub("[REDACTED_PHONE]", clean)
            had_phi = True

        if cls.MRN_PATTERN.search(clean):
            clean = cls.MRN_PATTERN.sub("[REDACTED_ID]", clean)
            had_phi = True

        return clean, had_phi

class MisinformationClassifier:
    """Screens community content for dangerous non-clinical or predatory diabetes advice."""

    DANGEROUS_TRIGGERS = [
        "stop insulin",
        "cure type 1",
        "cure type 2 permanently",
        "herbal cure",
        "cinnamon cures diabetes",
        "miracle water",
        "stop metformin",
        "replace insulin with",
        "doctors don't want you to know",
        "cure in 7 days"
    ]

    def __init__(self, gemini_api_key: Optional[str] = None):
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)

    async def evaluate_post(self, text: str) -> Dict[str, Any]:
        """Evaluates whether post violates clinical safety rules."""
        lower = text.lower()
        for trigger in self.DANGEROUS_TRIGGERS:
            if trigger in lower:
                return {
                    "is_safe": False,
                    "flagged_reason": f"Violates safety guidelines: Claims regarding '{trigger}' are dangerous.",
                    "confidence": 0.98
                }

        # If Gemini API key is configured, perform semantic safety check
        if self.gemini_api_key:
            candidate_models = [
                os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
                "gemini-3.5-flash",
                "gemini-flash-lite-latest",
                "gemini-3.6-flash",
            ]
            models_to_try = []
            for m in candidate_models:
                if m and m not in models_to_try:
                    models_to_try.append(m)

            prompt = f"""
Evaluate if this diabetes community post is clinically safe.
Flag unsafe if it advises stopping prescribed insulin/medication, advertises unproven miracle cures, or gives dangerous medical advice.

Post: "{text}"

Output strictly JSON:
{{
  "is_safe": true,
  "flagged_reason": ""
}}
"""
            for model_name in models_to_try:
                try:
                    model = genai.GenerativeModel(
                        model_name=model_name,
                        generation_config={"response_mime_type": "application/json", "temperature": 0.0}
                    )
                    resp = await model.generate_content_async(prompt)
                    if resp and resp.text:
                        data = json.loads(resp.text)
                        return {
                            "is_safe": bool(data.get("is_safe", True)),
                            "flagged_reason": str(data.get("flagged_reason", "")),
                            "confidence": 0.95
                        }
                except Exception:
                    continue

        return {"is_safe": True, "flagged_reason": "", "confidence": 0.90}
