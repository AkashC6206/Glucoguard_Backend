"""
GlucoGuard - AI, ML & Clinical RAG Backend Server
FastAPI service delivering food intelligence NLP, causal experiments, counterfactual ranking,
and evidence-based clinical RAG without heavy infrastructure costs.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

from ml.food_nlp import FoodNLPParser
from ml.causal_engine import NOf1CausalComparator, CounterfactualActionRanker
from ml.moderation import PrivacyScrubber, MisinformationClassifier
from rag.clinical_rag import ClinicalRAGAssistant

load_dotenv()

app = FastAPI(
    title="GlucoGuard Clinical AI & ML Backend",
    description="Evidence-based diabetes intelligence, food NLP, causal modeling, and clinical RAG.",
    version="1.0.0"
)

# Enable CORS for Flutter mobile client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines
food_parser = FoodNLPParser()
rag_assistant = ClinicalRAGAssistant()
misinformation_classifier = MisinformationClassifier()

# -----------------------------------------------------------------------------
# Request & Response Models
# -----------------------------------------------------------------------------
class FoodParseRequest(BaseModel):
    meal_text: str = Field(description="Natural language meal description", example="Two rotis with dal and cucumber salad")

class CounterfactualRequest(BaseModel):
    carbs_g: float = Field(example=50.0)
    fiber_g: float = Field(default=5.0, example=5.0)
    baseline_glucose: float = Field(default=110.0, example=115.0)

class NOf1TrialRequest(BaseModel):
    trial_name: str = Field(default="15-min Post-Dinner Walk", example="15-min Post-Dinner Walk")
    baseline_readings: List[float] = Field(example=[180, 175, 185, 190, 170, 182, 178])
    intervention_readings: List[float] = Field(example=[145, 140, 150, 155, 138, 142, 144])

class RAGQuestionRequest(BaseModel):
    question: str = Field(example="What is the ADA recommended target for post-prandial blood sugar?")

class ModerationRequest(BaseModel):
    post_text: str = Field(example="My doctor adjusted my metformin, email me at test@example.com")

# -----------------------------------------------------------------------------
# REST Endpoints
# -----------------------------------------------------------------------------
@app.get("/health")
def health_check():
    return {
        "status": "online",
        "service": "GlucoGuard AI Backend",
        "version": "1.0.0",
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
        "supabase_configured": bool(os.getenv("SUPABASE_URL"))
    }

@app.post("/api/food/parse")
async def parse_food_meal(req: FoodParseRequest):
    """Section 4.1: Natural language food intelligence & portion multiplier extraction."""
    if not req.meal_text.strip():
        raise HTTPException(status_code=400, detail="meal_text cannot be empty.")
    result = await food_parser.parse_meal_text(req.meal_text.strip())
    return result

@app.post("/api/simulation/counterfactuals")
def simulate_counterfactuals(req: CounterfactualRequest):
    """Section 4.3: Real-time glycemic excursion counterfactual ranking."""
    options = CounterfactualActionRanker.simulate_options(
        carbs_g=req.carbs_g,
        fiber_g=req.fiber_g,
        baseline_glucose=req.baseline_glucose
    )
    return {"options": options}

@app.post("/api/experiments/n-of-1")
def evaluate_n_of_1_experiment(req: NOf1TrialRequest):
    """Section 4.4: Statistical significance (Wilcoxon + Cohen's d) for 7-day personal trials."""
    result = NOf1CausalComparator.compare_trial(
        baseline_readings=req.baseline_readings,
        intervention_readings=req.intervention_readings,
        trial_name=req.trial_name
    )
    return result

@app.post("/api/rag/ask")
async def ask_clinical_assistant(req: RAGQuestionRequest):
    """Section 5.1 - 5.3: Evidence-grounded Clinical RAG Assistant with mandatory guidelines."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    result = await rag_assistant.answer_patient_query(req.question.strip())
    return result

@app.post("/api/moderation/screen")
async def screen_community_post(req: ModerationRequest):
    """Section 4.6: PHI scrubbing and misinformation classification."""
    clean_text, had_phi = PrivacyScrubber.scrub_text(req.post_text)
    safety_result = await misinformation_classifier.evaluate_post(clean_text)
    return {
        "clean_text": clean_text,
        "had_phi_scrubbed": had_phi,
        "is_safe": safety_result["is_safe"],
        "flagged_reason": safety_result["flagged_reason"]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
