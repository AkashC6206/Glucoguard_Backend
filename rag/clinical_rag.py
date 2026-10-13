"""
GlucoGuard - Clinical RAG Assistant
Integrates Supabase pgvector retrieval with Google Gemini 1.5/2.0 Flash
grounded generation, mandatory citations (ADA, ICMR, WHO, CDC), and clinical safety guardrails.
"""

import os
from typing import Dict, Any, List, Optional
import google.generativeai as genai
from supabase import create_client, Client

class ClinicalRAGAssistant:
    def __init__(self, gemini_api_key: Optional[str] = None, supabase_url: Optional[str] = None, supabase_key: Optional[str] = None):
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_ANON_KEY")

        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)

        self.supabase: Optional[Client] = None
        if self.supabase_url and self.supabase_key:
            try:
                self.supabase = create_client(self.supabase_url, self.supabase_key)
            except Exception as e:
                print(f"Supabase init note: {e}")

    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Generates 768-dim embedding using free Gemini embedding model."""
        if not self.gemini_api_key:
            return None
        try:
            res = genai.embed_content(
                model="models/gemini-embedding-001",
                content=text,
                task_type="retrieval_query",
                output_dimensionality=768
            )
            return res["embedding"]
        except Exception:
            try:
                res = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="retrieval_query"
                )
                return res["embedding"]
            except Exception:
                return None

    async def retrieve_guideline_passages(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Queries Supabase pgvector using match_clinical_guidelines RPC function."""
        if not self.supabase:
            return self._static_guidelines_fallback(query)

        query_embedding = await self.get_embedding(query)
        if not query_embedding:
            return self._static_guidelines_fallback(query)

        try:
            res = self.supabase.rpc(
                "match_clinical_guidelines",
                {
                    "query_embedding": query_embedding,
                    "match_threshold": 0.50,
                    "match_count": top_k
                }
            ).execute()

            if res.data and len(res.data) > 0:
                return res.data
        except Exception:
            pass

        return self._static_guidelines_fallback(query)

    async def answer_patient_query(
        self,
        query: str,
        user_context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Synthesizes clinical answer grounded in retrieved guidelines, personalized with user biometrics & conversation memory."""
        passages = await self.retrieve_guideline_passages(query)
        context_str = "\n\n".join([
            f"Source: {p.get('source', 'Guideline')}\nChapter: {p.get('chapter', 'General')}\nExcerpt: {p.get('content', '')}"
            for p in passages
        ])

        # Format user biometrics snapshot if provided
        user_context_block = ""
        if user_context:
            uc_lines = ["Patient Real-Time Health & Biometrics Snapshot:"]
            if "latest_glucose" in user_context and user_context["latest_glucose"] is not None:
                ctx = user_context.get("latest_glucose_context", "recent")
                uc_lines.append(f"- Latest Blood Glucose: {user_context['latest_glucose']} mg/dL ({ctx})")
            if "avg_glucose_7d" in user_context and user_context["avg_glucose_7d"] is not None:
                uc_lines.append(f"- 7-Day Average Glucose: {user_context['avg_glucose_7d']} mg/dL")
            if "today_steps" in user_context:
                uc_lines.append(f"- Today's Step Count: {user_context['today_steps']} steps")
            if "today_exercise_minutes" in user_context:
                uc_lines.append(f"- Today's Workout Minutes: {user_context['today_exercise_minutes']} min")
            if "today_workouts" in user_context and user_context["today_workouts"]:
                uc_lines.append(f"- Today's Logged Activities: {', '.join(user_context['today_workouts'])}")
            if "today_carbs_g" in user_context:
                uc_lines.append(f"- Today's Total Carbs: ~{user_context['today_carbs_g']}g carbs")
            if "today_calories" in user_context:
                uc_lines.append(f"- Today's Total Calories: ~{user_context['today_calories']} kcal")
            if "today_meals" in user_context and user_context["today_meals"]:
                uc_lines.append(f"- Today's Logged Meals: {'; '.join(user_context['today_meals'])}")
            if "diabetes_type" in user_context:
                uc_lines.append(f"- Diagnosis: {user_context['diabetes_type']}")

            user_context_block = "\n".join(uc_lines) + "\n\n"

        # Format conversation memory history if provided
        history_block = ""
        if conversation_history:
            h_lines = ["Recent Advice Memory & Previous Recommendations:"]
            for h in conversation_history[-3:]:  # last 3 interactions
                q_text = h.get("question", "")
                a_text = h.get("answer", "")
                if len(a_text) > 150:
                    a_text = a_text[:150] + "..."
                h_lines.append(f"- User asked: \"{q_text}\" -> AI advised: \"{a_text}\"")
            history_block = "\n".join(h_lines) + "\n\n"

        if not self.gemini_api_key:
            return {
                "answer": (
                    "Based on ADA Standards of Care (2024), preprandial target glucose is 80–130 mg/dL "
                    "and peak postprandial glucose is <180 mg/dL. Ingesting 15g fast-acting carbohydrates "
                    "is recommended for hypoglycemia (Rule of 15).\n\n"
                    "⚠️ Disclaimer: Consult your personal physician or endocrinologist before altering any prescription medication or insulin dosages."
                ),
                "citations": [p.get("source", "ADA 2024") for p in passages],
                "retrieved_chunks": passages
            }

        prompt = f"""
You are the GlucoGuard Clinical AI Assistant, an empathetic, highly specialized diabetes guide.
Answer the patient's question accurately, personalized to their live health snapshot and recent advice history, strictly grounded in the official medical guidelines below.

{user_context_block}{history_block}Retrieved Clinical Guidelines:
{context_str}

Patient Question:
"{query}"

Guidelines:
1. Base your explanation strictly on the provided clinical guidelines (ADA, ICMR, WHO, CDC).
2. If the patient's real-time biometrics (glucose, steps, carbs, meals) are provided, directly reference and correlate them with the clinical advice (e.g., how their meal carbs or post-meal steps impact their current glucose reading).
3. If previous advice was given in the memory, maintain conversational consistency.
4. Explicitly cite the guideline source in your answer.
5. NEVER prescribe, modify, or suggest specific insulin dose units or prescription medication adjustments.
6. Always conclude with the mandatory safety disclaimer:
"⚠️ Medical Disclaimer: GlucoGuard provides clinical information for educational guidance. Always consult your healthcare provider or endocrinologist before modifying your insulin dosages or medical treatments."
"""

        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = await model.generate_content_async(prompt)
            return {
                "answer": response.text.strip(),
                "citations": list(set([p.get("source", "ADA Standards of Care") for p in passages])),
                "retrieved_chunks": passages
            }
        except Exception as e:
            return {
                "answer": f"Clinical RAG is currently offline: {e}",
                "citations": [],
                "retrieved_chunks": []
            }

    def _static_guidelines_fallback(self, query: str) -> List[Dict[str, Any]]:
        """Static clinical reference passages if database is not yet seeded."""
        return [
            {
                "source": "ADA Standards of Care in Diabetes 2024",
                "chapter": "Glycemic Targets & Hypoglycemia",
                "content": (
                    "ADA 2024 targets: Preprandial glucose 80–130 mg/dL; Peak postprandial <180 mg/dL. "
                    "Hypoglycemia (<70 mg/dL) treated with 15–20g fast-acting carbohydrate, recheck in 15 minutes."
                )
            },
            {
                "source": "ICMR Guidelines for Management of Type 2 Diabetes 2023",
                "chapter": "Dietary Interventions & Carbohydrate Profiling",
                "content": (
                    "Carbohydrates should not exceed 50-55% total energy intake. High fiber (>30-40g/day) "
                    "and eating protein/salad starters before rice or wheat blunts peak glucose spikes."
                )
            },
            {
                "source": "WHO Guidelines on Physical Activity and Diabetes",
                "chapter": "Post-Meal Exercise",
                "content": "A 10–20 minute brisk walk after meals enhances non-insulin-mediated GLUT4 glucose uptake."
            }
        ]
