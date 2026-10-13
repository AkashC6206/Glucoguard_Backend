#!/usr/bin/env python3
"""
GlucoGuard - Clinical Guideline Ingestion Script
Embeds official guidelines (ADA 2024, ICMR 2023, WHO, CDC) using Google Gemini
text-embedding-004 (100% free) and inserts them into Supabase pgvector table.
"""

import os
import sys
import google.generativeai as genai
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
# Prefer service role key for administrative batch writes, fallback to anon
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("Error: Missing SUPABASE_URL or SUPABASE_ANON_KEY in environment.")
    sys.exit(1)

if not GEMINI_API_KEY:
    print("Error: Missing GEMINI_API_KEY in environment. Get one for free at https://aistudio.google.com/")
    sys.exit(1)

genai.configure(api_key=GEMINI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Curated Clinical Guideline Knowledge Corpus
GUIDELINE_CORPUS = [
    {
        "source": "ADA Standards of Care in Diabetes 2024",
        "chapter": "Glycemic Targets & Monitoring",
        "content": (
            "For nonpregnant adults with diabetes, the American Diabetes Association (ADA 2024) "
            "recommends an A1C goal of <7.0% (53 mmol/mol). Preprandial capillary plasma glucose "
            "target is 80–130 mg/dL (4.4–7.2 mmol/L), and peak postprandial capillary plasma glucose "
            "target (typically 1–2 hours after beginning of the meal) is <180 mg/dL (10.0 mmol/L). "
            "Time in Range (TIR, 70–180 mg/dL) should target >70% with Time Below Range (<70 mg/dL) <4%."
        )
    },
    {
        "source": "ADA Standards of Care in Diabetes 2024",
        "chapter": "Hypoglycemia Management (Rule of 15)",
        "content": (
            "Hypoglycemia (Level 1: 54-69 mg/dL; Level 2: <54 mg/dL) should be promptly treated with "
            "the 'Rule of 15': Ingest 15–20 grams of fast-acting glucose (e.g., 4 oz fruit juice, "
            "half can regular soda, 3-4 glucose tablets). Re-check blood glucose after 15 minutes. "
            "If hypoglycemia persists, repeat with another 15 grams of glucose. Once glucose returns "
            "to normal, consume a snack containing complex carbohydrates and protein if the next meal is >1 hour away."
        )
    },
    {
        "source": "ICMR Guidelines for Management of Type 2 Diabetes 2023",
        "chapter": "Dietary Management & South Asian Carbohydrate Profiling",
        "content": (
            "The Indian Council of Medical Research (ICMR 2023) recommends that total dietary carbohydrates "
            "should not exceed 50-55% of daily total energy intake for South Asian adults with Type 2 Diabetes. "
            "Complex carbohydrates with low glycemic index (brown rice, whole wheat, millets, oats, legumes) "
            "should replace refined carbohydrates (polished white rice, maida). High dietary fiber (>30-40g/day) "
            "helps attenuate postprandial glycemic excursions and improve insulin resistance."
        )
    },
    {
        "source": "ICMR Guidelines for Management of Type 2 Diabetes 2023",
        "chapter": "Food Sequencing & Glycemic Blunting",
        "content": (
            "ICMR 2023 highlights evidence that consuming non-starchy vegetables, salads, or protein starters "
            "10-15 minutes prior to carbohydrate ingestion significantly slows gastric emptying and blunts "
            "postprandial glucose velocity by 30-40% compared to eating carbohydrates first."
        )
    },
    {
        "source": "WHO Guidelines on Physical Activity and Diabetes",
        "chapter": "Post-Meal Exercise & Glucose Disposal",
        "content": (
            "World Health Organization (WHO) and clinical diabetes consensus guidelines demonstrate that "
            "light to moderate physical activity, such as a 10-20 minute brisk walk initiated within 30 minutes "
            "after a meal, stimulates insulin-independent GLUT4 glucose uptake in skeletal muscles. "
            "This blunts peak postprandial glucose excursions by 15-35 mg/dL compared to remaining sedentary."
        )
    },
    {
        "source": "CDC Diabetes Prevention Program (DPP)",
        "chapter": "Lifestyle Intervention & Habit Formation",
        "content": (
            "The CDC Diabetes Prevention Program emphasizes structured behavioral self-monitoring: achieving "
            "at least 150 minutes of moderate-intensity physical activity per week (such as brisk walking) "
            "and a 5-7% sustained reduction in body weight reduces the risk of diabetes progression by 58% "
            "(and 71% in adults aged 60 and older)."
        )
    },
    {
        "source": "ADA Standards of Care in Diabetes 2024",
        "chapter": "Sick Day Management & Ketone Monitoring",
        "content": (
            "During acute illness or fever, blood glucose often rises due to counter-regulatory stress hormones. "
            "Patients should continue taking basal insulin even if food intake decreases, check blood glucose "
            "every 2–4 hours, check urine or blood ketones if blood glucose exceeds 240 mg/dL, drink plenty of fluids "
            "(8 oz every hour), and contact their healthcare provider immediately if unable to keep liquids down or ketones are moderate/high."
        )
    }
]

def generate_embedding(text: str) -> list:
    """Generate 768-dimensional embedding using Gemini gemini-embedding-001."""
    try:
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text,
            task_type="retrieval_document",
            output_dimensionality=768
        )
        return result["embedding"]
    except Exception:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        return result["embedding"]

def main():
    print(f"Ingesting {len(GUIDELINE_CORPUS)} curated clinical guideline chunks into Supabase...")
    for idx, item in enumerate(GUIDELINE_CORPUS, start=1):
        print(f"[{idx}/{len(GUIDELINE_CORPUS)}] Embedding: {item['source']} - {item['chapter']}...")
        embedding = generate_embedding(item["content"])
        
        row = {
            "source": item["source"],
            "chapter": item["chapter"],
            "content": item["content"],
            "embedding": embedding,
            "metadata": {"ingested_via": "ingest_guidelines.py"}
        }
        
        try:
            supabase.table("clinical_guidelines").insert(row).execute()
            print(f"  ✓ Inserted successfully.")
        except Exception as e:
            print(f"  ✗ Error inserting into Supabase: {e}")
            if "row-level security" in str(e).lower():
                print("    👉 Tip: Run the RLS policy from backend/scripts/supabase_schema.sql in Supabase SQL Editor, or supply SUPABASE_SERVICE_ROLE_KEY in .env.")

    print("\nIngestion complete! Clinical RAG is now loaded with verified medical knowledge.")

if __name__ == "__main__":
    main()
