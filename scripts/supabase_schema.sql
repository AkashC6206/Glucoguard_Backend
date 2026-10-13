-- =============================================================================
-- GlucoGuard - Clinical RAG & AI Vector Schema (Supabase PostgreSQL + pgvector)
-- =============================================================================
-- Run this in your Supabase SQL Editor (Dashboard -> SQL Editor -> New query)

-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Clinical Knowledge Corpus Table (ADA, ICMR, WHO, CDC Guidelines)
CREATE TABLE IF NOT EXISTS public.clinical_guidelines (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source TEXT NOT NULL,           -- e.g., 'ADA Standards of Care 2024', 'ICMR 2023'
    chapter TEXT,                   -- e.g., 'Glycemic Targets & Hypoglycemia'
    content TEXT NOT NULL,          -- Semantic guideline passage / chunk
    metadata JSONB DEFAULT '{}'::jsonb, -- e.g., {"url": "...", "page": 42}
    embedding VECTOR(768),          -- 768 dimensions for Gemini gemini-embedding-001
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Row Level Security (RLS) Policies
ALTER TABLE public.clinical_guidelines ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow read access to clinical_guidelines"
    ON public.clinical_guidelines
    FOR SELECT
    USING (true);

CREATE POLICY "Allow insert access to clinical_guidelines"
    ON public.clinical_guidelines
    FOR INSERT
    WITH CHECK (true);

-- 4. Fast Vector Index for Cosine Distance Search
CREATE INDEX IF NOT EXISTS clinical_guidelines_embedding_idx
    ON public.clinical_guidelines
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- 4. Vector Match RPC Function (Called by Clinical RAG Assistant)
CREATE OR REPLACE FUNCTION match_clinical_guidelines (
    query_embedding VECTOR(768),
    match_threshold FLOAT DEFAULT 0.60,
    match_count INT DEFAULT 4
)
RETURNS TABLE (
    id UUID,
    source TEXT,
    chapter TEXT,
    content TEXT,
    similarity FLOAT
)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    RETURN QUERY
    SELECT
        clinical_guidelines.id,
        clinical_guidelines.source,
        clinical_guidelines.chapter,
        clinical_guidelines.content,
        1 - (clinical_guidelines.embedding <=> query_embedding) AS similarity
    FROM public.clinical_guidelines
    WHERE 1 - (clinical_guidelines.embedding <=> query_embedding) > match_threshold
    ORDER BY clinical_guidelines.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- 5. Lifestyle Fingerprint Column on onboarding_profiles for non-PII peer matching
-- Vector format: [avg_glucose/200, daily_carbs/300, daily_steps/15000, sleep_hours/12, activity_freq/7]
ALTER TABLE IF EXISTS public.onboarding_profiles
    ADD COLUMN IF NOT EXISTS lifestyle_vector VECTOR(5);

-- 6. Peer Matching RPC function
CREATE OR REPLACE FUNCTION match_peer_lifestyle (
    target_vector VECTOR(5),
    match_count INT DEFAULT 5
)
RETURNS TABLE (
    user_id UUID,
    similarity FLOAT
)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    RETURN QUERY
    SELECT
        onboarding_profiles.user_id,
        1 - (onboarding_profiles.lifestyle_vector <=> target_vector) AS similarity
    FROM public.onboarding_profiles
    WHERE onboarding_profiles.lifestyle_vector IS NOT NULL
    ORDER BY onboarding_profiles.lifestyle_vector <=> target_vector
    LIMIT match_count;
END;
$$;

-- 7. AI Advice History & Memory Table
CREATE TABLE IF NOT EXISTS public.ai_advice_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    citations JSONB DEFAULT '[]'::jsonb,
    biometrics_snapshot JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE public.ai_advice_history ENABLE ROW LEVEL SECURITY;

-- RLS Policies
CREATE POLICY "Users can view own advice history"
    ON public.ai_advice_history FOR SELECT
    TO authenticated
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own advice history"
    ON public.ai_advice_history FOR INSERT
    TO authenticated
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own advice history"
    ON public.ai_advice_history FOR DELETE
    TO authenticated
    USING (auth.uid() = user_id);

