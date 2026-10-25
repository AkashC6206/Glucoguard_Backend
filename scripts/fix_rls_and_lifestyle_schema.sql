-- =============================================================================
-- GlucoGuard - Complete Database & RLS Migration Script
-- Fixes food_logs RLS 42501 error and enables Sleep, Mood, Timeline & Community
-- Execute in Supabase SQL Editor: Dashboard -> SQL Editor -> New Query -> Run
-- =============================================================================

-- 1. FIX: Food Logs Row Level Security (RLS)
ALTER TABLE IF EXISTS public.food_logs ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can manage own food logs" ON public.food_logs;
CREATE POLICY "Users can manage own food logs" ON public.food_logs
    FOR ALL
    TO authenticated
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

-- Fallback policy for anon / local testing if needed
DROP POLICY IF EXISTS "Anon can insert food logs" ON public.food_logs;
CREATE POLICY "Anon can insert food logs" ON public.food_logs
    FOR ALL
    TO anon
    USING (true)
    WITH CHECK (true);


-- 2. SLEEP TRACKING TABLE & POLICIES
CREATE TABLE IF NOT EXISTS public.sleep_logs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    sleep_start TIMESTAMPTZ NOT NULL,
    sleep_end TIMESTAMPTZ NOT NULL,
    duration_minutes INTEGER NOT NULL,
    quality_rating INTEGER CHECK (quality_rating BETWEEN 1 AND 5),
    source TEXT DEFAULT 'manual' CHECK (source IN ('manual', 'wearable', 'health_connect', 'healthkit')),
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE public.sleep_logs ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can manage own sleep logs" ON public.sleep_logs;
CREATE POLICY "Users can manage own sleep logs" ON public.sleep_logs
    FOR ALL
    TO authenticated
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Anon can manage sleep logs" ON public.sleep_logs;
CREATE POLICY "Anon can manage sleep logs" ON public.sleep_logs
    FOR ALL
    TO anon
    USING (true)
    WITH CHECK (true);


-- 3. WELLBEING & MOOD TRACKING TABLE & POLICIES
CREATE TABLE IF NOT EXISTS public.wellbeing_logs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    mood TEXT NOT NULL CHECK (mood IN ('great', 'good', 'neutral', 'low', 'anxious')),
    stress_level INTEGER NOT NULL CHECK (stress_level BETWEEN 1 AND 5),
    perceived_energy INTEGER CHECK (perceived_energy BETWEEN 1 AND 5),
    notes TEXT,
    logged_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE public.wellbeing_logs ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can manage own wellbeing logs" ON public.wellbeing_logs;
CREATE POLICY "Users can manage own wellbeing logs" ON public.wellbeing_logs
    FOR ALL
    TO authenticated
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Anon can manage wellbeing logs" ON public.wellbeing_logs;
CREATE POLICY "Anon can manage wellbeing logs" ON public.wellbeing_logs
    FOR ALL
    TO anon
    USING (true)
    WITH CHECK (true);


-- 4. ADAPTIVE LIFESTYLE SUGGESTIONS & PLANNER
CREATE TABLE IF NOT EXISTS public.adaptive_lifestyle_suggestions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    category TEXT NOT NULL CHECK (category IN ('nutrition', 'physical_activity', 'sleep_hygiene', 'routine')),
    suggestion_title TEXT NOT NULL,
    suggestion_text TEXT NOT NULL,
    clinical_rationale TEXT NOT NULL,
    guidance_source TEXT,
    is_dismissed BOOLEAN DEFAULT FALSE,
    is_completed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE public.adaptive_lifestyle_suggestions ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can manage own lifestyle suggestions" ON public.adaptive_lifestyle_suggestions;
CREATE POLICY "Users can manage own lifestyle suggestions" ON public.adaptive_lifestyle_suggestions
    FOR ALL
    TO authenticated
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "Anon can manage lifestyle suggestions" ON public.adaptive_lifestyle_suggestions;
CREATE POLICY "Anon can manage lifestyle suggestions" ON public.adaptive_lifestyle_suggestions
    FOR ALL
    TO anon
    USING (true)
    WITH CHECK (true);


-- 5. COMMUNITY FORUM POSTS & COMMENTS
CREATE TABLE IF NOT EXISTS public.forum_posts (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'General',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.forum_comments (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    post_id UUID NOT NULL REFERENCES public.forum_posts(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE public.forum_posts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.forum_comments ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Anyone can read forum posts" ON public.forum_posts;
CREATE POLICY "Anyone can read forum posts" ON public.forum_posts FOR SELECT USING (true);

DROP POLICY IF EXISTS "Users can insert forum posts" ON public.forum_posts;
CREATE POLICY "Users can insert forum posts" ON public.forum_posts FOR INSERT WITH CHECK (true);

DROP POLICY IF EXISTS "Users can delete own forum posts" ON public.forum_posts;
CREATE POLICY "Users can delete own forum posts" ON public.forum_posts FOR DELETE USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Anyone can read forum comments" ON public.forum_comments;
CREATE POLICY "Anyone can read forum comments" ON public.forum_comments FOR SELECT USING (true);

DROP POLICY IF EXISTS "Users can insert forum comments" ON public.forum_comments;
CREATE POLICY "Users can insert forum comments" ON public.forum_comments FOR INSERT WITH CHECK (true);

DROP POLICY IF EXISTS "Users can delete own forum comments" ON public.forum_comments;
CREATE POLICY "Users can delete own forum comments" ON public.forum_comments FOR DELETE USING (auth.uid() = user_id);


-- 6. FORUM POST VOTES
CREATE TABLE IF NOT EXISTS public.forum_post_votes (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    post_id UUID NOT NULL REFERENCES public.forum_posts(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    vote_type TEXT NOT NULL CHECK (vote_type IN ('helpful', 'unhelpful')),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(post_id, user_id)
);

ALTER TABLE public.forum_post_votes ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Anyone can read votes" ON public.forum_post_votes;
CREATE POLICY "Anyone can read votes" ON public.forum_post_votes FOR SELECT USING (true);

DROP POLICY IF EXISTS "Users can vote" ON public.forum_post_votes;
CREATE POLICY "Users can vote" ON public.forum_post_votes FOR ALL USING (true) WITH CHECK (true);
