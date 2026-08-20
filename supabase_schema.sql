-- Supabase Schema for Candidate Evaluator
-- Run this in your Supabase SQL Editor (Dashboard -> SQL Editor -> New Query)

-- ============================================================================
-- PROFILES TABLE
-- Stores user profile data and their Anthropic API key
-- ============================================================================
CREATE TABLE IF NOT EXISTS profiles (
    id UUID REFERENCES auth.users(id) PRIMARY KEY,
    email TEXT,
    anthropic_api_key TEXT,  -- User's own API key (stored securely)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- JOBS TABLE
-- Stores batch job queue for background processing
-- ============================================================================
CREATE TABLE IF NOT EXISTS jobs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES profiles(id) NOT NULL,
    job_name TEXT,
    job_type TEXT NOT NULL DEFAULT 'batch',
    status TEXT NOT NULL DEFAULT 'pending',  -- pending, processing, completed, failed, cancelled
    evaluation_mode TEXT NOT NULL DEFAULT 'criteria',  -- criteria, holistic, or screen
    role TEXT,  -- LEGACY: read as fallback only; new jobs put role in config
    config JSONB NOT NULL DEFAULT '{}',  -- mode-specific parameters (role, screen_description, ...)
    total_candidates INTEGER NOT NULL DEFAULT 0,
    completed_candidates INTEGER NOT NULL DEFAULT 0,
    failed_candidates INTEGER NOT NULL DEFAULT 0,
    current_candidate TEXT,  -- Currently processing candidate ID
    file_paths JSONB NOT NULL DEFAULT '[]',  -- Array of storage paths
    worker_id TEXT,  -- ID of worker processing this job
    error TEXT,  -- Error message if failed
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- ============================================================================
-- EVALUATIONS TABLE
-- Stores evaluation results
-- ============================================================================
CREATE TABLE IF NOT EXISTS evaluations (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES profiles(id) NOT NULL,
    job_id UUID REFERENCES jobs(id),  -- Optional, for batch jobs
    candidate_id TEXT NOT NULL,
    candidate_name TEXT,
    evaluation_type TEXT NOT NULL,  -- 'criteria' or 'holistic'
    role TEXT,  -- optional: clinician, engineer, phd
    role_specific_score NUMERIC,  -- score from role_specific_assessment
    result JSONB NOT NULL,  -- Full evaluation result
    overall_score NUMERIC,
    recommendation TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- UPLOADED FILES TABLE
-- Tracks uploaded file metadata
-- ============================================================================
CREATE TABLE IF NOT EXISTS uploaded_files (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES profiles(id) NOT NULL,
    evaluation_id UUID REFERENCES evaluations(id),
    job_id UUID REFERENCES jobs(id),
    file_name TEXT NOT NULL,
    storage_path TEXT NOT NULL,  -- Path in Supabase Storage
    file_type TEXT,
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_jobs_user_id ON jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_evaluations_user_id ON evaluations(user_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_job_id ON evaluations(job_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_created_at ON evaluations(created_at);
CREATE INDEX IF NOT EXISTS idx_uploaded_files_user_id ON uploaded_files(user_id);
CREATE INDEX IF NOT EXISTS idx_uploaded_files_job_id ON uploaded_files(job_id);

-- ============================================================================
-- ROW LEVEL SECURITY (RLS)
-- Users can only see/modify their own data
-- ============================================================================
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE evaluations ENABLE ROW LEVEL SECURITY;
ALTER TABLE uploaded_files ENABLE ROW LEVEL SECURITY;

-- Profiles: Users can only access their own profile
CREATE POLICY "Users can view own profile"
    ON profiles FOR SELECT
    USING (auth.uid() = id);

CREATE POLICY "Users can update own profile"
    ON profiles FOR UPDATE
    USING (auth.uid() = id);

-- Jobs: Users can only access their own jobs
CREATE POLICY "Users can view own jobs"
    ON jobs FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can create own jobs"
    ON jobs FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own jobs"
    ON jobs FOR UPDATE
    USING (auth.uid() = user_id);

-- Evaluations: Users can only access their own evaluations
CREATE POLICY "Users can view own evaluations"
    ON evaluations FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can create own evaluations"
    ON evaluations FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own evaluations"
    ON evaluations FOR DELETE
    USING (auth.uid() = user_id);

-- Uploaded files: Users can only access their own files
CREATE POLICY "Users can view own files"
    ON uploaded_files FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can create own files"
    ON uploaded_files FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own files"
    ON uploaded_files FOR DELETE
    USING (auth.uid() = user_id);

-- ============================================================================
-- SERVICE ROLE POLICIES FOR WORKER
-- The worker uses the service role key, which bypasses RLS
-- No additional policies needed for the worker
-- ============================================================================

-- ============================================================================
-- AUTO-CREATE PROFILE ON USER SIGNUP
-- ============================================================================
CREATE OR REPLACE FUNCTION handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO profiles (id, email)
    VALUES (NEW.id, NEW.email);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Drop trigger if exists and recreate
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION handle_new_user();

-- ============================================================================
-- STORAGE BUCKET SETUP
-- Run these in Supabase Dashboard -> Storage -> Create bucket
-- ============================================================================
-- 1. Create a bucket named "candidate-materials" (private)
-- 2. Add this storage policy in SQL:

-- Storage policy for candidate-materials bucket
-- Users can only access their own files (files in their user_id folder)

-- INSERT policy: Users can upload to their own folder
CREATE POLICY "Users can upload to own folder"
    ON storage.objects FOR INSERT
    WITH CHECK (
        bucket_id = 'candidate-materials' 
        AND (storage.foldername(name))[1] = auth.uid()::text
    );

-- SELECT policy: Users can view their own files
CREATE POLICY "Users can view own files"
    ON storage.objects FOR SELECT
    USING (
        bucket_id = 'candidate-materials' 
        AND (storage.foldername(name))[1] = auth.uid()::text
    );

-- DELETE policy: Users can delete their own files
CREATE POLICY "Users can delete own files"
    ON storage.objects FOR DELETE
    USING (
        bucket_id = 'candidate-materials' 
        AND (storage.foldername(name))[1] = auth.uid()::text
    );

-- ============================================================================
-- MIGRATION: Add role column to jobs (for existing deployments)
-- ============================================================================
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS role TEXT;

-- ============================================================================
-- MIGRATION: Add role columns to evaluations (for existing deployments)
-- ============================================================================
ALTER TABLE evaluations ADD COLUMN IF NOT EXISTS role TEXT;
ALTER TABLE evaluations ADD COLUMN IF NOT EXISTS role_specific_score NUMERIC;
CREATE INDEX IF NOT EXISTS idx_evaluations_role ON evaluations(role);
CREATE INDEX IF NOT EXISTS idx_evaluations_role_specific_score ON evaluations(role_specific_score);

-- ============================================================================
-- MIGRATION: Add config payload to jobs (for existing deployments)
-- Mode-specific job parameters (role, screening's screen_description, and
-- anything future modes need) live in this JSONB payload, so new modes don't
-- require schema migrations. evaluation_mode may now also be 'screen'.
-- The legacy jobs.role column is kept and read as a fallback for old rows.
-- ============================================================================
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS config JSONB NOT NULL DEFAULT '{}';
