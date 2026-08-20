# Deployment Guide

This guide covers deploying the Candidate Evaluator as a multi-user web application.

## Architecture Overview

```
┌─────────────────┐         ┌─────────────────┐
│  Streamlit App  │         │  Railway Worker │
│  (Streamlit     │         │  (Background    │
│   Cloud - FREE) │         │   jobs - FREE)  │
└────────┬────────┘         └────────┬────────┘
         │                           │
         │  1. Queue job             │  2. Poll & process
         │  4. Check status          │  3. Save results
         ▼                           ▼
┌─────────────────────────────────────────────┐
│              Supabase (FREE)                │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  │
│  │  Auth   │  │   Jobs   │  │  Results  │  │
│  │         │  │  Queue   │  │           │  │
│  └─────────┘  └──────────┘  └───────────┘  │
│                                             │
│  ┌──────────────────────────────────────┐  │
│  │         File Storage                  │  │
│  │    (uploaded candidate PDFs)          │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

## Step 1: Set Up Supabase (Free)

1. **Create Account**: Go to [supabase.com](https://supabase.com) and sign up

2. **Create Project**: Create a new project, note the:
   - **Project URL**: `https://xxxxx.supabase.co`
   - **Anon Key**: Found in Settings → API → `anon` `public`
   - **Service Role Key**: Found in Settings → API → `service_role` (for Railway worker)

3. **Run Database Schema**: 
   - Go to SQL Editor in Supabase Dashboard
   - Copy contents of `supabase_schema.sql` and run it
   - For an **existing deployment**, don't re-run the whole file (the
     `CREATE POLICY` statements will error) — run only the `MIGRATION`
     statements at the bottom of `supabase_schema.sql`, e.g.
     `ALTER TABLE jobs ADD COLUMN IF NOT EXISTS config JSONB NOT NULL DEFAULT '{}';`
     which is required by the Screening feature

4. **Create Storage Bucket**:
   - Go to Storage in Supabase Dashboard
   - Click "New Bucket"
   - Name: `candidate-materials`
   - Public: **No** (keep private)

## Step 2: Deploy Streamlit App (Free)

1. **Push to GitHub**: 
   ```bash
   git add .
   git commit -m "Add cloud deployment support"
   git push origin main
   ```

2. **Connect to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Main file path: `candidate_evaluator/web_app_cloud.py`

3. **Add Secrets**:
   - In Streamlit Cloud, go to your app → Settings → Secrets
   - Add:
   ```toml
   SUPABASE_URL = "https://your-project.supabase.co"
   SUPABASE_KEY = "your-anon-key"
   ```

4. **Deploy**: Click Deploy!

## Step 3: Deploy Railway Worker (Free)

The Railway worker processes batch jobs in the background.

1. **Create Railway Account**: Go to [railway.app](https://railway.app) and sign up

2. **Create New Project**:
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Connect your repository

3. **Configure Service**:
   - Railway will auto-detect the `Procfile`
   - It will run: `python -m candidate_evaluator.worker`

4. **Add Environment Variables**:
   - Go to your service → Variables
   - Add:
   ```
   SUPABASE_URL = https://your-project.supabase.co
   SUPABASE_KEY = your-service-role-key  # Use SERVICE ROLE key, not anon!
   ```

5. **Deploy**: Railway will automatically deploy

## Environment Variables Reference

### Streamlit Cloud (web app)
| Variable | Description | Where to Find |
|----------|-------------|---------------|
| `SUPABASE_URL` | Supabase project URL | Supabase → Settings → API |
| `SUPABASE_KEY` | Supabase **anon** key | Supabase → Settings → API |

### Railway (worker)
| Variable | Description | Where to Find |
|----------|-------------|---------------|
| `SUPABASE_URL` | Supabase project URL | Supabase → Settings → API |
| `SUPABASE_KEY` | Supabase **service role** key | Supabase → Settings → API |
| `EVAL_CONCURRENCY` | Optional: concurrent Claude calls per job (default 4). Raise for faster batches if your Anthropic rate limits allow; set 1 to run serially | — |

**Important**: The worker uses the **service role key** which bypasses Row Level Security. This allows it to read jobs from all users.

## How It Works

1. **User signs up/logs in** via Supabase Auth
2. **User enters their Anthropic API key** (stored encrypted in their profile)
3. **Single evaluations** run immediately in the Streamlit app
4. **Batch evaluations**:
   - PDFs uploaded to Supabase Storage
   - Job created in `jobs` table with status `pending`
   - Railway worker polls for pending jobs
   - Worker claims job, downloads files, runs evaluations
   - Results saved to `evaluations` table
   - User can check progress or close browser

## Cost Analysis (Free Tier Limits)

| Service | Free Tier | Limit |
|---------|-----------|-------|
| **Supabase** | Database | 500 MB |
| | Storage | 1 GB |
| | Auth | 50,000 MAU |
| **Streamlit Cloud** | Public apps | Unlimited |
| **Railway** | Execution | $5/month credit |
| **Anthropic API** | None | Users pay their own |

### When You Might Exceed Free Tier

- **500+ candidates stored** → May approach database limit
- **1000+ PDFs uploaded** → May approach storage limit
- **Heavy batch processing** → May exceed Railway free credit

## Local Development

1. **Create `.streamlit/secrets.toml`**:
   ```toml
   SUPABASE_URL = "https://your-project.supabase.co"
   SUPABASE_KEY = "your-anon-key"
   ```

2. **Run the app**:
   ```bash
   streamlit run candidate_evaluator/web_app_cloud.py
   ```

3. **Run the worker** (in separate terminal):
   ```bash
   export SUPABASE_URL="https://your-project.supabase.co"
   export SUPABASE_KEY="your-service-role-key"
   python -m candidate_evaluator.worker
   ```

## Troubleshooting

### "Supabase credentials not found"
- Check that secrets are properly configured in Streamlit Cloud
- For local dev, ensure `.streamlit/secrets.toml` exists

### Worker not processing jobs
- Check Railway logs for errors
- Verify the service role key is correct
- Ensure the worker is running (not crashed)

### File upload fails
- Check that the `candidate-materials` bucket exists
- Verify storage policies are set up correctly

### Authentication issues
- Make sure the Supabase schema was run completely
- Check that the `on_auth_user_created` trigger exists

## Security Notes

1. **API Keys**: Each user's Anthropic API key is stored in their Supabase profile. Supabase encrypts data at rest.

2. **Row Level Security**: Users can only see their own data. The worker uses a service role key to access all data.

3. **File Access**: Files are stored in user-specific folders. Storage policies ensure users can only access their own files.
