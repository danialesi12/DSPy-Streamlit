-- DSPy Prompt Studio -- schema dedicato, da eseguire sul Postgres di Supabase
-- (SQL Editor di Supabase -> incolla tutto -> Run)

create extension if not exists pgcrypto;  -- serve per gen_random_uuid()

create schema if not exists promptstudio;

create table if not exists promptstudio.clients (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  business_description text,        -- contesto libero usato per il cold-start
  created_at timestamptz not null default now()
);

create table if not exists promptstudio.llm_configs (
  id uuid primary key default gen_random_uuid(),
  client_id uuid not null references promptstudio.clients(id) on delete cascade,
  provider text not null,           -- 'anthropic' | 'openai' | ... (convenzione LiteLLM)
  model text not null,               -- es. 'claude-sonnet-4-5' / 'gpt-4o'
  is_default boolean not null default false,
  created_at timestamptz not null default now()
);

create table if not exists promptstudio.kb_documents (
  id uuid primary key default gen_random_uuid(),
  client_id uuid not null references promptstudio.clients(id) on delete cascade,
  title text,
  content text not null,              -- testo grezzo, concatenato nel contesto
  created_at timestamptz not null default now()
);

create table if not exists promptstudio.examples (
  id uuid primary key default gen_random_uuid(),
  client_id uuid not null references promptstudio.clients(id) on delete cascade,
  question text not null,
  expected_answer text not null,
  context text,                        -- override del contesto per questo esempio (opzionale)
  split text not null default 'train', -- 'train' | 'val'
  created_at timestamptz not null default now()
);

create table if not exists promptstudio.runs (
  id uuid primary key default gen_random_uuid(),
  client_id uuid not null references promptstudio.clients(id) on delete cascade,
  mode text not null,                  -- 'draft' | 'optimize'
  llm_config_id uuid references promptstudio.llm_configs(id),
  status text not null default 'pending', -- pending | running | success | failed
  metrics jsonb,
  compiled_program jsonb,               -- stato DSPy serializzato (dump_state())
  compiled_prompt_preview text,          -- versione leggibile per revisione umana
  error text,
  created_at timestamptz not null default now(),
  finished_at timestamptz
);

create index if not exists idx_llm_configs_client on promptstudio.llm_configs(client_id);
create index if not exists idx_kb_documents_client on promptstudio.kb_documents(client_id);
create index if not exists idx_examples_client on promptstudio.examples(client_id);
create index if not exists idx_runs_client on promptstudio.runs(client_id);
