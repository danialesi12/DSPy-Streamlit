# DSPy Prompt Studio — istruzioni di setup e deploy

Il progetto è un container Docker standard: gira identico in locale, su Render o su Coolify — nessuna dipendenza da Streamlit Community Cloud (niente più limite di un'app privata, niente sintassi `secrets.toml` speciale). La configurazione passa tutta per variabili d'ambiente normali.

Struttura del progetto:
dspy-prompt-studio/
├── app.py # UI Streamlit (punto di ingresso)
├── db.py # accesso a Postgres/Supabase (SQLAlchemy puro)
├── dspy_pipeline.py # logica DSPy (draft / optimize)
├── schema.sql # DDL da eseguire su Supabase
├── requirements.txt
├── Dockerfile
├── .env.example # template variabili d'ambiente
└── .gitignore


## 1. Creare lo schema su Supabase

1. Supabase → **SQL Editor** → incolla tutto il contenuto di `schema.sql` → **Run**.
2. Verifica che sia comparso lo schema `promptstudio` (Table Editor → cambia schema da `public` a `promptstudio`).

Isolato dal resto (es. da Cal.com), stesso database che usi già.

## 2. Variabili d'ambiente richieste

Due variabili, uguali ovunque tu faccia il deploy:
