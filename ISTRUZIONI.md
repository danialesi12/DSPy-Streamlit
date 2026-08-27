# DSPy Prompt Studio — istruzioni di setup e collegamento

Struttura del progetto:

```
dspy-prompt-studio/
├── app.py                        # UI Streamlit (punto di ingresso)
├── db.py                         # accesso a Postgres/Supabase
├── dspy_pipeline.py               # logica DSPy (draft / optimize)
├── schema.sql                     # DDL da eseguire su Supabase
├── requirements.txt
├── .gitignore
└── .streamlit/
    └── secrets.toml.example       # template, NON il file vero
```

## 1. Creare lo schema su Supabase

1. Apri il tuo progetto Supabase → **SQL Editor**.
2. Incolla tutto il contenuto di `schema.sql` e premi **Run**.
3. Verifica che sia comparso lo schema `promptstudio` con le tabelle `clients`, `llm_configs`, `kb_documents`, `examples`, `runs` (puoi controllarlo da **Table Editor** cambiando lo schema in alto da `public` a `promptstudio`).

Non serve toccare nient'altro sul lato Supabase: l'app scrive/legge solo in questo schema, isolato dal resto (es. dallo schema di Cal.com).

## 2. Recuperare la connection string

Su Supabase: **Project Settings → Database → Connection string → URI**. Avrai qualcosa tipo:

```
postgresql://postgres:LA-TUA-PASSWORD@db.xxxxxxxxxxxx.supabase.co:5432/postgres
```

Per usarla con SQLAlchemy/psycopg2 (richiesto da `st.connection`), aggiungi `+psycopg2` subito dopo `postgresql`:

```
postgresql+psycopg2://postgres:LA-TUA-PASSWORD@db.xxxxxxxxxxxx.supabase.co:5432/postgres
```

Questa è la stringa da mettere in `secrets.toml` sotto `[connections.promptstudio_db]` → `url`.

## 3. Provare in locale (consigliato prima di mettere online)

```bash
cd dspy-prompt-studio
python3 -m venv .venv
source .venv/bin/activate          # su Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# apri .streamlit/secrets.toml e inserisci le tue chiavi API + la connection string vera

streamlit run app.py
```

Si apre in automatico su `http://localhost:8501`. Crea un cliente di prova, aggiungi una descrizione, configura un provider LLM e prova una run in modalità "draft" per verificare che tutto sia collegato correttamente prima di passare al deploy.

## 4. Deploy su Streamlit Community Cloud (gratis, app privata)

1. Crea un repository su GitHub e carica questi file **senza** `.streamlit/secrets.toml` (resta escluso grazie al `.gitignore` — verifica comunque prima del push che non sia stato aggiunto per errore).
2. Vai su [streamlit.io/cloud](https://streamlit.io/cloud), accedi con GitHub.
3. **New app** → seleziona il repository, il branch, e `app.py` come file principale.
4. Prima del deploy, apri **Advanced settings** e imposta l'app come **Private** — hai diritto a **una** app privata gratuita per workspace, che è esattamente il caso d'uso di questo tool. Le app private non vengono indicizzate dai motori di ricerca e richiedono comunque l'URL diretto per essere aperte.
5. Sempre in Advanced settings, sezione **Secrets**: incolla il contenuto del tuo `secrets.toml` locale (con i valori veri, non il file `.example`).
6. Fai partire il deploy. Al termine avrai un URL tipo `https://<nome-app>.streamlit.app`.

Ogni volta che fai `git push` sul branch collegato, l'app si aggiorna automaticamente in pochi secondi.

## 5. Come si collega DSPy (spiegazione del funzionamento)

- `dspy_pipeline.build_lm(provider, model, api_key)` costruisce un `dspy.LM("provider/model", api_key=...)` — la sintassi `provider/model` è la convenzione LiteLLM che DSPy usa sotto il cofano, per questo cambiare fornitore è solo un cambio di stringa.
- **Modalità draft** (`run_draft`): nessun training set, il modulo `SupportAgent` gira zero-shot con il contesto (descrizione + knowledge base) passato così com'è. Va bene per i clienti nuovi di cui hai solo una descrizione testuale.
- **Modalità optimize** (`run_optimize`): quando ci sono esempi domanda/risposta salvati con split `train`, vengono trasformati in `dspy.Example` e usati da `BootstrapFewShot` per compilare un modulo migliorato (seleziona i few-shot demo più efficaci, verificandoli con una metrica). La metrica di default (`answer_match_metric`) usa lo stesso LM come "giudice" per stabilire se una risposta è equivalente a quella attesa — genera una chiamata LLM aggiuntiva per ogni esempio valutato durante la compilazione, tienilo presente per il costo/tempo se il dataset cresce molto.
- Ogni run (draft o optimize) salva il risultato in `promptstudio.runs`: sia il programma compilato serializzato (`compiled_program`, tramite `agent.dump_state()`/`agent.load_state()` — non serve mai toccare file su disco) sia una versione testuale leggibile (`compiled_prompt_preview`) per rivederlo senza dover ricaricare DSPy.
- Nella tab **Test & Export** puoi ricaricare qualsiasi run passata, farle una domanda di prova, ed esportarla sia come JSON (se il sistema di destinazione userà anche lui DSPy) sia come testo semplice (prompt + esempi, utilizzabile in qualsiasi altro stack).

## 6. Aggiungere un nuovo provider LLM

1. In `app.py`, aggiungi una riga al dizionario `PROVIDERS` in cima al file, es.:
   ```python
   PROVIDERS = {
       "anthropic": [...],
       "openai": [...],
       "google": ["gemini-2.5-pro"],  # nuovo
   }
   ```
2. Aggiungi il secret corrispondente (`GOOGLE_API_KEY`) sia in locale che nei Secrets dell'app su Streamlit Cloud.

Non serve toccare `dspy_pipeline.py`: `build_lm` funziona automaticamente con qualsiasi provider supportato da LiteLLM.

## 7. Note pratiche

- **Costi/tempo**: una run in modalità `optimize` con molti esempi può richiedere alcuni minuti e diverse chiamate LLM (sia per generare le risposte candidate sia per la metrica di validazione). Streamlit blocca la sessione durante l'esecuzione — è previsto e accettabile per questo tool a uso personale.
- **Sicurezza**: non committare mai `.streamlit/secrets.toml` reale su GitHub — solo `secrets.toml.example`. Le chiavi vere vivono solo nei Secrets dell'app su Streamlit Cloud (o nel file locale non tracciato).
- **Knowledge base grandi**: `build_context` oggi concatena tutto il testo. Se in futuro le KB per cliente diventano molto grandi (centinaia di documenti), conviene passare a un vero retrieval (embeddings + `pgvector`, già disponibile su Supabase) invece di mandare tutto nel contesto ad ogni chiamata.
- **Risorse su Streamlit Community Cloud**: se vedi l'errore "this app has gone over its resource limits" durante run di ottimizzazione pesanti, è il segnale per spostare l'app su un hosting con risorse dedicate (es. un piccolo VPS separato via Docker).
