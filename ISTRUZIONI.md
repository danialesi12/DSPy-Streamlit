# DSPy Prompt Studio — istruzioni di setup e deploy

Il progetto è un container Docker standard: gira identico in locale, su Render o su Coolify — nessuna dipendenza da Streamlit Community Cloud (niente più limite di un'app privata, niente sintassi `secrets.toml` speciale). La configurazione passa tutta per variabili d'ambiente normali.

Struttura del progetto:

```
dspy-prompt-studio/
├── app.py                # UI Streamlit (punto di ingresso)
├── db.py                 # accesso a Postgres/Supabase (SQLAlchemy puro)
├── dspy_pipeline.py       # logica DSPy (draft / optimize)
├── schema.sql             # DDL da eseguire su Supabase
├── requirements.txt
├── Dockerfile
├── .env.example            # template variabili d'ambiente
└── .gitignore
```

## 1. Creare lo schema su Supabase

1. Supabase → **SQL Editor** → incolla tutto il contenuto di `schema.sql` → **Run**.
2. Verifica che sia comparso lo schema `promptstudio` (Table Editor → cambia schema da `public` a `promptstudio`).

Isolato dal resto (es. da Cal.com), stesso database che usi già.

## 2. Variabili d'ambiente richieste

Tre variabili, uguali ovunque tu faccia il deploy:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql+psycopg2://postgres:LA-TUA-PASSWORD@db.xxxxxxxxxxxx.supabase.co:5432/postgres
```

La connection string la trovi su Supabase: **Project Settings → Database → Connection string → URI** — aggiungi `+psycopg2` subito dopo `postgresql`.

## 3. Provare in locale

```bash
cd dspy-prompt-studio
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# apri .env e inserisci le tue chiavi vere + la connection string vera

streamlit run app.py
```

Si apre su `http://localhost:8501`. Crea un cliente di prova e lancia una run in modalità "draft" per verificare che tutto sia collegato prima di passare al deploy.

## 4. Deploy su Render

Render è la scelta più semplice: non tocca la tua VPS (quindi zero rischio di aggravare il carico CPU di cui abbiamo parlato per Cal.com/Stirling PDF), e i "Private Service" di Render non sono indicizzati né elencati pubblicamente da nessuna parte.

1. Push del progetto su un repository GitHub (`.env` resta escluso grazie al `.gitignore` — verifica comunque prima del push).
2. Su Render: **New → Web Service** → collega il repository.
3. Render rileva automaticamente il `Dockerfile` e lo usa per la build (non serve specificare altro come "build command"/"start command").
4. In **Environment**, aggiungi le tre variabili del punto 2.
5. **Instance Type**: il piano gratuito/più economico va benissimo per questo uso (poche run a settimana, traffico I/O-bound verso i provider LLM).
6. Deploy. Render ti dà un URL tipo `https://<nome-servizio>.onrender.com`.

Nota: sul piano gratuito di Render il servizio "dorme" dopo un periodo di inattività e si risveglia alla richiesta successiva (qualche secondo di attesa) — per un tool che usi tu a raffica quando serve, è un compromesso accettabile.

## 5. Deploy su Coolify (in alternativa)

Puoi usare Coolify allo stesso modo (Docker Compose/Dockerfile-based), collegando lo stesso repository e impostando le stesse tre variabili d'ambiente nella sezione "Environment Variables" della risorsa.

**Attenzione però**: se il Coolify in questione è quello sulla VPS di cui abbiamo parlato (6 vCPU, load average spesso 15-20 con Supabase + Cal.com + Stirling PDF già in esecuzione), aggiungere qui un altro servizio rischia di sommarsi a un sistema già sovraccarico — specialmente durante le run di ottimizzazione DSPy, che possono generare picchi di CPU. Se hai un'istanza Coolify separata (altra VPS, non quella satura), va benissimo; se è la stessa, Render resta la scelta più sicura per non aggravare la situazione.

## 6. Come si collega DSPy

- `dspy_pipeline.build_lm(provider, model, api_key)` costruisce un `dspy.LM("provider/model", api_key=...)` — la sintassi `provider/model` è la convenzione LiteLLM usata da DSPy, per questo cambiare fornitore è solo un cambio di stringa.
- **Modalità draft** (`run_draft`): nessun training set, il modulo `SupportAgent` gira zero-shot con il contesto (descrizione + knowledge base) così com'è. Per i clienti nuovi di cui hai solo una descrizione testuale.
- **Modalità optimize** (`run_optimize`): con esempi domanda/risposta salvati (split `train`), vengono trasformati in `dspy.Example` e usati da `BootstrapFewShot` per compilare un modulo migliore (seleziona i few-shot demo più efficaci, verificandoli con una metrica). La metrica di default (`answer_match_metric`) usa lo stesso LM come giudice — genera una chiamata aggiuntiva per ogni esempio valutato durante la compilazione.
- Ogni run salva il risultato in `promptstudio.runs`: il programma compilato serializzato (`compiled_program`, via `agent.dump_state()`/`agent.load_state()`) e una versione testuale leggibile (`compiled_prompt_preview`).
- Nella tab **Test & Export** ricarichi qualsiasi run passata, le fai una domanda di prova, ed esporti sia come JSON sia come testo semplice.

## 7. Aggiungere un nuovo provider LLM

1. In `app.py`, aggiungi una riga al dizionario `PROVIDERS` in cima al file.
2. Aggiungi la variabile d'ambiente corrispondente (es. `GOOGLE_API_KEY`) sia in locale (`.env`) sia nel servizio (Render/Coolify).

Non serve toccare `dspy_pipeline.py`: `build_lm` funziona con qualsiasi provider supportato da LiteLLM.

## 8. Note pratiche

- **Costi/tempo**: una run `optimize` con molti esempi può richiedere alcuni minuti e diverse chiamate LLM. Streamlit blocca la sessione durante l'esecuzione — accettabile per questo tool a uso personale.
- **Sicurezza**: `.env` non va mai su GitHub (escluso dal `.gitignore`). Le chiavi vere vivono solo nelle variabili d'ambiente del servizio (Render/Coolify) o nel file locale non tracciato.
- **Knowledge base grandi**: `build_context` oggi concatena tutto il testo. Con KB molto grandi per cliente, valuta un vero retrieval (embeddings + `pgvector`, già su Supabase) invece di mandare tutto nel contesto ad ogni chiamata.
