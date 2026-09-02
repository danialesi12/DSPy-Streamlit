"""
Logica DSPy: definizione del modulo "risponditore", costruzione dell'LM
multi-provider, e le due modalità richieste (draft / optimize).

Convenzioni:
- `provider` segue la sintassi LiteLLM usata da dspy.LM: "anthropic", "openai", ecc.
- Il "programma compilato" viene serializzato con agent.dump_state() (dict JSON-friendly)
  e ricaricato con agent.load_state(state) — non serve toccare il filesystem.
- Esistono due strategie di retrieval, scelte a run-time (vedi app.py):
  "full_context" (SupportAgent, storica: tutta la KB concatenata in una stringa)
  e "tools" (SupportAgentWithTools: un dspy.ReAct che recupera solo le pagine
  pertinenti tramite get_index_pages/get_page_by_id). I tool sono chiusi su
  un client_id specifico (vedi build_kb_tools) perché l'app è multi-tenant:
  non sono funzioni globali come in un agente a singolo cliente.
"""

from __future__ import annotations

import dspy
from dspy.teleprompt import BootstrapFewShot

import db


class AnswerCustomerQuestion(dspy.Signature):
    """Rispondi alla domanda del cliente usando esclusivamente le informazioni
    fornite nel contesto aziendale. Se l'informazione non è presente nel
    contesto, dillo chiaramente invece di inventare una risposta."""

    business_context: str = dspy.InputField(
        desc="Knowledge base e descrizione dell'attività del cliente"
    )
    question: str = dspy.InputField(desc="Domanda posta dall'utente finale")
    answer: str = dspy.OutputField(desc="Risposta basata solo sul contesto fornito")


class SupportAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.respond = dspy.ChainOfThought(AnswerCustomerQuestion)

    def forward(self, business_context: str, question: str):
        return self.respond(business_context=business_context, question=question)


class AnswerCustomerQuestionWithTools(dspy.Signature):
    """Rispondi alla domanda del cliente usando i tool per recuperare le
    informazioni pertinenti dalla knowledge base. Non rispondere mai basandoti
    solo sui titoli restituiti da get_index_pages: per qualsiasi domanda
    specifica devi prima chiamare get_page_by_id sulla pagina più pertinente.
    Se dopo aver consultato le pagine rilevanti l'informazione non c'è,
    dillo chiaramente invece di inventare una risposta."""

    question: str = dspy.InputField(desc="Domanda posta dall'utente finale")
    answer: str = dspy.OutputField(desc="Risposta basata solo sul contenuto recuperato dai tool")


class SupportAgentWithTools(dspy.Module):
    """Variante di SupportAgent che recupera il contesto con dei tool
    (dspy.ReAct) invece di ricevere l'intera KB concatenata in un campo.
    I tool vanno costruiti per il cliente corrente con build_kb_tools()
    e passati qui: il modulo in sé non sa nulla di quale cliente sia."""

    def __init__(self, tools: list[dspy.Tool], max_iters: int = 4):
        super().__init__()
        self.respond = dspy.ReAct(AnswerCustomerQuestionWithTools, tools=tools, max_iters=max_iters)

    def forward(self, question: str):
        return self.respond(question=question)


def build_kb_tools(client_id: str) -> list[dspy.Tool]:
    """Costruisce i tool di retrieval per UN cliente specifico. Leggono la KB
    da Postgres a ogni chiamata (nessuno stato tenuto in memoria tra una
    chiamata e l'altra): sicuro anche perché Streamlit esegue ogni rerun in
    un thread nuovo (vedi il commento su dspy.context in run_draft)."""

    def get_index_pages() -> str:
        """Elenco di id e titolo di ogni pagina della knowledge base di
        questo cliente. Usalo per capire quale pagina è pertinente prima
        di chiamare get_page_by_id: non contiene il testo completo."""
        docs = db.list_kb_documents(client_id)
        if docs.empty:
            return "Nessun documento in knowledge base."
        return "\n".join(
            f"- id={row.id} — {row.title or '(senza titolo)'}" for row in docs.itertuples()
        )

    def get_page_by_id(page_id: str) -> str:
        """Contenuto completo di una pagina della knowledge base. page_id deve
        essere copiato ESATTAMENTE da get_index_pages: non inventarlo, non
        abbreviarlo, non dedurlo per somiglianza."""
        docs = db.list_kb_documents(client_id)
        match = docs[docs["id"].astype(str) == str(page_id)]
        if match.empty:
            return "Id non trovato: richiama get_index_pages per la lista aggiornata."
        return match.iloc[0]["content"]

    return [
        dspy.Tool(get_index_pages, name="get_index_pages"),
        dspy.Tool(
            get_page_by_id,
            name="get_page_by_id",
            arg_desc={"page_id": "Id esatto, copiato letteralmente da get_index_pages"},
        ),
    ]


def build_lm(provider: str, model: str, api_key: str, **kwargs) -> dspy.LM:
    """Costruisce un dspy.LM per il provider/modello scelto.
    Esempi: build_lm("anthropic", "claude-sonnet-4-5", api_key=...)
             build_lm("openai", "gpt-4o", api_key=...)
    """
    return dspy.LM(f"{provider}/{model}", api_key=api_key, **kwargs)


def build_context(business_description: str, kb_documents: list[dict]) -> str:
    """Concatena descrizione business + documenti KB in un unico blocco di testo.
    Semplice per iniziare; quando la KB cresce molto, sostituire con retrieval
    (embeddings + pgvector) invece di concatenare tutto (vedi ISTRUZIONI.md)."""
    parts = [business_description or ""]
    for doc in kb_documents:
        title = doc.get("title") or "Documento"
        parts.append(f"\n\n## {title}\n{doc.get('content', '')}")
    return "\n".join(parts).strip()


def run_draft(lm: dspy.LM, business_context: str, sample_question: str | None = None) -> dict:
    """Modalità cold-start (cliente nuovo, nessun esempio): il modulo gira
    zero-shot con la sola descrizione/KB come contesto. Nessuna vera
    ottimizzazione: è un punto di partenza ragionevole da rifinire a mano
    o, in futuro, con esempi reali (modalità 'optimize')."""
    agent = SupportAgent()

    # dspy.context (a differenza di dspy.configure) puo' essere usato da
    # qualsiasi thread -- necessario perche' Streamlit esegue ogni rerun
    # (ogni click su un bottone) in un thread nuovo, e dspy.configure()
    # accetta modifiche solo dal thread che lo ha chiamato la prima volta.
    with dspy.context(lm=lm):
        sample_answer = None
        if sample_question:
            pred = agent(business_context=business_context, question=sample_question)
            sample_answer = pred.answer

    return {
        "agent": agent,
        "preview": _extract_prompt_preview(agent),
        "sample_answer": sample_answer,
    }


def run_draft_with_tools(lm: dspy.LM, client_id: str, sample_question: str | None = None) -> dict:
    """Equivalente di run_draft ma con retrieval a tool invece che contesto
    concatenato: niente business_context da costruire, i tool leggono la KB
    del cliente al bisogno."""
    agent = SupportAgentWithTools(build_kb_tools(client_id))

    with dspy.context(lm=lm):
        sample_answer = None
        if sample_question:
            pred = agent(question=sample_question)
            sample_answer = pred.answer

    return {
        "agent": agent,
        "preview": _extract_prompt_preview(agent),
        "sample_answer": sample_answer,
    }


def answer_match_metric(example, pred, trace=None) -> bool:
    """Metrica di default per la compilazione: usa l'LM configurato come
    giudice per stabilire se la risposta prodotta è equivalente a quella
    attesa. Genera una chiamata LLM aggiuntiva per ogni esempio valutato:
    su dataset grandi valuta se sostituirla con una metrica più economica
    (es. similarity testuale) — vedi ISTRUZIONI.md."""
    judge = dspy.Predict("expected, actual -> equivalent: bool")
    result = judge(expected=example.answer, actual=pred.answer)
    return bool(result.equivalent)


def run_optimize(
    lm: dspy.LM,
    trainset: list[dict],
    max_bootstrapped_demos: int = 4,
    metric=None,
) -> dict:
    """Modalità optimize (cliente con esempi reali): usa gli esempi come
    training set per compilare un modulo migliorato con BootstrapFewShot.
    Ogni elemento di trainset è un dict con chiavi: context, question, expected_answer.
    """
    examples = [
        dspy.Example(
            business_context=ex["context"],
            question=ex["question"],
            answer=ex["expected_answer"],
        ).with_inputs("business_context", "question")
        for ex in trainset
    ]

    teleprompter = BootstrapFewShot(
        metric=metric or answer_match_metric,
        max_bootstrapped_demos=max_bootstrapped_demos,
    )
    # vedi il commento in run_draft: dspy.context() invece di dspy.configure()
    # perché BootstrapFewShot può valutare gli esempi anche da thread interni.
    with dspy.context(lm=lm):
        compiled_agent = teleprompter.compile(SupportAgent(), trainset=examples)

    return {
        "agent": compiled_agent,
        "preview": _extract_prompt_preview(compiled_agent),
    }


def run_optimize_with_tools(
    lm: dspy.LM,
    client_id: str,
    trainset: list[dict],
    metric=None,
) -> dict:
    """Equivalente di run_optimize per la modalità a tool. Ogni elemento di
    trainset è un dict con chiavi: question, expected_answer. Il campo
    "context" per-esempio (usato in modalità full_context per fare override
    del contesto) qui non si applica: il contesto lo recupera l'agente stesso
    tramite i tool, sempre dal DB corrente, non da un override statico."""
    examples = [
        dspy.Example(
            question=ex["question"],
            answer=ex["expected_answer"],
        ).with_inputs("question")
        for ex in trainset
    ]

    teleprompter = BootstrapFewShot(metric=metric or answer_match_metric)
    with dspy.context(lm=lm):
        compiled_agent = teleprompter.compile(
            SupportAgentWithTools(build_kb_tools(client_id)), trainset=examples
        )

    return {
        "agent": compiled_agent,
        "preview": _extract_prompt_preview(compiled_agent),
    }


def _extract_prompt_preview(agent: dspy.Module) -> str:
    """Estrae una versione testuale leggibile delle istruzioni + demo
    compilati, utile per la revisione umana senza dover ricaricare DSPy."""
    state = agent.dump_state()
    lines = []
    for name, predictor_state in state.items():
        sig = predictor_state.get("signature", {})
        lines.append(f"### {name}")
        lines.append(sig.get("instructions", ""))
        demos = predictor_state.get("demos", [])
        if demos:
            lines.append(f"\n{len(demos)} esempi few-shot selezionati:")
            for i, demo in enumerate(demos, 1):
                lines.append(f"\n**Esempio {i}**")
                for k, v in demo.items():
                    lines.append(f"- {k}: {v}")
        lines.append("\n---\n")
    return "\n".join(lines)


def save_program_to_dict(agent: dspy.Module) -> dict:
    """Serializza lo stato del modulo (istruzioni + demo) in un dict
    JSON-friendly, da salvare nella colonna compiled_program (jsonb)."""
    return agent.dump_state()


def load_program_from_dict(state: dict) -> SupportAgent:
    agent = SupportAgent()
    agent.load_state(state)
    return agent


def load_program_from_dict_with_tools(state: dict, client_id: str) -> SupportAgentWithTools:
    """Attenzione: dump_state()/load_state() serializzano SOLO istruzioni e
    demo del predictor, non i tool (sono funzioni Python, non JSON-friendly).
    Per questo qui i tool vanno ricostruiti da zero con build_kb_tools(),
    per lo stesso client_id usato durante l'ottimizzazione, PRIMA di
    chiamare load_state — altrimenti l'agente ricaricato non avrebbe
    nessun tool con cui rispondere."""
    agent = SupportAgentWithTools(build_kb_tools(client_id))
    agent.load_state(state)
    return agent


def ask(agent: SupportAgent, lm: dspy.LM, business_context: str, question: str) -> str:
    with dspy.context(lm=lm):
        pred = agent(business_context=business_context, question=question)
    return pred.answer


def ask_with_tools(agent: SupportAgentWithTools, lm: dspy.LM, question: str) -> str:
    with dspy.context(lm=lm):
        pred = agent(question=question)
    return pred.answer
