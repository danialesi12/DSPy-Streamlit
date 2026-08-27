"""
Logica DSPy: definizione del modulo "risponditore", costruzione dell'LM
multi-provider, e le due modalità richieste (draft / optimize).

Convenzioni:
- `provider` segue la sintassi LiteLLM usata da dspy.LM: "anthropic", "openai", ecc.
- Il "programma compilato" viene serializzato con agent.dump_state() (dict JSON-friendly)
  e ricaricato con agent.load_state(state) — non serve toccare il filesystem.
"""

from __future__ import annotations

import dspy
from dspy.teleprompt import BootstrapFewShot


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
    dspy.configure(lm=lm)
    agent = SupportAgent()

    sample_answer = None
    if sample_question:
        pred = agent(business_context=business_context, question=sample_question)
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
    dspy.configure(lm=lm)

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
    compiled_agent = teleprompter.compile(SupportAgent(), trainset=examples)

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


def ask(agent: SupportAgent, lm: dspy.LM, business_context: str, question: str) -> str:
    dspy.configure(lm=lm)
    pred = agent(business_context=business_context, question=question)
    return pred.answer
