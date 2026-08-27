"""
Livello di accesso dati: CRUD su Postgres (schema `promptstudio`) tramite
SQLAlchemy "puro" -- niente st.connection/st.secrets, cosi' l'app gira
identica su Coolify, Render, o qualsiasi altro host Docker: basta passare
la variabile d'ambiente DATABASE_URL.
"""

import json
import os
import uuid

import pandas as pd
from sqlalchemy import create_engine, text

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        database_url = os.environ.get("DATABASE_URL")
        if not database_url:
            raise RuntimeError(
                "Manca la variabile d'ambiente DATABASE_URL "
                "(es. postgresql+psycopg2://user:password@host:5432/postgres)."
            )
        _engine = create_engine(database_url, pool_pre_ping=True)
    return _engine


def _query(sql: str, params: dict | None = None) -> pd.DataFrame:
    with get_engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def _execute(sql: str, params: dict | None = None):
    with get_engine().begin() as conn:  # begin() fa commit automatico se non ci sono errori
        conn.execute(text(sql), params or {})


def _maybe_json(value):
    """jsonb a volte torna gia' come dict/list, a volte come stringa:
    gestiamo entrambi i casi senza far esplodere l'app."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    return json.loads(value)


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

def list_clients() -> pd.DataFrame:
    return _query(
        "select id, name, business_description, created_at "
        "from promptstudio.clients order by created_at desc"
    )


def create_client(name: str, business_description: str) -> str:
    new_id = str(uuid.uuid4())
    _execute(
        "insert into promptstudio.clients (id, name, business_description) "
        "values (:id, :name, :desc)",
        {"id": new_id, "name": name, "desc": business_description},
    )
    return new_id


def update_client_description(client_id: str, business_description: str):
    _execute(
        "update promptstudio.clients set business_description = :desc where id = :id",
        {"desc": business_description, "id": client_id},
    )


def get_client(client_id: str):
    df = _query("select * from promptstudio.clients where id = :id", {"id": client_id})
    return df.iloc[0] if not df.empty else None


# ---------------------------------------------------------------------------
# Knowledge base documents
# ---------------------------------------------------------------------------

def list_kb_documents(client_id: str) -> pd.DataFrame:
    return _query(
        "select * from promptstudio.kb_documents where client_id = :cid order by created_at desc",
        {"cid": client_id},
    )


def add_kb_document(client_id: str, title: str, content: str):
    _execute(
        "insert into promptstudio.kb_documents (client_id, title, content) "
        "values (:cid, :title, :content)",
        {"cid": client_id, "title": title, "content": content},
    )


def delete_kb_document(doc_id: str):
    _execute("delete from promptstudio.kb_documents where id = :id", {"id": doc_id})


# ---------------------------------------------------------------------------
# Examples (question / expected_answer)
# ---------------------------------------------------------------------------

def list_examples(client_id: str) -> pd.DataFrame:
    return _query(
        "select * from promptstudio.examples where client_id = :cid order by created_at desc",
        {"cid": client_id},
    )


def add_example(client_id: str, question: str, expected_answer: str, context: str | None, split: str = "train"):
    _execute(
        "insert into promptstudio.examples "
        "(client_id, question, expected_answer, context, split) "
        "values (:cid, :q, :a, :ctx, :split)",
        {"cid": client_id, "q": question, "a": expected_answer, "ctx": context, "split": split},
    )


def delete_example(example_id: str):
    _execute("delete from promptstudio.examples where id = :id", {"id": example_id})


# ---------------------------------------------------------------------------
# LLM configs
# ---------------------------------------------------------------------------

def list_llm_configs(client_id: str) -> pd.DataFrame:
    return _query(
        "select * from promptstudio.llm_configs where client_id = :cid order by created_at desc",
        {"cid": client_id},
    )


def add_llm_config(client_id: str, provider: str, model: str, is_default: bool = False):
    if is_default:
        _execute(
            "update promptstudio.llm_configs set is_default = false where client_id = :cid",
            {"cid": client_id},
        )
    _execute(
        "insert into promptstudio.llm_configs (client_id, provider, model, is_default) "
        "values (:cid, :provider, :model, :is_default)",
        {"cid": client_id, "provider": provider, "model": model, "is_default": is_default},
    )


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def create_run(client_id: str, mode: str, llm_config_id: str) -> str:
    new_id = str(uuid.uuid4())
    _execute(
        "insert into promptstudio.runs (id, client_id, mode, llm_config_id, status) "
        "values (:id, :cid, :mode, :llm, 'running')",
        {"id": new_id, "cid": client_id, "mode": mode, "llm": llm_config_id},
    )
    return new_id


def finish_run(
    run_id: str,
    status: str,
    compiled_program: dict | None = None,
    compiled_prompt_preview: str | None = None,
    metrics: dict | None = None,
    error: str | None = None,
):
    _execute(
        "update promptstudio.runs set "
        "status = :status, "
        "compiled_program = cast(:program as jsonb), "
        "compiled_prompt_preview = :preview, "
        "metrics = cast(:metrics as jsonb), "
        "error = :error, "
        "finished_at = now() "
        "where id = :id",
        {
            "status": status,
            "program": json.dumps(compiled_program) if compiled_program is not None else None,
            "preview": compiled_prompt_preview,
            "metrics": json.dumps(metrics) if metrics is not None else None,
            "error": error,
            "id": run_id,
        },
    )


def list_runs(client_id: str) -> pd.DataFrame:
    return _query(
        "select id, client_id, mode, llm_config_id, status, metrics, "
        "compiled_prompt_preview, error, created_at, finished_at "
        "from promptstudio.runs where client_id = :cid order by created_at desc",
        {"cid": client_id},
    )


def get_run(run_id: str):
    df = _query("select * from promptstudio.runs where id = :id", {"id": run_id})
    if df.empty:
        return None
    row = df.iloc[0].copy()
    row["compiled_program"] = _maybe_json(row["compiled_program"])
    row["metrics"] = _maybe_json(row["metrics"])
    return row
