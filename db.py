"""
Livello di accesso dati: CRUD su Postgres (schema `promptstudio`) tramite
la connessione SQL nativa di Streamlit (st.connection), configurata in
.streamlit/secrets.toml sotto [connections.promptstudio_db].
"""

import json
import uuid

import streamlit as st
from sqlalchemy import text


def get_conn():
    return st.connection("promptstudio_db", type="sql")


def _maybe_json(value):
    """jsonb a volte torna già come dict/list (psycopg2 lo deserializza da solo),
    a volte come stringa: gestiamo entrambi i casi senza far esplodere l'app."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    return json.loads(value)


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

def list_clients():
    conn = get_conn()
    return conn.query(
        "select id, name, business_description, created_at "
        "from promptstudio.clients order by created_at desc",
        ttl=0,
    )


def create_client(name: str, business_description: str) -> str:
    conn = get_conn()
    new_id = str(uuid.uuid4())
    with conn.session as s:
        s.execute(
            text(
                "insert into promptstudio.clients (id, name, business_description) "
                "values (:id, :name, :desc)"
            ),
            {"id": new_id, "name": name, "desc": business_description},
        )
        s.commit()
    return new_id


def update_client_description(client_id: str, business_description: str):
    conn = get_conn()
    with conn.session as s:
        s.execute(
            text(
                "update promptstudio.clients set business_description = :desc "
                "where id = :id"
            ),
            {"desc": business_description, "id": client_id},
        )
        s.commit()


def get_client(client_id: str):
    conn = get_conn()
    df = conn.query(
        "select * from promptstudio.clients where id = :id",
        params={"id": client_id},
        ttl=0,
    )
    return df.iloc[0] if not df.empty else None


# ---------------------------------------------------------------------------
# Knowledge base documents
# ---------------------------------------------------------------------------

def list_kb_documents(client_id: str):
    conn = get_conn()
    return conn.query(
        "select * from promptstudio.kb_documents where client_id = :cid "
        "order by created_at desc",
        params={"cid": client_id},
        ttl=0,
    )


def add_kb_document(client_id: str, title: str, content: str):
    conn = get_conn()
    with conn.session as s:
        s.execute(
            text(
                "insert into promptstudio.kb_documents (client_id, title, content) "
                "values (:cid, :title, :content)"
            ),
            {"cid": client_id, "title": title, "content": content},
        )
        s.commit()


def delete_kb_document(doc_id: str):
    conn = get_conn()
    with conn.session as s:
        s.execute(text("delete from promptstudio.kb_documents where id = :id"), {"id": doc_id})
        s.commit()


# ---------------------------------------------------------------------------
# Examples (question / expected_answer)
# ---------------------------------------------------------------------------

def list_examples(client_id: str):
    conn = get_conn()
    return conn.query(
        "select * from promptstudio.examples where client_id = :cid "
        "order by created_at desc",
        params={"cid": client_id},
        ttl=0,
    )


def add_example(client_id: str, question: str, expected_answer: str, context: str | None, split: str = "train"):
    conn = get_conn()
    with conn.session as s:
        s.execute(
            text(
                "insert into promptstudio.examples "
                "(client_id, question, expected_answer, context, split) "
                "values (:cid, :q, :a, :ctx, :split)"
            ),
            {"cid": client_id, "q": question, "a": expected_answer, "ctx": context, "split": split},
        )
        s.commit()


def delete_example(example_id: str):
    conn = get_conn()
    with conn.session as s:
        s.execute(text("delete from promptstudio.examples where id = :id"), {"id": example_id})
        s.commit()


# ---------------------------------------------------------------------------
# LLM configs
# ---------------------------------------------------------------------------

def list_llm_configs(client_id: str):
    conn = get_conn()
    return conn.query(
        "select * from promptstudio.llm_configs where client_id = :cid "
        "order by created_at desc",
        params={"cid": client_id},
        ttl=0,
    )


def add_llm_config(client_id: str, provider: str, model: str, is_default: bool = False):
    conn = get_conn()
    with conn.session as s:
        if is_default:
            s.execute(
                text("update promptstudio.llm_configs set is_default = false where client_id = :cid"),
                {"cid": client_id},
            )
        s.execute(
            text(
                "insert into promptstudio.llm_configs (client_id, provider, model, is_default) "
                "values (:cid, :provider, :model, :is_default)"
            ),
            {"cid": client_id, "provider": provider, "model": model, "is_default": is_default},
        )
        s.commit()


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def create_run(client_id: str, mode: str, llm_config_id: str) -> str:
    conn = get_conn()
    new_id = str(uuid.uuid4())
    with conn.session as s:
        s.execute(
            text(
                "insert into promptstudio.runs (id, client_id, mode, llm_config_id, status) "
                "values (:id, :cid, :mode, :llm, 'running')"
            ),
            {"id": new_id, "cid": client_id, "mode": mode, "llm": llm_config_id},
        )
        s.commit()
    return new_id


def finish_run(
    run_id: str,
    status: str,
    compiled_program: dict | None = None,
    compiled_prompt_preview: str | None = None,
    metrics: dict | None = None,
    error: str | None = None,
):
    conn = get_conn()
    with conn.session as s:
        s.execute(
            text(
                "update promptstudio.runs set "
                "status = :status, "
                "compiled_program = cast(:program as jsonb), "
                "compiled_prompt_preview = :preview, "
                "metrics = cast(:metrics as jsonb), "
                "error = :error, "
                "finished_at = now() "
                "where id = :id"
            ),
            {
                "status": status,
                "program": json.dumps(compiled_program) if compiled_program is not None else None,
                "preview": compiled_prompt_preview,
                "metrics": json.dumps(metrics) if metrics is not None else None,
                "error": error,
                "id": run_id,
            },
        )
        s.commit()


def list_runs(client_id: str):
    conn = get_conn()
    df = conn.query(
        "select id, client_id, mode, llm_config_id, status, metrics, "
        "compiled_prompt_preview, error, created_at, finished_at "
        "from promptstudio.runs where client_id = :cid order by created_at desc",
        params={"cid": client_id},
        ttl=0,
    )
    return df


def get_run(run_id: str):
    conn = get_conn()
    df = conn.query(
        "select * from promptstudio.runs where id = :id",
        params={"id": run_id},
        ttl=0,
    )
    if df.empty:
        return None
    row = df.iloc[0].copy()
    row["compiled_program"] = _maybe_json(row["compiled_program"])
    row["metrics"] = _maybe_json(row["metrics"])
    return row
