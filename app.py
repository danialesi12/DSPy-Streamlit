"""
DSPy Prompt Studio -- tool interno per generare/ottimizzare, per ciascun
cliente, un modulo DSPy che risponde a domande sulla sua attività usando
la sua knowledge base specifica.

Setup: vedi ISTRUZIONI.md (schema Supabase, secrets.toml, deploy su
Streamlit Community Cloud).
"""

import json
import os

import streamlit as st

# Comodità solo per lo sviluppo in locale: se esiste un file .env lo carica
# nelle variabili d'ambiente. Su Coolify/Render le variabili le imposti nella
# dashboard del servizio, quindi python-dotenv semplicemente non trova nulla
# da fare e non serve preoccuparsene in produzione.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import db
import dspy_pipeline as pipeline

st.set_page_config(page_title="DSPy Prompt Studio", page_icon="🧩", layout="wide")

# Modelli "suggeriti" in UI (scorciatoia per i più usati), tutti raggiunti
# tramite OpenRouter con un'unica chiave (OPENROUTER_API_KEY). Il formato è
# "openrouter/<provider>/<modello>", dove <provider>/<modello> è esattamente
# l'id modello che OpenRouter usa nel suo catalogo (vedi openrouter.ai/models).
# Non è più l'unico modo di scegliere un modello: nella tab "Descrizione & LLM"
# c'è anche un campo per digitare qualsiasi id OpenRouter direttamente dalla
# UI, senza toccare questo file. Questa lista resta utile solo come elenco di
# scorciatoie rapide per i modelli che usi più spesso.
PROVIDERS = {
    "openrouter": [
        "anthropic/claude-sonnet-4-5",
        "anthropic/claude-opus-4-1",
        "anthropic/claude-haiku-4-5",
        "openai/gpt-4o",
        "openai/gpt-4.1",
        "google/gemini-2.5-pro",
    ],
}


def get_api_key(provider: str) -> str:
    key_name = f"{provider.upper()}_API_KEY"
    value = os.environ.get(key_name)
    if not value:
        st.error(f"Manca la variabile d'ambiente `{key_name}`. Impostala nel servizio (vedi ISTRUZIONI.md).")
        st.stop()
    return value


# ---------------------------------------------------------------------------
# Sidebar: selezione/creazione cliente
# ---------------------------------------------------------------------------
st.sidebar.title("🧩 DSPy Prompt Studio")

clients_df = db.list_clients()
client_names = clients_df["name"].tolist() if not clients_df.empty else []
selected_name = st.sidebar.selectbox("Cliente", ["— seleziona —"] + client_names)

with st.sidebar.expander("➕ Nuovo cliente"):
    new_name = st.text_input("Nome cliente", key="new_client_name")
    new_desc = st.text_area("Descrizione attività (contesto business)", key="new_client_desc")
    if st.button("Crea cliente"):
        if new_name.strip():
            db.create_client(new_name.strip(), new_desc.strip())
            st.toast(f"Cliente '{new_name}' creato con successo.", icon="✅")
            st.rerun()
        else:
            st.warning("Inserisci un nome.")

if selected_name == "— seleziona —":
    st.info("Seleziona o crea un cliente dalla sidebar per iniziare.")
    st.stop()

client_row = clients_df[clients_df["name"] == selected_name].iloc[0]
client_id = client_row["id"]

st.title(f"Cliente: {selected_name}")

tab_client, tab_kb, tab_examples, tab_run, tab_history, tab_test = st.tabs(
    ["Descrizione & LLM", "Knowledge base", "Esempi", "Esegui", "Storico run", "Test & Export"]
)

# ---------------------------------------------------------------------------
# Tab: descrizione business + configurazione LLM
# ---------------------------------------------------------------------------
with tab_client:
    st.subheader("Descrizione business (usata nel cold-start)")
    desc = st.text_area(
        "Descrizione", value=client_row["business_description"] or "", height=150, key="desc_area"
    )
    if st.button("Salva descrizione"):
        db.update_client_description(client_id, desc)
        st.toast("Descrizione aggiornata con successo.", icon="✅")
        st.rerun()

    st.subheader("Configurazione LLM")
    llm_configs = db.list_llm_configs(client_id)
    if not llm_configs.empty:
        st.dataframe(llm_configs[["provider", "model", "is_default"]], use_container_width=True)
    else:
        st.caption("Nessuna configurazione ancora. Aggiungine una qui sotto.")

    # Tutto passa da OpenRouter con un'unica chiave, quindi il "provider" è
    # fisso. Il modello invece si può scegliere dalla lista di suggerimenti
    # oppure digitare direttamente -- così per provare un modello nuovo del
    # catalogo OpenRouter non serve più modificare app.py e rifare il deploy.
    provider = "openrouter"
    CUSTOM_OPTION = "✏️ Altro (inserisci id modello manualmente)"
    model_choice = st.selectbox("Modello", PROVIDERS[provider] + [CUSTOM_OPTION])
    if model_choice == CUSTOM_OPTION:
        model = st.text_input(
            "Id modello OpenRouter",
            placeholder="es. mistralai/mistral-large-2411",
            help=(
                "Copia l'id esatto del modello da openrouter.ai/models (formato "
                "<provider>/<modello>). Con la stessa OPENROUTER_API_KEY funziona "
                "subito, senza bisogno di modificare il codice o rifare il deploy."
            ),
        ).strip()
    else:
        model = model_choice

    is_default = st.checkbox("Imposta come default per questo cliente", value=True)
    if st.button("Aggiungi configurazione LLM"):
        if model:
            db.add_llm_config(client_id, provider, model, is_default)
            st.toast("Configurazione LLM salvata con successo.", icon="✅")
            st.rerun()
        else:
            st.warning("Inserisci un id modello valido.")

    st.markdown("---")
    with st.expander("⚠️ Zona pericolosa"):
        st.write(
            f"Elimina definitivamente il cliente **{selected_name}** insieme a tutta la sua "
            "knowledge base, gli esempi e lo storico delle run. L'operazione non è reversibile."
        )
        confirm_delete = st.checkbox(
            f"Confermo di voler eliminare '{selected_name}' e tutti i suoi dati",
            key="confirm_delete_client",
        )
        if st.button("🗑️ Elimina cliente", disabled=not confirm_delete):
            db.delete_client(client_id)
            st.toast(f"Cliente '{selected_name}' eliminato con successo.", icon="🗑️")
            st.rerun()

# ---------------------------------------------------------------------------
# Tab: knowledge base
# ---------------------------------------------------------------------------
with tab_kb:
    st.subheader("Documenti knowledge base")
    kb_docs = db.list_kb_documents(client_id)
    if not kb_docs.empty:
        for _, doc in kb_docs.iterrows():
            with st.expander(doc["title"] or "(senza titolo)"):
                st.write(doc["content"])
                if st.button("Elimina", key=f"del_kb_{doc['id']}"):
                    db.delete_kb_document(doc["id"])
                    st.toast("Documento eliminato con successo.", icon="🗑️")
                    st.rerun()
    else:
        st.caption("Nessun documento ancora.")

    st.markdown("---")
    st.subheader("Aggiungi documento")
    doc_title = st.text_input("Titolo", key="new_doc_title")
    doc_content = st.text_area(
        "Contenuto (FAQ, policy, info prodotto, ecc.)", height=200, key="new_doc_content"
    )
    if st.button("Aggiungi alla knowledge base"):
        if doc_content.strip():
            db.add_kb_document(client_id, doc_title.strip(), doc_content.strip())
            st.toast("Documento aggiunto con successo.", icon="✅")
            st.rerun()
        else:
            st.warning("Il contenuto non può essere vuoto.")

# ---------------------------------------------------------------------------
# Tab: esempi (domanda / risposta attesa)
# ---------------------------------------------------------------------------
with tab_examples:
    st.subheader("Esempi domanda/risposta (per l'ottimizzazione)")
    examples_df = db.list_examples(client_id)
    if not examples_df.empty:
        st.dataframe(examples_df[["question", "expected_answer", "split"]], use_container_width=True)
        options = ["—"] + examples_df["id"].tolist()
        del_id = st.selectbox(
            "Elimina esempio",
            options,
            format_func=lambda x: "—"
            if x == "—"
            else examples_df.loc[examples_df["id"] == x, "question"].values[0][:60],
        )
        if del_id != "—" and st.button("Elimina esempio selezionato"):
            db.delete_example(del_id)
            st.toast("Esempio eliminato con successo.", icon="🗑️")
            st.rerun()
    else:
        st.caption("Nessun esempio ancora — per i clienti nuovi va bene, userai la modalità 'draft'.")

    st.markdown("---")
    st.subheader("Aggiungi esempio")
    q = st.text_area("Domanda", key="new_ex_q")
    a = st.text_area("Risposta attesa", key="new_ex_a")
    ctx = st.text_area(
        "Contesto specifico (opzionale — altrimenti si usa la KB generale del cliente)",
        key="new_ex_ctx",
    )
    split = st.radio("Split", ["train", "val"], horizontal=True, key="new_ex_split")
    if st.button("Aggiungi esempio"):
        if q.strip() and a.strip():
            db.add_example(client_id, q.strip(), a.strip(), ctx.strip() or None, split)
            st.toast("Esempio aggiunto con successo.", icon="✅")
            st.rerun()
        else:
            st.warning("Domanda e risposta sono obbligatorie.")

# ---------------------------------------------------------------------------
# Tab: esegui (draft o optimize)
# ---------------------------------------------------------------------------
with tab_run:
    llm_configs = db.list_llm_configs(client_id)
    if llm_configs.empty:
        st.warning("Configura almeno un provider/modello LLM nella tab 'Descrizione & LLM' prima di eseguire.")
    else:
        default_rows = llm_configs[llm_configs["is_default"]]
        default_idx = default_rows.index[0] if not default_rows.empty else llm_configs.index[0]
        llm_choice = st.selectbox(
            "Configurazione LLM da usare",
            llm_configs.index.tolist(),
            index=llm_configs.index.tolist().index(default_idx),
            format_func=lambda i: f"{llm_configs.loc[i, 'provider']}/{llm_configs.loc[i, 'model']}",
        )
        llm_config_row = llm_configs.loc[llm_choice]

        examples_df = db.list_examples(client_id)
        train_examples = (
            examples_df[examples_df["split"] == "train"] if not examples_df.empty else examples_df
        )

        suggested_mode = "optimize" if len(train_examples) >= 3 else "draft"
        st.write(
            f"Modalità suggerita: **{suggested_mode}** "
            f"({len(train_examples)} esempi di training disponibili — ne servono almeno alcuni "
            "per un'ottimizzazione sensata)."
        )
        mode = st.radio(
            "Modalità",
            ["draft", "optimize"],
            index=["draft", "optimize"].index(suggested_mode),
            horizontal=True,
        )
        retrieval = st.radio(
            "Retrieval",
            ["full_context", "tools"],
            format_func=lambda r: "Contesto pieno (KB intera nel prompt)"
            if r == "full_context"
            else "Tool su KB (get_index_pages / get_page_by_id, come un agente RAG)",
            horizontal=True,
            help=(
                "'Contesto pieno' è il comportamento storico: semplice, ma con KB grandi "
                "pesa su token e latenza. 'Tool su KB' fa recuperare al modello solo le "
                "pagine pertinenti tramite ReAct, più vicino a un agente in produzione."
            ),
        )

        kb_docs = db.list_kb_documents(client_id)
        business_context = pipeline.build_context(
            client_row["business_description"] or "",
            kb_docs.to_dict("records") if not kb_docs.empty else [],
        )

        if st.button("🚀 Avvia run", type="primary"):
            api_key = get_api_key(llm_config_row["provider"])
            lm = pipeline.build_lm(llm_config_row["provider"], llm_config_row["model"], api_key)

            run_id = db.create_run(client_id, mode, llm_config_row["id"], retrieval=retrieval)
            try:
                with st.spinner(f"Eseguo run in modalità '{mode}'... può richiedere qualche minuto."):
                    if retrieval == "tools":
                        if mode == "draft":
                            result = pipeline.run_draft_with_tools(lm, client_id)
                        else:
                            if train_examples.empty:
                                raise ValueError("Nessun esempio di training disponibile per l'ottimizzazione.")
                            trainset = [
                                {
                                    "question": row["question"],
                                    "expected_answer": row["expected_answer"],
                                }
                                for _, row in train_examples.iterrows()
                            ]
                            result = pipeline.run_optimize_with_tools(lm, client_id, trainset)
                    elif mode == "draft":
                        result = pipeline.run_draft(lm, business_context)
                    else:
                        if train_examples.empty:
                            raise ValueError("Nessun esempio di training disponibile per l'ottimizzazione.")
                        trainset = [
                            {
                                # row["context"] e' NaN (float) quando la colonna e' NULL nel
                                # DB -- "NaN or business_context" NON fa fallback perche' NaN
                                # e' truthy in Python (a differenza di None/""), quindi va
                                # controllato esplicitamente: solo una stringa non vuota conta
                                # come "contesto specifico impostato".
                                "context": row["context"]
                                if isinstance(row["context"], str) and row["context"].strip()
                                else business_context,
                                "question": row["question"],
                                "expected_answer": row["expected_answer"],
                            }
                            for _, row in train_examples.iterrows()
                        ]
                        result = pipeline.run_optimize(lm, trainset)

                    program_state = pipeline.save_program_to_dict(result["agent"])
                    db.finish_run(
                        run_id,
                        status="success",
                        compiled_program=program_state,
                        compiled_prompt_preview=result["preview"],
                    )
                st.success("Run completata.")
                st.text_area("Anteprima prompt compilato", result["preview"], height=300)
            except Exception as e:  # noqa: BLE001 -- vogliamo salvare qualsiasi errore nella run
                db.finish_run(run_id, status="failed", error=str(e))
                st.error(f"Run fallita: {e}")

# ---------------------------------------------------------------------------
# Tab: storico run
# ---------------------------------------------------------------------------
with tab_history:
    runs_df = db.list_runs(client_id)
    if runs_df.empty:
        st.info("Nessuna run ancora eseguita per questo cliente.")
    else:
        st.dataframe(runs_df[["created_at", "mode", "retrieval", "status"]], use_container_width=True)
        run_ids = runs_df["id"].tolist()
        selected_run_id = st.selectbox(
            "Visualizza dettaglio run",
            run_ids,
            format_func=lambda i: (
                f"{runs_df.loc[runs_df['id'] == i, 'created_at'].values[0]} — "
                f"{runs_df.loc[runs_df['id'] == i, 'mode'].values[0]} — "
                f"{runs_df.loc[runs_df['id'] == i, 'status'].values[0]}"
            ),
        )
        run_row = runs_df[runs_df["id"] == selected_run_id].iloc[0]
        if run_row["status"] == "failed":
            st.error(run_row["error"])
        else:
            st.text_area("Prompt compilato", run_row["compiled_prompt_preview"] or "", height=300)

# ---------------------------------------------------------------------------
# Tab: test & export
# ---------------------------------------------------------------------------
with tab_test:
    runs_df = db.list_runs(client_id)
    success_runs = runs_df[runs_df["status"] == "success"] if not runs_df.empty else runs_df
    if success_runs.empty:
        st.info("Esegui almeno una run con successo (tab 'Esegui') prima di testare/esportare.")
    else:
        run_id_for_test = st.selectbox(
            "Run da usare",
            success_runs["id"].tolist(),
            format_func=lambda i: (
                f"{success_runs.loc[success_runs['id'] == i, 'created_at'].values[0]} — "
                f"{success_runs.loc[success_runs['id'] == i, 'mode'].values[0]}"
            ),
        )
        run_row = db.get_run(run_id_for_test)
        llm_configs_all = db.list_llm_configs(client_id)
        llm_config_row = llm_configs_all[llm_configs_all["id"] == run_row["llm_config_id"]].iloc[0]

        st.subheader("Prova il modulo compilato")
        test_question = st.text_input("Domanda di prova", key="test_question")
        if st.button("Chiedi") and test_question.strip():
            api_key = get_api_key(llm_config_row["provider"])
            lm = pipeline.build_lm(llm_config_row["provider"], llm_config_row["model"], api_key)
            with st.spinner("Genero risposta..."):
                if run_row["retrieval"] == "tools":
                    # I tool non sono nello stato salvato (dump_state salva solo
                    # istruzioni + demo): vanno ricostruiti per questo client_id
                    # prima di ricaricare lo stato, altrimenti l'agente non avrebbe
                    # nulla con cui rispondere.
                    agent = pipeline.load_program_from_dict_with_tools(
                        run_row["compiled_program"], client_id
                    )
                    answer = pipeline.ask_with_tools(agent, lm, test_question.strip())
                else:
                    agent = pipeline.load_program_from_dict(run_row["compiled_program"])
                    kb_docs = db.list_kb_documents(client_id)
                    business_context = pipeline.build_context(
                        client_row["business_description"] or "",
                        kb_docs.to_dict("records") if not kb_docs.empty else [],
                    )
                    answer = pipeline.ask(agent, lm, business_context, test_question.strip())
            st.write(answer)

        st.markdown("---")
        st.subheader("Esporta")
        st.download_button(
            "⬇️ Programma DSPy compilato (JSON)",
            data=json.dumps(run_row["compiled_program"], indent=2, ensure_ascii=False),
            file_name=f"{selected_name}_{run_id_for_test}.json",
            mime="application/json",
        )
        st.download_button(
            "⬇️ Prompt in versione testuale",
            data=run_row["compiled_prompt_preview"] or "",
            file_name=f"{selected_name}_{run_id_for_test}.txt",
            mime="text/plain",
        )
