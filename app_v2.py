import os
import io
import re
import time
import json
import random
from datetime import datetime
from typing import List, Dict, Tuple

import streamlit as st

# ───────────────────────────────────────────────
# NLP y utilidades
# ───────────────────────────────────────────────
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Parsing PDF: primero PyMuPDF (más preciso). Si falla, PyPDF2.
import fitz  # PyMuPDF
import PyPDF2

# Descargas silenciosas para NLTK
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)  # compat tokenizer
nltk.download("stopwords", quiet=True)

STOPWORDS = set(stopwords.words("spanish")) | set(stopwords.words("english"))

# ───────────────────────────────────────────────
# Extracción de texto del PDF (optimizada)
# ───────────────────────────────────────────────
def extract_text_pymupdf(file_obj, max_pages: int = 0):
    """Extrae texto con PyMuPDF y detecta posibles títulos por tamaño de fuente."""
    try:
        data = file_obj.read()
        doc = fitz.open(stream=data, filetype="pdf")
        total = doc.page_count
        limit = total if max_pages in (None, 0) else min(max_pages, total)

        all_lines, headings, sizes = [], [], []
        # 1ª pasada: recoger tamaños de fuente
        for i in range(limit):
            pg = doc.load_page(i)
            for b in pg.get_text("dict")["blocks"]:
                if "lines" not in b:
                    continue
                for ln in b["lines"]:
                    for sp in ln["spans"]:
                        sizes.append(sp["size"])
        size_thresh = None
        if sizes:
            sizes_sorted = sorted(sizes)
            median = sizes_sorted[len(sizes_sorted)//2]
            size_thresh = median * 1.15
        # 2ª pasada: texto y títulos
        for i in range(limit):
            pg = doc.load_page(i)
            for b in pg.get_text("dict")["blocks"]:
                if "lines" not in b:
                    continue
                for ln in b["lines"]:
                    text_line = "".join(sp["text"] for sp in ln["spans"]).strip()
                    if not text_line:
                        continue
                    all_lines.append(text_line)
                    if size_thresh:
                        for sp in ln["spans"]:
                            if sp["size"] >= size_thresh and len(text_line) >= 6:
                                headings.append(text_line)
                                break
        return "\n".join(all_lines), total, limit, headings
    except Exception:
        return None, 0, 0, []


def extract_text_pypdf2(file_obj, max_pages: int = 0):
    file_obj.seek(0)
    reader = PyPDF2.PdfReader(file_obj)
    total = len(reader.pages)
    limit = total if max_pages in (None, 0) else min(max_pages, total)
    pages = []
    for idx in range(limit):
        try:
            pages.append(reader.pages[idx].extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages), total, limit, []


def extract_text_from_pdf(uploaded, max_pages: int = 0):
    """Usa PyMuPDF y cae en PyPDF2 si hace falta."""
    uploaded.seek(0)
    text, total, used, headings = extract_text_pymupdf(uploaded, max_pages=max_pages)
    if not text:
        uploaded.seek(0)
        text, total, used, headings = extract_text_pypdf2(uploaded, max_pages=max_pages)
    return text, total, used, headings


# ───────────────────────────────────────────────
# Helpers de corpus y recuperación
# ───────────────────────────────────────────────
def clean_sentence(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s


def valid_candidate_word(w: str) -> bool:
    if len(w) < 5:
        return False
    if not re.match(r"^[A-Za-zÁÉÍÓÚÜÑáéíóúüñ\-]+$", w):
        return False
    return w.lower() not in STOPWORDS


def split_to_sentences(text: str) -> List[str]:
    sents = [clean_sentence(s) for s in sent_tokenize(text)]
    # nos quedamos con oraciones con algo de contenido
    return [s for s in sents if len(s) > 30]


def build_vectorizer(sentences: List[str]):
    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True,
                                 max_df=0.9, min_df=1)
    X = vectorizer.fit_transform(sentences) if sentences else None
    return vectorizer, X


def top_k_similar(query: str, sentences: List[str], vectorizer, X, k=3):
    if X is None or not sentences:
        return []
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, X).flatten()
    idx = sims.argsort()[::-1][:k]
    return [(int(i), float(sims[i])) for i in idx]


# ───────────────────────────────────────────────
# Generación de preguntas por nivel
# ───────────────────────────────────────────────

def _harder_distractors(answer: str, sent_idx: int, sentences: List[str], vectorizer, X, how_many: int) -> List[str]:
    """Genera distractores más "confundibles":
    - busca oraciones parecidas a la de la respuesta
    - extrae candidatos con longitud similar y no stopwords
    """
    pool = []
    # oraciones similares a la de la respuesta
    if sentences:
        base = sentences[sent_idx]
        similar_idxs = [i for i, _ in top_k_similar(base, sentences, vectorizer, X, k=min(6, len(sentences)))]
        for i in similar_idxs:
            for w in word_tokenize(sentences[i]):
                if valid_candidate_word(w) and abs(len(w) - len(answer)) <= 2:
                    pool.append(w)
    # fallback general
    if not pool:
        for s in sentences:
            for w in word_tokenize(s):
                if valid_candidate_word(w) and abs(len(w) - len(answer)) <= 2:
                    pool.append(w)
    # limpieza, sin duplicados y sin la respuesta
    seen, clean = set(), []
    for w in pool:
        wl = w.lower()
        if wl == answer.lower():
            continue
        if wl in seen:
            continue
        seen.add(wl)
        clean.append(w)
        if len(clean) >= how_many:
            break
    return clean


def make_questions(sentences: List[str], *, level: str, num_questions: int, options: int, seed: int,
                   vectorizer=None, X=None) -> List[Dict]:
    random.seed(seed)
    qs = []
    used = set()

    while len(qs) < num_questions and len(used) < max(1, len(sentences)):
        i = random.randrange(0, max(1, len(sentences)))
        if i in used or not sentences:
            continue
        s = sentences[i]

        # Candidatos a respuesta
        words = [w for w in word_tokenize(s) if valid_candidate_word(w)]
        if not words:
            used.add(i)
            continue

        # Selección de respuesta usando palabra menos frecuente en el corpus
        # Calculamos frecuencia global de palabras
        if "word_freq" not in st.session_state:
            freq = {}
            for sen in sentences:
                for w in word_tokenize(sen):
                    wl = w.lower()
                    if valid_candidate_word(wl):
                        freq[wl] = freq.get(wl, 0) + 1
            st.session_state["word_freq"] = freq
        freq = st.session_state["word_freq"]

        # palabra candidata menos frecuente
        words_sorted = sorted(words, key=lambda w: (freq.get(w.lower(), 1_000_000), random.random()))
        answer = words_sorted[0]

        if level == "Básico":
            # Cloze sencillo + distractores globales por longitud
            pool = [w for s2 in sentences for w in word_tokenize(s2) if valid_candidate_word(w) and abs(len(w) - len(answer)) <= 2 and w.lower() != answer.lower()]
            random.shuffle(pool)
            distractors = []
            for w in pool:
                if w not in distractors:
                    distractors.append(w)
                if len(distractors) >= options - 1:
                    break

        elif level == "Intermedio":
            # Cloze con distractores de oraciones similares (más confusión)
            distractors = _harder_distractors(answer, i, sentences, vectorizer, X, options - 1)
            if len(distractors) < options - 1:
                # relleno si faltan
                pool = [w for s2 in sentences for w in word_tokenize(s2) if valid_candidate_word(w) and w.lower() != answer.lower()]
                random.shuffle(pool)
                for w in pool:
                    if w not in distractors:
                        distractors.append(w)
                    if len(distractors) >= options - 1:
                        break

        else:  # "Avanzado"
            # Cloze + reordenación de opciones con distractores muy próximos semánticamente
            distractors = _harder_distractors(answer, i, sentences, vectorizer, X, options - 1)
            # adicional: si hay títulos detectados, intentamos elegir palabras frecuentes en el mismo bloque (aproximado)
            if len(distractors) < options - 1:
                pool = []
                left = max(0, i - 2)
                right = min(len(sentences), i + 3)
                for s2 in sentences[left:right]:
                    for w in word_tokenize(s2):
                        if valid_candidate_word(w) and abs(len(w) - len(answer)) <= 2 and w.lower() != answer.lower():
                            pool.append(w)
                random.shuffle(pool)
                for w in pool:
                    if w not in distractors:
                        distractors.append(w)
                    if len(distractors) >= options - 1:
                        break

        if len(distractors) < options - 1:
            used.add(i)
            continue

        cloze = re.compile(re.escape(answer), re.IGNORECASE).sub("_____", s, count=1)
        opts = distractors + [answer]
        random.shuffle(opts)
        qs.append({
            "type": "cloze",
            "sentence_index": i,
            "context_sentence": s,
            "question": f"Completa el hueco: {cloze}",
            "options": opts,
            "correct_index": opts.index(answer),
            "answer": answer
        })
        used.add(i)

    return qs


# ───────────────────────────────────────────────
# Explicación con IA (opcional)
# ───────────────────────────────────────────────

def maybe_llm_explanation(query: str, context: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return ""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = (
            "Eres un tutor. Explica brevemente por qué la respuesta es incorrecta y cuál es la correcta, "
            "usando solo el siguiente contexto del PDF. Sé claro y conciso, en 3-5 frases.\n\n"
            f"Pregunta/Fragmento: {query}\n\n"
            f"Contexto del PDF:\n{context}\n"
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""


# ───────────────────────────────────────────────
# Guardado / carga de tests + historial
# ───────────────────────────────────────────────

def export_test_payload(meta: Dict, questions: List[Dict]) -> str:
    return json.dumps({"meta": meta, "questions": questions}, ensure_ascii=False, indent=2)


def import_test_payload(file) -> Tuple[Dict, List[Dict]]:
    data = json.load(file)
    return data.get("meta", {}), data.get("questions", [])


def add_result_to_history(score: int, total: int, mode: str):
    hist = st.session_state.setdefault("history", [])
    hist.append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "score": score,
        "total": total,
        "mode": mode
    })


# ───────────────────────────────────────────────
# UI principal
# ───────────────────────────────────────────────

st.set_page_config(page_title="PDF → Test con explicación", page_icon="📝", layout="centered")

# Branding
logo_path = "logo.png"
col1, col2 = st.columns([1, 6])
with col1:
    if os.path.exists(logo_path):
        st.image(logo_path, width=64)
with col2:
    st.title("📝 PDF → Test con explicación")

st.caption("Elige nivel (Básico/Intermedio/Avanzado), guarda tests, modo examen con tiempo y seguimiento de resultados. IA opcional para explicar errores (añade tu OPENAI_API_KEY).")

with st.expander("⚙️ Ajustes", expanded=True):
    level = st.selectbox("Nivel de dificultad", ["Básico", "Intermedio", "Avanzado"], index=1)
    num_q = st.slider("Número de preguntas", 3, 40, 10)
    num_opts = st.slider("Opciones por pregunta", 3, 6, 4)
    seed = st.number_input("Semilla (reproducibilidad)", min_value=0, value=42, step=1)
    max_pages = st.number_input("Número máximo de páginas del PDF a leer (0 = todas)", min_value=0, value=0, step=1)

    st.markdown("---")
    st.subheader("🧪 Modo examen")
    exam_mode = st.checkbox("Activar modo examen (con tiempo)")
    exam_minutes = st.number_input("Minutos de examen", 1, 180, 20)
    start_exam = st.button("🎬 Empezar examen / reiniciar")

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        regen = st.button("🔄 Nuevo test aleatorio")
    with c2:
        st.download_button(
            "💾 Descargar test (JSON)",
            data=export_test_payload(
                {"created": datetime.now().isoformat(timespec="seconds"),
                 "level": level, "num_q": num_q, "num_opts": num_opts, "seed": seed, "max_pages": int(max_pages)},
                st.session_state.get("questions", [])
            ).encode("utf-8"),
            file_name="test_guardado.json",
            mime="application/json"
        )
    with c3:
        uploaded_test = st.file_uploader("Cargar test (JSON)", type=["json"], key="upload_test_json")

uploaded_pdf = st.file_uploader("Sube tu PDF", type=["pdf"])

# Cargar un test guardado (sin PDF)
if uploaded_test is not None:
    try:
        meta, qs = import_test_payload(uploaded_test)
        st.session_state.questions = qs
        st.session_state.selected = {}
        st.session_state.show_feedback = {}
        st.session_state.show_explain = {}
        st.success("Test cargado desde JSON.")
    except Exception as e:
        st.error(f"No he podido cargar el test: {e}")

# Si hay PDF subido, (re)generar test cuando toque
if uploaded_pdf and ("questions" not in st.session_state or regen or start_exam):
    with st.spinner("Extrayendo texto del PDF..."):
        raw_text, total_pages, used_pages, headings = extract_text_from_pdf(uploaded_pdf, max_pages=int(max_pages))

    st.info(f"El PDF tiene **{total_pages}** páginas. Estoy usando **{used_pages}**. Detecté **{len(headings)}** posibles títulos.")

    if not raw_text or len(raw_text.strip()) < 50:
        st.error("No he podido extraer suficiente texto del PDF. Comprueba que el archivo tiene texto seleccionable.")
        st.stop()

    sents = split_to_sentences(raw_text)
    vectorizer, X = build_vectorizer(sents)

    if "regen_counter" not in st.session_state:
        st.session_state.regen_counter = 0
    if regen:
        st.session_state.regen_counter += 1

    if start_exam and exam_mode:
        st.session_state.exam_started_at = time.time()
        st.session_state.exam_ends_at = time.time() + (int(exam_minutes) * 60)
        st.session_state.selected = {}
        st.session_state.show_feedback = {}
        st.session_state.show_explain = {}

    effective_seed = seed + (st.session_state.regen_counter * 9973)
    st.session_state.questions = make_questions(
        sents, level=level, num_questions=int(num_q), options=int(num_opts), seed=int(effective_seed),
        vectorizer=vectorizer, X=X
    )
    st.session_state.index_data = {"sentences": sents}
    st.session_state.vectorizer = vectorizer
    st.session_state.X = X

# Temporizador del examen
if st.session_state.get("exam_ends_at") and exam_mode:
    remaining = max(0, int(st.session_state["exam_ends_at"] - time.time()))
    mins, secs = remaining // 60, remaining % 60
    st.warning(f"⏳ Tiempo restante: {mins:02d}:{secs:02d}")
    if remaining == 0:
        st.experimental_rerun()
    else:
        st.autorefresh(interval=1000, key="timer")

# Renderizado de preguntas
questions = st.session_state.get("questions", [])
if questions:
    sents = st.session_state.get("index_data", {}).get("sentences", [])
    vectorizer = st.session_state.get("vectorizer")
    X = st.session_state.get("X")
    total = len(questions)
    correct_count = 0

    st.subheader("Preguntas")
    for qi, q in enumerate(questions):
        with st.container(border=True):
            st.markdown(f"**P{qi+1}. {q['question']}**")
            chosen = st.radio(
                "Elige una opción:",
                options=list(range(len(q["options"]))),
                format_func=lambda i: q["options"][i],
                index=st.session_state.get("selected", {}).get(qi, 0),
                key=f"radio_{qi}"
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Comprobar", key=f"check_{qi}"):
                    st.session_state.setdefault("selected", {})[qi] = chosen
                    st.session_state.setdefault("show_feedback", {})[qi] = True
            with c2:
                if st.button("Explícame mi error", key=f"exp_{qi}"):
                    st.session_state.setdefault("selected", {})[qi] = chosen
                    st.session_state.setdefault("show_feedback", {})[qi] = True
                    st.session_state.setdefault("show_explain", {})[qi] = True

            if st.session_state.get("show_feedback", {}).get(qi):
                correct = q["correct_index"]
                if chosen == correct:
                    st.success("✅ ¡Correcto!")
                    correct_count += 1
                else:
                    st.error(f"❌ Incorrecto. La opción correcta es: **{q['options'][correct]}**")

                if st.session_state.get("show_explain", {}).get(qi):
                    query = q["context_sentence"]
                    top_idxs = top_k_similar(query, sents, vectorizer, X, k=3)
                    evidence = [sents[i] for i, _ in top_idxs] if top_idxs else [q["context_sentence"]]
                    st.markdown("**Explicación (basada en el PDF):**")
                    llm = maybe_llm_explanation(query, "\n".join(evidence))
                    if llm:
                        st.write(llm)
                    else:
                        st.write("Según el documento, el fragmento más relacionado dice:\n\n" + "\n\n".join([f"• {e}" for e in evidence]))
                        st.caption("(*) Generado sin IA: se muestran frases del PDF relacionadas con la pregunta.")

    with st.container(border=True):
        st.subheader("Resultados")
        answered = len([1 for qi in range(total) if st.session_state.get("show_feedback", {}).get(qi)])
        st.write(f"Preguntas respondidas: **{answered}/{total}**")
        st.write(f"Aciertos (hasta ahora): **{correct_count}**")
        if st.button("Guardar resultado en historial"):
            add_result_to_history(correct_count, total, "examen" if exam_mode else "práctica")
            st.success("Resultado guardado. Mira el historial debajo.")

    hist = st.session_state.get("history", [])
    if hist:
        st.subheader("Historial")
        import pandas as pd
        df = pd.DataFrame(hist)
        st.dataframe(df, use_container_width=True)
        st.download_button("⬇️ Descargar historial (CSV)", data=df.to_csv(index=False).encode("utf-8"),
                           file_name="historial_resultados.csv", mime="text/csv")
else:
    st.info("Sube un PDF o carga un test (JSON) para empezar.")
