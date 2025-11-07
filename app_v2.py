
import os
import io
import re
import random
from typing import List, Dict, Tuple

import streamlit as st

# Lightweight NLP without big external models
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import PyPDF2
from datetime import datetime

# Ensure NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)  # newer NLTK tokenizers
nltk.download("stopwords", quiet=True)

STOPWORDS = set(stopwords.words("spanish")) | set(stopwords.words("english"))

# -------------------------
# PDF text extraction
# -------------------------
def extract_text_from_pdf(file, max_pages: int = 0) -> str:
    """
    Read text from a PDF-like file object. If max_pages > 0, read only up to that many pages.
    """
    reader = PyPDF2.PdfReader(file)
    total = len(reader.pages)
    limit = total if max_pages in (None, 0) else min(max_pages, total)
    pages = []
    for idx in range(limit):
        p = reader.pages[idx]
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages), total, limit

# -------------------------
# Sentence cleanup
# -------------------------
def clean_sentence(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    s = s.strip()
    return s

def valid_candidate_word(w: str) -> bool:
    if len(w) < 5:
        return False
    if not re.match(r"^[A-Za-zÁÉÍÓÚÜÑáéíóúüñ\-]+$", w):
        return False
    wl = w.lower()
    return wl not in STOPWORDS

# -------------------------
# Build corpus and retrieval
# -------------------------
def split_to_sentences(text: str) -> List[str]:
    # Tokenize Spanish/English-ish text into sentences
    sents = sent_tokenize(text)
    sents = [clean_sentence(s) for s in sents if len(clean_sentence(s)) > 30]
    return sents

def build_vectorizer(sentences: List[str]):
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1,2),
        lowercase=True,
        max_df=0.9,
        min_df=1
    )
    X = vectorizer.fit_transform(sentences)
    return vectorizer, X

def top_k_similar(query: str, sentences: List[str], vectorizer, X, k=3) -> List[Tuple[int,float]]:
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, X).flatten()
    idx = sims.argsort()[::-1][:k]
    return [(int(i), float(sims[i])) for i in idx]

# -------------------------
# Question generation (cloze)
# -------------------------
def make_cloze_questions(sentences: List[str], num_questions: int = 10, options: int = 4, seed: int = 42) -> List[Dict]:
    random.seed(seed)
    # Build a bag of candidate words from all sentences
    tokens_all = []
    for s in sentences:
        tokens_all += [w for w in word_tokenize(s) if valid_candidate_word(w)]
    # De-duplicate while preserving order
    vocab = []
    seen = set()
    for t in tokens_all:
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            vocab.append(t)
    # Fallback if very small vocab
    if len(vocab) < 8:
        vocab = [w for w in tokens_all[:50]]

    questions = []
    used_sent_idx = set()
    tries = 0
    while len(questions) < num_questions and tries < max(1, len(sentences)) * 3:
        tries += 1
        i = random.randrange(0, max(1, len(sentences)))
        if i in used_sent_idx or not sentences:
            continue
        s = sentences[i]
        words = [w for w in word_tokenize(s) if valid_candidate_word(w)]
        if len(words) < 1:
            continue
        # Pick an answer word (prefer long-ish uncommon)
        words_sorted = sorted(words, key=lambda w: (-len(w), random.random()))
        answer = words_sorted[0]
        # Build distractors from vocab with similar length and not equal
        distractor_pool = [w for w in vocab if w.lower() != answer.lower() and abs(len(w) - len(answer)) <= 2]
        random.shuffle(distractor_pool)
        distractors = []
        for w in distractor_pool:
            wl = w.lower()
            if wl != answer.lower() and w not in distractors:
                distractors.append(w)
            if len(distractors) >= options - 1:
                break
        if len(distractors) < options - 1:
            # broaden
            distractor_pool = [w for w in vocab if w.lower() != answer.lower()]
            random.shuffle(distractor_pool)
            for w in distractor_pool:
                if w not in distractors:
                    distractors.append(w)
                if len(distractors) >= options - 1:
                    break
        if len(distractors) < options - 1:
            continue

        # Create cloze: replace first occurrence of answer with ____ (case-insensitive)
        pattern = re.compile(re.escape(answer), re.IGNORECASE)
        cloze = pattern.sub("_____", s, count=1)
        opts = distractors + [answer]
        random.shuffle(opts)
        correct_idx = opts.index(answer)

        q = {
            "sentence_index": i,
            "context_sentence": s,
            "question": f"Completa el hueco: {cloze}",
            "options": opts,
            "correct_index": correct_idx,
            "answer": answer
        }
        questions.append(q)
        used_sent_idx.add(i)
    return questions

# -------------------------
# Optional: OpenAI integration (if key provided)
# -------------------------
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

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="PDF → Test con explicación", page_icon="📝", layout="centered")

st.title("📝 PDF → Test con explicación")
st.caption("Sube un PDF y genera preguntas tipo test. Al fallar, pulsa “Explícame mi error” para ver la justificación basada en el PDF.")

with st.expander("⚙️ Ajustes", expanded=True):
    num_q = st.slider("Número de preguntas", min_value=3, max_value=30, value=10, step=1)
    num_opts = st.slider("Opciones por pregunta", min_value=3, max_value=6, value=4, step=1)
    seed = st.number_input("Semilla (reproducibilidad)", min_value=0, value=42, step=1)
    max_pages = st.number_input("Número máximo de páginas del PDF a leer (0 = todas)", min_value=0, value=0, step=1)
    col_a, col_b = st.columns([1,1])
    with col_a:
        regen = st.button("🔄 Nuevo test aleatorio")
    with col_b:
        st.caption("Usa este botón para crear otro test distinto sin tocar la semilla.")

uploaded = st.file_uploader("Sube tu PDF", type=["pdf"])

if uploaded:
    with st.spinner("Extrayendo texto del PDF..."):
        uploaded.seek(0)
        raw_text, total_pages, used_pages = extract_text_from_pdf(uploaded, max_pages=max_pages)

    st.info(f"El PDF tiene **{total_pages}** páginas. Estoy usando **{used_pages}** página(s) según tu ajuste.")

    if not raw_text or len(raw_text.strip()) < 50:
        st.error("No he podido extraer suficiente texto del PDF. Comprueba que el archivo tiene texto seleccionable.")
        st.stop()

    sents = split_to_sentences(raw_text)
    if len(sents) < 5:
        st.warning("El documento tiene pocas frases detectadas. Aun así, intentaré generar preguntas.")
    vectorizer, X = build_vectorizer(sents)

    if "regen_counter" not in st.session_state:
        st.session_state.regen_counter = 0
    if regen:
        st.session_state.regen_counter += 1

    signature = (num_q, num_opts, seed, max_pages, st.session_state.regen_counter)
    if st.session_state.get("signature") != signature:
        effective_seed = seed + (st.session_state.regen_counter * 9973)
        st.session_state.questions = make_cloze_questions(sents, num_questions=num_q, options=num_opts, seed=int(effective_seed))
        st.session_state.selected = {}
        st.session_state.show_feedback = {}
        st.session_state.show_explain = {}
        st.session_state.signature = signature

    st.subheader("Preguntas")
    for qi, q in enumerate(st.session_state.questions):
        with st.container(border=True):
            st.markdown(f"**P{qi+1}. {q['question']}**")
            chosen = st.radio(
                "Elige una opción:",
                options=list(range(len(q["options"]))),
                format_func=lambda i: q["options"][i],
                index=st.session_state.selected.get(qi, 0),
                key=f"radio_{qi}"
            )
            col1, col2 = st.columns([1,1])
            with col1:
                if st.button("Comprobar", key=f"check_{qi}"):
                    st.session_state.selected[qi] = chosen
                    st.session_state.show_feedback[qi] = True
            with col2:
                if st.button("Explícame mi error", key=f"exp_{qi}"):
                    st.session_state.selected[qi] = chosen
                    st.session_state.show_feedback[qi] = True
                    st.session_state.show_explain[qi] = True

            if st.session_state.show_feedback.get(qi):
                correct = q["correct_index"]
                if chosen == correct:
                    st.success("✅ ¡Correcto!")
                else:
                    st.error(f"❌ Incorrecto. La opción correcta es: **{q['options'][correct]}**")

                if st.session_state.show_explain.get(qi):
                    query = q["context_sentence"]
                    top_idxs = top_k_similar(query, sents, vectorizer, X, k=3)
                    evidence = [sents[i] for i,_ in top_idxs]
                    st.markdown("**Explicación (basada en el PDF):**")
                    llm = maybe_llm_explanation(query, "\n".join(evidence))
                    if llm:
                        st.write(llm)
                    else:
                        st.write(
                            "Según el documento, el fragmento más relacionado dice:\n\n" +
                            "\n\n".join([f"• {e}" for e in evidence])
                        )
                        st.caption("(*) Generado sin IA: se muestran las frases del PDF más relacionadas con la pregunta.")
else:
    st.info("Sube un PDF para empezar.")
