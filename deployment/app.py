import streamlit as st
import torch
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import pathlib
from model import TransformerEncoder
from tokenizer import IndicSentencePieceTokenizer
from dataset import get_indic_processor
from supabase import create_client
from datetime import datetime
import uuid
from keys import SUPABASE_URL, SUPABASE_KEY

# ==========================================================
# CONFIG
# ==========================================================
DEVICE = torch.device("cpu")
BASE_DIR = pathlib.Path(__file__).resolve().parent
TOKENIZER_FILE = str(BASE_DIR / "indic_tokenizer.model")
LABEL_MAP_FILE = str(BASE_DIR / "label2id.json")
CHECKPOINT_FILE = str(BASE_DIR / "shortest_model.pt")

MAX_LEN = 256
VOCAB_SIZE = 32000

# ==========================================================
# LOAD ALL (cached)
# ==========================================================
@st.cache_resource(show_spinner=False)
def load_all():
    # Load processor
    processor = get_indic_processor()

    # Load tokenizer
    tokenizer = IndicSentencePieceTokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.load(TOKENIZER_FILE)

    # Load label maps
    with open(LABEL_MAP_FILE, "r", encoding="utf-8") as f:
        label2id = json.load(f)
    id2label = {int(v): k for k, v in label2id.items()}
    num_langs = len(label2id)

    # Initialize model (same as training)
    model = TransformerEncoder(
        vocab_size=len(tokenizer),
        embed_dim=256,
        num_layers=2,
        num_heads=8,
        ff_dim=1024,
        phase="phase2",
    ).to(DEVICE)

    # Add classifier head
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.2),
        torch.nn.Linear(256, num_langs)
    ).to(DEVICE)

    # Load checkpoint
    checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()

    return processor, tokenizer, model, label2id, id2label, num_langs


processor, tokenizer, model, label2id, id2label, NUM_LANGS = load_all()


@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()



def store_feedback_supabase(text, predicted, correct_list, confidences_dict):
    session_id = st.session_state.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        st.session_state["session_id"] = session_id

    data = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "input_text": text,
        "predicted": predicted,
        "correct": ",".join(correct_list),
        "confidences": confidences_dict,
    }

    response = supabase.table("feedback").insert(data).execute()

    # # DEBUG PRINT
    # st.write("Supabase insert response:", response)




# ==========================================================
# INFERENCE FUNCTION
# ==========================================================
def predict_texts(texts, model, tokenizer, processor, id2label, max_len=256):
    model.eval()
    preds, probs_all = [], []

    with torch.no_grad():
        for text in texts:
            text_proc = processor.process(text)
            enc = tokenizer.batch_encode([text_proc], max_length=max_len)
            # st.caption(enc)
            ids = enc["input_ids"].to(DEVICE)
            masks = enc["attention_mask"].to(DEVICE)

            logits, _ = model(ids, masks)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            pred_id = int(np.argmax(probs))
            preds.append(id2label[pred_id])
            probs_all.append(probs)
    return preds, probs_all


# ==========================================================
# UI SECTIONS (Improved UX)
# ==========================================================

st.markdown("<h1 style='text-align:center;'>BhashaDetector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:16px;'>Identifies Indic Languages from text using a Transformer Encoder</p>", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.header("🔧 Options")
    mode = st.radio("Mode", ["Single Prediction", "Batch CSV"])

    st.subheader("📌 Model Info")
    st.success(f"Model loaded on: `{DEVICE}`")
    st.info(f"Languages Supported: **{NUM_LANGS}**")

# ------------------------
# SINGLE SENTENCE MODE (Improved - no nested feedback button)
# ------------------------
if mode == "Single Prediction":
    st.subheader("📝 Single Sentence Prediction")

    text_input = st.text_area(
        "Enter text Indic language (supports both native & roman script) :",
        placeholder="Type or paste your sentence here...",
        height=140,
        key="input_text"
    )

    # Predict button
    predict_btn = st.button("🔍 Predict Language", use_container_width=True)

    if predict_btn:
        if not text_input.strip():
            st.warning("⚠️ Please enter text before predicting.")
        else:
            with st.spinner("Analyzing language..."):
                preds, probs_all = predict_texts(
                    [text_input], model, tokenizer, processor, id2label, MAX_LEN
                )

            pred = preds[0]
            probs = probs_all[0]

            # Save prediction results in session_state
            st.session_state["last_pred_text"] = text_input
            st.session_state["last_pred_label"] = pred
            st.session_state["last_pred_probs"] = probs.tolist()

    # If a prediction exists, show results & feedback
    if "last_pred_label" in st.session_state:
        pred = st.session_state["last_pred_label"]
        probs = np.array(st.session_state["last_pred_probs"])
        text_input_saved = st.session_state["last_pred_text"]

        st.success(f"#### Predicted Language: **{pred}**")

        df_probs = pd.DataFrame({
            "Language": [id2label[i] for i in range(len(probs))],
            "Confidence": probs
        })

        st.write("#### Confidence Scores")
        st.bar_chart(df_probs.set_index("Language"))

        st.markdown("---")
        st.write("#### Feedback for wrong predictions")
        st.caption("Help improve the model by marking correct language(s).")

        # Display checkboxes without nested button
        correct_list = []
        cols = st.columns(5)
        for idx, row in df_probs.iterrows():
            lang = row["Language"]
            key = f"fb_{lang}"
            cb = cols[idx % 5].checkbox(lang, key=key)
            if cb:
                correct_list.append(lang)

        # custom_label = st.text_input("Or enter another language (optional)", key="custom_label")

        submitted = st.button("📤 Submit Feedback", type="primary", use_container_width=True)

        if submitted:
            selected = correct_list.copy()
            # if custom_label.strip():
            #     selected.append(custom_label.strip())

            if not selected:
                st.warning("⚠️ Please select or enter at least one correct label.")
            else:
                conf_dict = {id2label[i]: float(probs[i]) for i in range(len(probs))}

                store_feedback_supabase(
                    text=text_input_saved,
                    predicted=pred,
                    correct_list=selected,
                    confidences_dict=conf_dict,
                )

                st.success("🙏 Thank you! Your feedback has been recorded.")

                # Clear feedback and prediction after submit
                for key in list(st.session_state.keys()):
                    if key.startswith("fb_") or key in ["last_pred_probs", "last_pred_label", "last_pred_text"]:
                        del st.session_state[key]



# ------------------------
# BATCH CSV MODE
# ------------------------
elif mode == "Batch CSV":
    st.subheader("📁 Batch CSV Prediction")
    uploaded = st.file_uploader("Upload CSV with a `text` column", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        if "text" not in df.columns:
            st.error("❌ CSV must contain a `text` column.")
        else:
            st.write("### 🔍 Preview")
            st.dataframe(df.head())

            if st.button("🚀 Run Batch Predictions", use_container_width=True):
                total_rows = len(df)
                st.info(f"Processing {total_rows:,} samples...")
                progress_bar = st.progress(0)
                status_text = st.empty()

                preds = []
                batch_size = 512  # adjust as needed
                start_time = time.time()

                with st.spinner("Running inference..."):
                    for i in range(0, total_rows, batch_size):
                        batch_start = time.time()
                        batch_texts = df["text"].iloc[i:i + batch_size].astype(str).tolist()
                        batch_preds, _ = predict_texts(batch_texts, model, tokenizer, processor, id2label, MAX_LEN)
                        preds.extend(batch_preds)

                        # Progress computation
                        progress = min((i + batch_size) / total_rows, 1.0)
                        elapsed = time.time() - start_time
                        rate = (i + batch_size) / elapsed if elapsed > 0 else 0
                        remaining = (total_rows - (i + batch_size)) / rate if rate > 0 else 0

                        # Update Streamlit UI
                        progress_bar.progress(int(progress * 100))
                        status_text.text(
                            f"✅ Processed {min(i + batch_size, total_rows)}/{total_rows} "
                            f"({progress * 100:.1f}%) | "
                            f"⏱️ Elapsed: {elapsed:.1f}s | "
                            f"⌛ ETA: {remaining:.1f}s"
                        )

                df["predicted"] = preds
                total_time = time.time() - start_time
                progress_bar.progress(100)
                status_text.text(f"✅ Prediction complete! Total time: {total_time:.1f}s")

                st.dataframe(df.head())

                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Predictions", csv_data, "predictions.csv", "text/csv")

                # If ground-truth labels are available, show metrics
                if "label" in df.columns:
                    y_true = df["label"].astype(str)
                    y_pred = df["predicted"].astype(str)
                    acc = accuracy_score(y_true, y_pred)
                    st.metric("Accuracy", f"{acc * 100:.2f}%")

                    report = classification_report(y_true, y_pred, output_dict=True)
                    st.write("### Classification Report")
                    st.dataframe(pd.DataFrame(report).transpose())

                    # Confusion matrix
                    cm = confusion_matrix(y_true, y_pred, labels=sorted(id2label.values()))
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=False, cmap="Blues",
                                xticklabels=sorted(id2label.values()),
                                yticklabels=sorted(id2label.values()))
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("True")
                    st.pyplot(fig)
