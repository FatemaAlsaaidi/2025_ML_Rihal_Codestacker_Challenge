# streamlit_app_level4.py
# -----------------------
import io
import numpy as np
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path
from typing import List, Dict, Any
import tempfile

from level4_extraction import extract_text_from_pdf, ocr_pdf_first_n_pages, parse_fields_from_text, features_from_rows
from level4_severity import assign_severity

st.set_page_config(page_title="CityX - Level 4: PDF Extraction & Inference", page_icon="📄", layout="wide")
st.title("Police Report Extraction → Classification → Severity")

# --------- Static model path ---------
DEFAULT_MODEL_PATH = Path("../models/crime_category_text_model.pkl")

# Flexible checker: accepts a Pipeline or a dict containing a pipeline or (vectorizer + classifier)
def _normalize_loaded_object(obj):
    # If it's a pipeline-like with predict
    if hasattr(obj, "predict"):
        return obj
    # If it's a dictionary
    if isinstance(obj, dict):
        for key in ("pipeline", "model", "estimator", "clf"):
            if key in obj and hasattr(obj[key], "predict"):
                return obj[key]
        if "vectorizer" in obj and "classifier" in obj and hasattr(obj["classifier"], "predict"):
            class _VecClfWrapper:
                def __init__(self, vectorizer, classifier):
                    self.vectorizer = vectorizer
                    self.classifier = classifier
                def predict(self, texts):
                    X = self.vectorizer.transform(texts)
                    return self.classifier.predict(X)
                def predict_proba(self, texts):
                    if hasattr(self.classifier, "predict_proba"):
                        X = self.vectorizer.transform(texts)
                        return self.classifier.predict_proba(X)
                    raise AttributeError("Classifier has no predict_proba")
                def decision_function(self, texts):
                    if hasattr(self.classifier, "decision_function"):
                        X = self.vectorizer.transform(texts)
                        return self.classifier.decision_function(X)
                    raise AttributeError("Classifier has no decision_function")
            return _VecClfWrapper(obj["vectorizer"], obj["classifier"])
    raise ValueError("Unsupported model format. Expect a Pipeline or dict with 'pipeline' or ('vectorizer' + 'classifier').")

@st.cache_resource
def load_static_model(path: Path):
    if not path.exists():
        return None, f"Model not found at: {path}"
    try:
        obj = joblib.load(path)
        model = _normalize_loaded_object(obj)
        return model, None
    except Exception as e:
        return None, f"Failed to load model: {e}"

with st.sidebar:
    st.header("⚙️ Settings")
    # Allow user to see/override the static model path (optional)
    model_path_str = st.text_input("Static model path", value=str(DEFAULT_MODEL_PATH))
    #use_ocr = st.toggle("Try OCR fallback for scanned PDFs (slower)", value=False)
    st.divider()
    st.markdown("**Batch PDF upload**")
    pdf_files = st.file_uploader("Police report PDFs", type=["pdf"], accept_multiple_files=True, key="pdf_upl")

# --- Load static model once ---
model_status = st.empty()
clf, load_err = load_static_model(Path(model_path_str))
#if clf is not None:
    #model_status.success(f" Static model loaded from: {model_path_str}")
#else:
    #model_status.error(load_err or "Could not load the static model.")
    #st.stop()  # No model → stop the app early

# --- Extraction ---
st.subheader("1) Extract key fields from uploaded PDFs")
rows: List[Dict[str, Any]] = []

if pdf_files:
    progress = st.progress(0)
    tmp_dir = Path(tempfile.gettempdir()) / "cityx_uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for i, upl in enumerate(pdf_files):
        progress.progress((i+1)/len(pdf_files))

        tmp_path = tmp_dir / upl.name
        # Streamlit versions differ: sometimes .getbuffer(), other times .read()
        try:
            data = upl.getbuffer()
        except Exception:
            data = upl.read()
        with open(tmp_path, "wb") as f:
            f.write(data)

        text = extract_text_from_pdf(str(tmp_path))
        if not text and use_ocr:
            text = ocr_pdf_first_n_pages(str(tmp_path), n_pages=2, dpi=200)
        if not text:
            st.warning(f"Could not extract text from {upl.name}. It might be a scanned PDF and OCR is disabled or unavailable.")
            continue

        parsed = parse_fields_from_text(text)
        parsed["__source_pdf__"] = upl.name
        rows.append(parsed)

    progress.progress(1.0)

if not rows:
    st.info("Upload one or more PDF files to extract. Parsed results will appear here.")
else:
    df_extracted = features_from_rows(rows)
    st.write("**Extracted (editable) table** – adjust any fields if needed before inference:")
    edited = st.data_editor(df_extracted, num_rows="dynamic", use_container_width=True, key="editor_level4")

    st.subheader("2) Run Level-2 Classifier on Descriptions → Predict Category & Severity")
    can_infer = "Descript" in edited.columns
    infer_btn = st.button("Predict", type="primary", disabled=not can_infer)

    if infer_btn and can_infer:
        X_text = edited["Descript"].fillna("").astype(str).tolist()
        y_pred = None
        conf = None
        try:
            y_pred = clf.predict(X_text)
            if hasattr(clf, "decision_function"):
                arr = clf.decision_function(X_text)
                conf = np.abs(arr) if getattr(arr, "ndim", 1) == 1 else arr.max(axis=1)
            elif hasattr(clf, "predict_proba"):
                arr = clf.predict_proba(X_text)
                conf = arr.max(axis=1)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            y_pred = None

        if y_pred is None or len(y_pred) != len(edited):
            st.error(f"Prediction returned {0 if y_pred is None else len(y_pred)} results for {len(edited)} rows. Check model format or input.")
        else:
            result = edited.copy()
            result["PredictedCategory"] = pd.Series(y_pred, index=result.index).astype(str).str.upper()
            result["AssignedSeverity"] = result["PredictedCategory"].apply(assign_severity)
            if conf is not None:
                result["Confidence"] = pd.Series(conf, index=result.index).round(3)

            st.success("Inference complete.")
            st.dataframe(result, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Reports", len(result))
            with c2: st.metric("Unique predicted types", result["PredictedCategory"].nunique())
            with c3: st.metric("Avg confidence", float(result.get("Confidence", pd.Series([0])).mean() or 0))

            st.download_button("Download Predictions (CSV)", data=result.to_csv(index=False), file_name="predictions_level4.csv", mime="text/csv")
            st.download_button("Download Predictions (JSON)", data=result.to_json(orient="records"), file_name="predictions_level4.json", mime="application/json")

            # Quick map preview (if coordinates available)
            try:
                import folium
                from streamlit_folium import folium_static
                df_map = result.dropna(subset=["Latitude (Y)", "Longitude (X)"])
                if not df_map.empty:
                    m = folium.Map(
                        location=[df_map["Latitude (Y)"].median(), df_map["Longitude (X)"].median()],
                        zoom_start=12,
                        tiles="CartoDB positron"
                    )

                    for _, r in df_map.iterrows():
                        folium.CircleMarker(
                            location=[r["Latitude (Y)"], r["Longitude (X)"]],
                            radius=5,
                            popup=f"{r.get('PredictedCategory','?')} | Sev {r.get('AssignedSeverity','?')}",
                        ).add_to(m)
                    st.write("**Map preview (if coordinates present):**")
                    folium_static(m, width=1000, height=450)
                else:
                    st.info("No valid coordinates to map. You can add/edit Lat/Lon in the table above.")
            except Exception as e:
                st.info(f"Map preview unavailable: {e}")