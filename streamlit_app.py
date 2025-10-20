import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd
from pathlib import Path

st.set_page_config(page_title='Sentiment Classifier', layout='centered')

MODEL_PATHS = [
    Path('best_pipeline.joblib'),
    Path('logreg_model.joblib')
]

VECT_PATHS = [
    Path('tfidf_vectorizer.joblib')
]

@st.cache_resource
def load_pipeline():
    # Try to load a single pipeline first
    for p in MODEL_PATHS:
        if p.exists():
            try:
                pipe = joblib.load(p)
                return pipe, 'pipeline'
            except Exception:
                pass
    # fallback: load separate model + vectorizer
    model = None
    vec = None
    for p in MODEL_PATHS:
        if p.exists():
            try:
                model = joblib.load(p)
            except Exception:
                pass
    for v in VECT_PATHS:
        if v.exists():
            try:
                vec = joblib.load(v)
            except Exception:
                pass
    if model is not None and vec is not None:
        return (vec, model), 'separate'
    return None, None

pipe_obj, mode = load_pipeline()

st.title('Sentiment Analysis (TF-IDF + Logistic Regression)')
st.write('Enter text below and click Predict. The app will use a saved pipeline if available.')

user_input = st.text_area('Text to classify', value='I love this product. It works perfectly!')

if st.button('Predict'):
    if pipe_obj is None:
        st.error('No trained model pipeline found in the workspace. Please run the training notebook to create `best_pipeline.joblib` or `logreg_model.joblib` + `tfidf_vectorizer.joblib`.')
    else:
        if mode == 'pipeline':
            pipe = pipe_obj
            pred = pipe.predict([user_input])[0]
            probs = pipe.predict_proba([user_input])[0] if hasattr(pipe, 'predict_proba') else None
            st.write('Prediction:', int(pred))
            if probs is not None:
                st.write('Probability (per class):', probs.tolist())
            # if pipeline has a named tfidf step we can extract feature names and classifier coeffs
            try:
                tfidf = pipe.named_steps['tfidf']
                clf = pipe.named_steps['clf']
                vec_input = tfidf.transform([user_input])
                coefs = clf.coef_[0]
                # compute token contributions
                feature_names = tfidf.get_feature_names_out()
                nonzero_idx = vec_input.nonzero()[1]
                contribs = []
                for idx in nonzero_idx:
                    contribs.append((feature_names[idx], coefs[idx], vec_input[0, idx]))
                contribs = sorted(contribs, key=lambda x: -abs(x[1]*x[2]))[:20]
                if contribs:
                    st.write('Top contributing tokens (token, coef, tfidf):')
                    st.table(pd.DataFrame(contribs, columns=['token','coef','tfidf']))
            except Exception:
                pass
        elif mode == 'separate':
            vec, model = pipe_obj
            Xq = vec.transform([user_input])
            pred = model.predict(Xq)[0]
            probs = model.predict_proba(Xq)[0] if hasattr(model, 'predict_proba') else None
            st.write('Prediction:', int(pred))
            if probs is not None:
                st.write('Probability (per class):', probs.tolist())
            try:
                feature_names = vec.get_feature_names_out()
                coefs = model.coef_[0]
                nonzero_idx = Xq.nonzero()[1]
                contribs = []
                for idx in nonzero_idx:
                    contribs.append((feature_names[idx], coefs[idx], Xq[0, idx]))
                contribs = sorted(contribs, key=lambda x: -abs(x[1]*x[2]))[:20]
                if contribs:
                    st.write('Top contributing tokens (token, coef, tfidf):')
                    st.table(pd.DataFrame(contribs, columns=['token','coef','tfidf']))
            except Exception:
                pass

st.markdown('---')
st.write('Model files checked in workspace:')
for p in MODEL_PATHS + VECT_PATHS:
    st.write(p.name, '-', 'FOUND' if p.exists() else 'MISSING')
