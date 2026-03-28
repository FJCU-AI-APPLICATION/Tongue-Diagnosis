"""Streamlit application entry point."""

import streamlit as st
from tongue_backend import __version__ as backend_version
from tongue_ai.classification import classify_tongue

st.set_page_config(page_title="Tongue Diagnosis", layout="wide")
st.title("Tongue Diagnosis System")
st.write(f"Backend version: {backend_version}")

uploaded = st.file_uploader("Upload a tongue image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    st.image(uploaded, caption="Uploaded image", use_container_width=True)
    result = classify_tongue({"coating": "thin_white", "color": "pale_red", "shape": "normal"})
    st.json(result)
else:
    st.info("Please upload a tongue image to begin analysis.")
