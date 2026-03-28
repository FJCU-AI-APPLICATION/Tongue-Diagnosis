"""Streamlit application entry point."""

import json

import httpx
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Tongue Diagnosis", layout="wide")
st.title("Tongue Diagnosis System")

# Sidebar: backend connection status
with st.sidebar:
    st.header("Backend")
    try:
        resp = httpx.get(f"{API_BASE}/health", timeout=5)
        info = resp.json()
        st.success(f"Connected — AI v{info['ai_version']}")
    except httpx.ConnectError:
        st.error("Backend not reachable. Start it with:\n\n"
                 "`uv run uvicorn tongue_backend.app:app --port 8000`")

    use_streaming = st.toggle("Stream results (SSE)", value=False)

uploaded = st.file_uploader("Upload a tongue image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    st.image(uploaded, caption="Uploaded image", use_container_width=True)

    if st.button("Analyze"):
        uploaded.seek(0)
        files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}

        if use_streaming:
            # SSE streaming endpoint
            st.subheader("Streaming Analysis")
            with httpx.stream("POST", f"{API_BASE}/api/analyze/stream",
                              files=files, timeout=60) as resp:
                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[len("data: "):]
                    if payload == "[DONE]":
                        st.success("Analysis complete")
                        break
                    event = json.loads(payload)
                    st.subheader(event["step"].title())
                    st.json(event["result"])
        else:
            # Single HTTP request
            with st.spinner("Analyzing..."):
                resp = httpx.post(f"{API_BASE}/api/analyze",
                                 files=files, timeout=60)
                result = resp.json()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("Detection")
                st.json(result["detection"])
            with col2:
                st.subheader("Features")
                st.json(result["features"])
            with col3:
                st.subheader("Classification")
                st.json(result["classification"])
else:
    st.info("Please upload a tongue image to begin analysis.")
