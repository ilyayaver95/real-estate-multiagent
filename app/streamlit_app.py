"""Entry point: multipage navigation for the assistant (Chat + Monitoring).

Streamlit Community Cloud runs this file. The views live in app/views/ and are registered with
st.navigation so the sidebar shows proper page names instead of file names.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Asset Manager Assistant", page_icon="🏢", layout="wide")

chat = st.Page("views/chat.py", title="Chat", icon="💬", default=True)
monitoring = st.Page("views/monitoring.py", title="Monitoring", icon="📊", url_path="monitoring")
st.navigation([chat, monitoring]).run()
