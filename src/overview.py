# src/overview.py
import streamlit as st
from datetime import datetime

def render(df):
    base_ctr = df["click"].mean()
    time_window = f"{df['dt'].min()} → {df['dt'].max()}"

    st.title("Next Best Action (NBA) — Prototype")
    st.write("""
    **Use case**: predecir propensión a interactuar (CTR) para priorizar la *próxima mejor acción*.  
    **Dataset**: Avazu CTR sample (50k). Anónimo, tabular, 10 días, binaria (click/no click).
    **Limitations**: contexto publicitario y features anonimizadas; transferimos el enfoque a NBA (propensionado).
    """)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Cols", f"{df.shape[1]}")
    c3.metric("Base CTR", f"{base_ctr:.4f}")
    c4.metric("Time window", time_window)

    with st.expander("Rationale & Scope"):
        st.markdown("""
- **Por qué Avazu**: problema binario tabular bien conocido (CTR) → mapa limpio a *propensity modeling* para NBA.
- **Objetivo**: prototipar scoring de propensión y entender drivers (no entrenar en vivo).
- **Validación**: split temporal (último día como validación) para simular producción.
""")
