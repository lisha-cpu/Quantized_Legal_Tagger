import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Quantized Legal Tagger",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Quantized Legal Tagger")
st.caption("GPT-2 Legal Clause Classification | Model Compression & Performance Analysis")

if not os.path.exists("benchmark_results.csv"):
    st.error("benchmark_results.csv not found. Run python src/benchmark.py first.")
    st.stop()

df = pd.read_csv("benchmark_results.csv")

original = df[df["Model"] == "Original"].iloc[0]
quantized = df[df["Model"] == "Dynamic INT8"].iloc[0]

size_reduction = (1 - quantized["Size MB"] / original["Size MB"]) * 100
speed_improvement = (1 - quantized["Mean latency ms"] / original["Mean latency ms"]) * 100

st.header("Performance Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Original Model Size",
    f"{original['Size MB']:.2f} MB"
)

c2.metric(
    "Quantized Model Size",
    f"{quantized['Size MB']:.2f} MB",
    f"-{size_reduction:.2f}%"
)

c3.metric(
    "Original Latency",
    f"{original['Mean latency ms']:.2f} ms"
)

c4.metric(
    "Quantized Latency",
    f"{quantized['Mean latency ms']:.2f} ms",
    f"-{speed_improvement:.2f}%"
)

st.divider()

st.header("Model Comparison")

tab1, tab2 = st.tabs(["Model Size", "Inference Latency"])

with tab1:
    st.subheader("Model Size Comparison")
    size_df = df.set_index("Model")["Size MB"]
    st.bar_chart(size_df)
    st.caption("Lower model size means less storage required.")

with tab2:
    st.subheader("Inference Latency Comparison")
    latency_df = df.set_index("Model")["Mean latency ms"]
    st.bar_chart(latency_df)
    st.caption("Lower latency means faster inference.")

st.divider()

st.header("Detailed Benchmark Data")
st.dataframe(df, use_container_width=True)

st.divider()

st.header("Quantization Summary")

st.write(
    f"The dynamic INT8 model reduced storage by **{size_reduction:.2f}%** "
    f"and improved measured inference speed by **{speed_improvement:.2f}%** "
    "compared with the original model in this benchmark."
)

st.success("Benchmark data loaded successfully.")

st.divider()

st.markdown(
    """
    ### About this project
    - **Architecture:** GPT-2
    - **Dataset:** LexGLUE / LEDGAR
    - **Task:** Legal clause classification
    - **Optimization:** Dynamic INT8 quantization
    - **Frameworks:** PyTorch, Hugging Face Transformers, Streamlit
    """
)