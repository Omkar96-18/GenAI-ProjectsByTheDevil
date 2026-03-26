import os
import pandas as pd
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations


# LangChain NEW IMPORTS (UPDATED)
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# LangGraph
from langgraph.graph import StateGraph

# -------------------------------
# ENV LOAD
# -------------------------------


# -------------------------------
# LLM SETUP (OpenRouter)
# -------------------------------
llm = ChatOllama(
    model='llama3:8b-instruct-q4_k_m',
    temperature=0.7
)

# -------------------------------
# STREAMLIT CONFIG
# -------------------------------
st.set_page_config(page_title="AI Schema Engine", layout="wide")

st.title("🧠 AI Relationship Detection Engine (LangGraph Powered)")

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_files = st.file_uploader(
    "Upload CSV files",
    type=["csv"],
    accept_multiple_files=True
)

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def clean(col):
    return col.dropna()

def uniqueness(col):
    return col.nunique() / len(col) if len(col) else 0

# -------------------------------
# STEP 1: DETECTION NODE
# -------------------------------
def detect_node(state):
    tables = state["tables"]
    relationships = []

    def clean(col):
        return col.dropna()

    def uniqueness(col):
        return col.nunique() / len(col) if len(col) else 0

    # -------------------------------
    # STEP 1: PK DETECTION
    # -------------------------------
    pk_map = {}

    for t, df in tables.items():
        pk_map[t] = {}
        for c in df.columns:
            col = clean(df[c])
            if len(col) == 0:
                continue

            u = uniqueness(col)
            pk_map[t][c] = u > 0.95  # relaxed PK detection

    # -------------------------------
    # STEP 2: COLUMN COMPARISON
    # -------------------------------
    for (t1, df1), (t2, df2) in combinations(tables.items(), 2):

        for c1 in df1.columns:
            for c2 in df2.columns:

                col1 = clean(df1[c1])
                col2 = clean(df2[c2])

                if len(col1) == 0 or len(col2) == 0:
                    continue

                # TYPE CHECK
                if col1.dtype != col2.dtype:
                    continue

                set1 = set(col1)
                set2 = set(col2)

                common = set1.intersection(set2)

                if len(common) == 0:
                    continue

                # -------------------------------
                # STEP 3: OVERLAP (RELAXED)
                # -------------------------------
                overlap = len(common) / max(len(set1), len(set2))

                if overlap < 0.3:
                    continue

                # -------------------------------
                # STEP 4: SUBSET CHECK (BOTH WAYS)
                # -------------------------------
                subset_1_to_2 = len(common) / len(set1)
                subset_2_to_1 = len(common) / len(set2)

                # -------------------------------
                # STEP 5: UNIQUENESS
                # -------------------------------
                u1 = uniqueness(col1)
                u2 = uniqueness(col2)

                # -------------------------------
                # STEP 6: DETERMINE DIRECTION
                # -------------------------------
                relation = None
                direction = None

                # Case: col1 → col2
                if subset_1_to_2 > 0.8:
                    if u1 < 0.7 and u2 > 0.9:
                        relation = "MANY_TO_ONE"
                        direction = (t1, c1, t2, c2)

                # Case: col2 → col1
                if subset_2_to_1 > 0.8:
                    if u2 < 0.7 and u1 > 0.9:
                        relation = "MANY_TO_ONE"
                        direction = (t2, c2, t1, c1)

                # -------------------------------
                # STEP 7: PK ANCHOR BOOST 🔥
                # -------------------------------
                if pk_map[t2].get(c2, False) and subset_1_to_2 > 0.7:
                    relation = "MANY_TO_ONE"
                    direction = (t1, c1, t2, c2)

                if pk_map[t1].get(c1, False) and subset_2_to_1 > 0.7:
                    relation = "MANY_TO_ONE"
                    direction = (t2, c2, t1, c1)

                # -------------------------------
                # STEP 8: SAVE RELATIONSHIP
                # -------------------------------
                if relation and direction:
                    src_t, src_c, tgt_t, tgt_c = direction

                    score = round(
                        0.5 * overlap +
                        0.3 * max(subset_1_to_2, subset_2_to_1) +
                        0.2 * (1 if pk_map[tgt_t].get(tgt_c, False) else 0),
                        2
                    )

                    if score > 0.6:
                        relationships.append({
                            "from": f"{src_t}.{src_c}",
                            "to": f"{tgt_t}.{tgt_c}",
                            "relation": relation,
                            "score": score
                        })

    return {"relationships": relationships}
# -------------------------------
# STEP 2: LLM NODE
# -------------------------------
def llm_node(state):
    relationships = state["relationships"]

    if not relationships:
        return {"explanation": "No relationships found."}

    prompt = f"""
    Explain these database relationships clearly:

    {relationships}

    Include:
    - Why they exist
    - Type of relationship
    - Real-world meaning
    """

    response = llm.invoke([HumanMessage(content=prompt)])

    return {"explanation": response.content}

# -------------------------------
# LANGGRAPH SETUP
# -------------------------------
builder = StateGraph(dict)

builder.add_node("detect", detect_node)
builder.add_node("llm", llm_node)

builder.set_entry_point("detect")
builder.add_edge("detect", "llm")

graph_flow = builder.compile()

# -------------------------------
# MAIN UI LOGIC
# -------------------------------
if uploaded_files:

    tables = {}
    for file in uploaded_files:
        df = pd.read_csv(file).head(500)
        tables[file.name.replace(".csv", "")] = df

    st.success(f"Loaded {len(tables)} tables")

    if st.button("🚀 Run AI Pipeline"):

        # RUN LANGGRAPH
        result = graph_flow.invoke({"tables": tables})

        relationships = result.get("relationships", [])
        explanation = result.get("explanation", "")

        # -------------------------------
        # SHOW RELATIONSHIPS
        # -------------------------------
        st.subheader("📊 Detected Relationships")
        st.json(relationships)

        # -------------------------------
        # SHOW LLM OUTPUT
        # -------------------------------
        st.subheader("🧠 AI Explanation")
        st.write(explanation)

        # -------------------------------
        # GRAPH VISUALIZATION
        # -------------------------------
        if relationships:
            st.subheader("🌐 Relationship Graph")

            G = nx.DiGraph()

            for r in relationships:
                src = r["from"].split(".")[0]
                tgt = r["to"].split(".")[0]

                G.add_edge(src, tgt, label=r["relation"])

            fig, ax = plt.subplots(figsize=(10, 6))
            pos = nx.spring_layout(G)

            nx.draw(
                G, pos,
                with_labels=True,
                node_size=2500,
                node_color="lightblue",
                ax=ax
            )

            edge_labels = nx.get_edge_attributes(G, 'label')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

            st.pyplot(fig)

        else:
            st.warning("No relationships detected")