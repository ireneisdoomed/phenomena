from dotenv import load_dotenv
import os

from langchain.embeddings.openai import OpenAIEmbeddings
import streamlit as st

from scripts.query_index_by_text import get_most_likely_efos, get_phenotypes_for_disease, load_index

st.title('Phenomena - Phenotype Similarity Search')

st.write("""
    This is a prototype of a search engine for phenotypes. It uses the OpenAI API to embed phenotypes and Pinecone to index them. The search engine is
    currently limited to the phenotypes in the Experimental Factor Ontology (EFO).

    Take a visual look at our disease universe here: https://atlas.nomic.ai/map/8f94a626-f4c2-4722-a8f9-025654e47c47/06f74b36-a1f3-4850-a955-e36c875b06dd

    Our goal is to provide a tool that helps researchers to:
    - **Uncover new mechanistic insights**. Identification of new molecular pathways that are involved in the disease, and which may be potential drug targets.
    - **Uncover new disease biomarkers**. Identification of molecules or patterns tart are over-represented in the presence of disease. 
    - **Analysis of gene to phenotype signals within a context**. Identification and subtyping of diseases help in the interpretation of already known gene to phenotype signals, instead of treating all associations in an isolated manner. 
    """
)

try:
    index = load_index()
    print("Index loaded")
except Exception:
    print("Index could not be found. Please check your Pinecone API key.")

load_dotenv()
openai_key = os.getenv("OPENAI_TOKEN")
embeddings_client = OpenAIEmbeddings(openai_api_key=openai_key, model="text-embedding-ada-002")


st.subheader("🦠 Search for a disease")
if text := st.text_input(
    "Look up your disease here: ", placeholder="diabetes mellitus"
):
    st.write(f"Looking up {text}...")
    vector = embeddings_client.embed_query(text)
    efo, vector = get_most_likely_efos(vector, index)
    st.markdown(f"The mapped term for {text} is: **{efo}**")


    st.subheader(f"🔗 Find phenotypes of {text}")
    similarity_threshold = st.slider('Set a similarity threshold', 0.0, 1.0, 0.8)
    if st.button("Get the top 20 more similar phenotypes"):
        phenotypes = get_phenotypes_for_disease(vector, 20, similarity_threshold, index)
        st.markdown("### These are the top 20 phenotypes")
        for pheno in phenotypes:
            ot_template="https://platform.opentargets.org/disease/"
            st.write(f'Phenotype ID: {ot_template + pheno["id"]} ({pheno["score"]})')

st.markdown("""---""")
st.subheader("🤔 How does this work?")
st.markdown("""
    - Starting point: Open Targets Literature Matches Dataset - Disease entity recognition on literature
    - Semantic understanding of diseases: OpenAI API - Text Embeddings
    - Similarity search: Pinecone - Cloud-native vector search engine
    - Visualisation: Atlas - Vector visualisation
""")

st.markdown("""---""")
st.markdown("See the code on [GitHub](https://github.com/ireneisdoomed/phenomena)")