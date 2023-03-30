from dotenv import load_dotenv
import os

from langchain.embeddings.openai import OpenAIEmbeddings
import streamlit as st

from scripts.query_index_by_text import get_most_likely_efos, get_phenotypes_for_disease, load_index

st.title('Phenomena - Phenotype Similarity Search')

st.write("""
    This is a prototype of a search engine for phenotypes. It uses the OpenAI API to embed phenotypes and Pinecone to index them. The search engine is
    currently limited to the phenotypes in the Experimental Factor Ontology (EFO).
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


st.subheader("Search for a disease")
if text := st.text_input(
    "Look up your disease here: ", placeholder="diabetes mellitus"
):
    st.write(f"Looking up {text}...")
    vector = embeddings_client.embed_query(text)
    efo, vector = get_most_likely_efos(vector, index)
    st.write(f"The mapped term for {text} is: {efo}")


    st.subheader(f"Find phenotypes of {text}")
    similarity_threshold = st.slider('Set a similarity threshold', 0.0, 1.0, 0.8)
    if st.button("Get the top 20 more similar phenotypes"):
        phenotypes = get_phenotypes_for_disease(vector, 20, similarity_threshold, index)
        st.markdown("### These are the top 20 phenotypes")
        for pheno in phenotypes:
            st.write(f'Phenotype ID: {pheno["id"]} ({pheno["score"]})')

