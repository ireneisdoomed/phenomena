# Phenomena

The project involves using machine learning and natural language processing techniques to extract and mine disease-phenotype relationships from large datasets that may not be obvious or easily discoverable through traditional methods such as manual curation.

The expected timelines and deliverables are described in the Projects tab.

## What are we trying to solve?

Having a better understanding of the relationships between diseases and phenotypes can help in many ways:

- **Uncover new mechanistic insights**. Identification of new molecular pathways that are involved in the disease, and which may be potential drug targets.
- **Uncover new disease biomarkers**. Identification of molecules or patterns tart are over-represented in the presence of disease. 
- **Analysis of gene to phenotype signals within a context**. Identification and subtyping of diseases help in the interpretation of already known gene to phenotype signals, instead of treating all associations in an isolated manner. 

## How are we going to solve it?

We will leverage deep phenotyping data and ontology-based approaches to identify these clusters.
The latest developments in text mining and large language models will be used to extract information from information rich sources such as the **scientific literature** and **electronic health records**.

### Conceptual framework

- Generation of a dataset that establishes the most relevant phenotypes for each disease - and analysis of the findings we can derive from that.
  - Entity relationship extraction from the literature/electronic health records.
  - Embedding of the extracted entities into a vector space.
  - Clustering of the entities based on their similarity.

- Normalization of the extracted phenotypes to a common ontology to differentiate between diseases and phenotypes.
  - Generation of a vector space representation of the ontology.
  - Similarity search of the extracted phenotypes and diseases with the ontology terms.

- Benchmark with the manually curated disease to phenotype available from Open Targets.
  - Extract accuracy and specificity of the method.
  - Extract coverage gain to reinforce our value proposal.

### Technical framework

#### 1. Generation of a dataset that establishes the most relevant phenotypes for each disease.

- To mine the literature we will use the PUBMED dataset as a base, which contains over 30 million scientific articles and is made available in the HuggingFace platform here: https://huggingface.co/datasets/pubmed. As an alternative, and if the missing data is too large, we can use [The Pile dataset](https://pile.eleuther.ai/).

- To extract the phenotypes/diseases from the literature we will use one of the many pre-trained models available.
  - BioBERT Diseases NER: [alvaroalon2/biobert_diseases_ner](https://huggingface.co/alvaroalon2/biobert_diseases_ner)
  - Stanford's BioMedLM: https://huggingface.co/stanford-crfm/BioMedLM
  - SparkNLP Disease NER pipeline: https://nlp.johnsnowlabs.com/2023/03/14/ner_diseases_large_pipeline_en.html
  - Biomedical NER All (distilbert-base-uncased): https://huggingface.co/d4data/biomedical-ner-all

- To ground the extracted entities to EFO we will first create a vector space with the ontology embeddings and then we will cluster the entities based on the similarity of their encoded vector with the terms in the ontology vector store.
    - As vector stores there are many we can use: FAISS (simple and local), LlamaIndex, ChromaDB or Pinecone (cloud-based). Langchain has wrappers around the main vector stores that accommodate common methods for similarity search and clustering, so I'd suggest using implementing them with Langchain in case we want to change the library in the future.

- Determination of a good similarity score that would allow us to extract the most relevant phenotypes for each disease. Make available the dataset in HuggingFace.

#### 2. Visualization of the top k phenotypes for each disease in a web application.

- Creation of a graph database that stores the extracted phenotypes and diseases.
- Visualization of the top k phenotypes for each disease in a web application.
  - Possible FE frameworks: Streamlit, Gradio (we could host it in HF).
  - Possible graph visualization library: memgraph.com

## Results
TBC