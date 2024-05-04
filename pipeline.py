from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators.openai import OpenAIGenerator
from haystack.components.generators.utils import print_streaming_chunk
from haystack.utils import Secret
from haystack import Pipeline
from datasets import load_dataset
from haystack import Document
import streamlit as st
import os

OPENAI_KEY = os.getenv("OPENAI_KEY")
DATA_DIR = "data/Letters"


def clean_data(doc):
    if isinstance(doc, dict):
        return doc.get('text', '')
    else:
        return doc


@st.cache_resource
def rag_pipeline() -> Pipeline:
    
    docs = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.txt'):  # Ensure we're reading text files
            file_path = os.path.join(DATA_DIR, filename)
            with open(file_path, 'r') as f:
                text = f.read()
                docs.append(Document(content=text))

    doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    doc_embedder.warm_up()
    docs_with_embeddings = doc_embedder.run(docs)["documents"]

    document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
    document_store.write_documents(docs_with_embeddings)
    
    retriever = InMemoryEmbeddingRetriever(document_store=document_store)

    text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    
    # Answer user question, focusing on the most relevant document
    template = """
    Briefly answer the question exclusively based on the following documents. If you can't locate an answer, reply "I can't seem to deduce that from the documents provided. Please rephrase.":

    {% for document in documents[:1] %}   
        {{ document.content }}
    {% endfor %}
    Question: {{question}}
    Answer:
    """

    prompt_builder = PromptBuilder(template=template)    

    generator = OpenAIGenerator(api_key=Secret.from_token(OPENAI_KEY), model="gpt-3.5-turbo", streaming_callback=print_streaming_chunk)

    rag_pipeline = Pipeline()

    rag_pipeline.add_component("text_embedder", text_embedder)
    rag_pipeline.add_component("retriever", retriever)
    rag_pipeline.add_component("prompt_builder", prompt_builder)
    rag_pipeline.add_component("llm", generator)

    # Connect the components to each other
    rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")

    return rag_pipeline
