from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams

from uuid import uuid4
from langchain_core.documents import Document

from .doc2text import pdf2text

def setterQdrant(pdf_path):
    embeddings = HuggingFaceEmbeddings(
        model_name="models/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={'device':'cuda'}
    )

    client = QdrantClient(host="localhost", port=6333)

    client.create_collection(
        collection_name="demo_collection",
        vectors_config=VectorParams(size= 768,distance=Distance.COSINE),
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name="demo_collection",
        embedding=embeddings,
    )

    sentences = pdf2text(pdf_path)

    documents = []

    for sen in sentences:
        doc = Document(
            page_content=sen,
            metadata={"source": f"{pdf_path}"},
        )
        documents.append(doc)

    uuids = [str(uuid4()) for _ in range(len(documents))]

    vector_store.add_documents(documents=documents, ids=uuids)

def delete_collection():
    client = QdrantClient(host="localhost", port=6333)
    client.delete_collection(collection_name='demo_collection')