from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

def retrieval_qdrant(question):
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-mpnet-base-v2")

    client = QdrantClient(host="localhost", port=6333)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name="demo_collection",
        embedding=embeddings,
    )

    db_results = vector_store.similarity_search(
        question, k=3
    )

    retrieval_result = []
    for res in db_results:
        retrieval_result.append(res.page_content)
    
    return retrieval_result


# retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})

# print(retriever.invoke("대출계약 체결로 부담해야 하는 금액은 뭐야?"))