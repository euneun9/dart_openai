import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings  # langchain-huggingface에서 가져오기
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.schema.runnable import RunnablePassthrough
import locale
import warnings

# UTF-8 설정
def getpreferredencoding(do_setlocale=True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

# 모델과 토크나이저 로드
model_id = "kyujinpy/Ko-PlatYi-6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", torch_dtype=torch.float16)

# 텍스트 생성 파이프라인 설정
text_generation_pipeline = pipeline(
    model=model, tokenizer=tokenizer, task="text-generation", temperature=0.2, return_full_text=True, max_new_tokens=300
)

# Prompt 템플릿과 LLM 체인 설정
prompt_template = """
### [INST]
Instruction: Answer the question based on your knowledge.
Here is context to help:

{context}

### QUESTION:
{question}

[/INST]
"""
prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
koplatyi_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# 체인을 구성하여 LLM과 연결
llm_chain = prompt | koplatyi_llm

# PDF 파일 로드 및 텍스트 분할
pdf_path = "./[삼성물산]반기보고서(2024.08.14).pdf"  # 실제 PDF 파일 경로로 수정
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(pages)

# 임베딩 모델 및 벡터 데이터베이스 생성
embedding_model_name = "jhgan/ko-sbert-nli"
hf_embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, encode_kwargs={'normalize_embeddings': True})
db = FAISS.from_documents(texts, hf_embeddings)
retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 3})

# RAG 체인 설정
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | llm_chain
warnings.filterwarnings('ignore')

# 질문 실행 및 결과 출력
result = rag_chain.invoke("삼성전자에 대해 알려줘.")

for i in result['context']:
    print(f"주어진 근거: {i.page_content} / 출처: {i.metadata['source']} - {i.metadata['page']} \n\n")

print(f"\n답변: {result['text']}")
