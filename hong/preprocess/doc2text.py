import re
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

def pdf2text(pdf_path):
    pdf = pdfplumber.open(pdf_path)
    pages = pdf.pages
    text = ""
    for page in pages:
        sub = page.extract_text()
        text += sub

    # 특수문자 제거
    cleaned_text = re.sub(r'[^\w\s.,]', '', text)

    # .를 기준으로 문장단위 chunk 생성
    tokenized_sentence = cleaned_text.split(".")

    # 1000자씩 100자를 겹쳐서 chunk 생성
    # text_splitter = RecursiveCharacterTextSplitter(
    #     # Set a really small chunk size, just to show.
    #     chunk_size=200,
    #     chunk_overlap=20,
    #     length_function=len,
    #     is_separator_regex=False,
    # )
    # tokenized_sentence = text_splitter.split_text(cleaned_text)

    return tokenized_sentence