import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import tempfile
import os

st.set_page_config(page_title="Gemini 문제 생성기", layout="wide")

st.title("🎓 Gemini 기반 기말고사 문제 생성기")

with st.sidebar:
    st.header("⚙️ 설정")
    # 구글 API 키 입력
    api_key = st.text_input("Google API Key를 입력하세요", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    
    num_questions = st.slider("문제 수", 1, 10, 5)

uploaded_file = st.file_uploader("PDF 업로드", type="pdf")

if uploaded_file and api_key:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # Gemini 임베딩 사용
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(texts, embeddings)
    retriever = vectorstore.as_retriever()

    # Gemini 모델 사용
    llm = ChatGoogleGenerativeAI(model="gemini-pro")

    if st.button("🚀 문제 생성 시작"):
        with st.spinner("Gemini가 분석 중..."):
            qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
            prompt = f"{num_questions}개의 기말고사 문제를 형식에 맞춰 만들어줘. 질문 | 보기 | 정답 | 해설 순서로 작성해."
            response = qa_chain.invoke(prompt)
            st.markdown(response["result"])

    os.remove(tmp_path)

