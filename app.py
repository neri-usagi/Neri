import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
import tempfile
import os

# 페이지 설정
st.set_page_config(page_title="AI 기말고사 문제 생성기", layout="wide")

st.title("🎓 AI 기말고사 지능형 문제 생성기")
st.markdown("PDF 파일을 업로드하면 AI가 중요도를 분석하여 맞춤형 문제를 생성합니다.")

# 사이드바: 설정
with st.sidebar:
    st.header("⚙️ 설정")
    api_key = st.text_input("OpenAI API Key를 입력하세요", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    model_name = st.selectbox("모델 선택", ["gpt-4o", "gpt-3.5-turbo"])
    num_questions = st.slider("생성할 문제 수", 1, 10, 5)
    difficulty = st.select_slider("난이도", options=["하", "중", "상"])

# 메인 로직
uploaded_file = st.file_uploader("학습 자료 PDF를 업로드하세요", type="pdf")

if uploaded_file and api_key:
    # 1. PDF 로드 및 텍스트 추출
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    
    # 2. 텍스트 분할 (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # 3. 벡터 데이터베이스 생성 (RAG)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)
    retriever = vectorstore.as_retriever()

    # 4. 중요도 분석 및 문제 생성
    llm = ChatOpenAI(model=model_name, temperature=0.7)

    if st.button("🚀 문제 생성 시작"):
        with st.spinner("AI가 자료를 분석하여 기말고사 문제를 출제 중입니다..."):
            
            # 중요도 분석 프롬프트
            importance_prompt = f"""
            당신은 교육 전문가입니다. 다음 텍스트를 분석하여 기말고사에서 가장 중요하게 다뤄질 개념 3가지를 선정하고 이유를 설명하세요.
            이후, 해당 개념들을 바탕으로 {num_questions}개의 문제를 만드세요.
            난이도: {difficulty}
            
            형식:
            1. 중요 개념 및 분석 결과
            2. 문제 세트 (질문, 보기, 정답, 해설 포함)
            """
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever
            )
            
            response = qa_chain.invoke(importance_prompt)
            
            st.success("✅ 문제 생성 완료!")
            st.markdown("---")
            st.markdown(response["result"])
            
            # 결과 다운로드 기능
            st.download_button(
                label="📄 문제지 다운로드 (TXT)",
                data=response["result"],
                file_name="exam_questions.txt",
                mime="text/plain"
            )

elif not api_key:
    st.warning("⚠️ 사이드바에 OpenAI API Key를 입력해야 작동합니다.")

# 임시 파일 삭제
if 'tmp_path' in locals():
    os.remove(tmp_path)
