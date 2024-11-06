## 자연어 처리

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_upstage import UpstageEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_upstage import ChatUpstage
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import openai

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from sqlalchemy import create_engine, inspect
from langchain import LLMChain, PromptTemplate

##docs
import pandas as pd
from langchain.schema import Document  
from rank_bm25 import BM25Okapi
import numpy as np
import nltk
from nltk.tokenize import word_tokenize

from config import answer_examples

store = {}

# 세션 채팅 이력
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


## Vector DB로부터 retreiver 수행 
def get_retriever():
    embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
    index_name = "purchase-index-v3"
    
    database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
    retriever = database.as_retriever()

    return retriever


# retriever 이력
def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"), # 나중에 시스템에서 메세지 목록으로 들어가는 값
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever


## LLM 모델 (gpt-3.5-turbo, gpt-4o)
def get_llm(model='gpt-3.5-turbo'):
# def get_llm(model='gpt-4o'):
    llm = ChatOpenAI(model=model)
    return llm

# 키워드 사전
def get_dictionary_chain():
    #dictionary = ["비밀번호를 나타내는 표현 -> 패스워드"]
    dictionary = ["회사를 나타내는 표헌 -> 파트너사"]
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요
        사전: {dictionary}
        
        질문: {{question}}
    """)

    dictionary_chain = prompt | llm | StrOutputParser()
    
    return dictionary_chain


def get_rag_chain():
    llm = get_llm()
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )

    system_prompt = (
        "당신은 파트너사 분석가입니다. 파트너사에 대한 정확한 정보를 수집하고 분석하여 경영진에게 보고하는 것을 목표로 사용자의 질문에 답변해주세요"
        "아래에 제공된 문서를 활용해서 답변해주시고, 문서내에서 답변을 알 수 없다면 모른다고 답변해주세요. 가독성이 좋게 답변해주세요."
        "\n\n"
        "{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            # few_shot_prompt,
            # MessagesPlaceholder("chat_history"), #나중에 들어가는 값
            ("human", "{input}"),
            #("system", "현재 문맥: {context}"),  # context를 출력하는 부분
        ]
    )


    # 벡터 DB에서 검색기 생성
    retriever = get_retriever()
    # RAG 체인 생성
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    # history_aware_retriever를 사용하지 않고 단순한 retrieval chain 생성
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain  # history 관련 부분을 제거

def get_ai_response(user_message):
    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain()

    # BM25 결과 가져오기
    #bm25_results = get_bm25(user_message)
    bm25_results = get_docs(user_message)

    # BM25 결과를 문맥으로 변환
    bm25_context = "\n\n".join(bm25_results)  # BM25 결과를 문맥으로 변환

    # 사용자 질문과 BM25 결과를 rag_chain에 전달하여 최종 응답을 받습니다.
    ai_response = rag_chain.invoke(
        {
            "input": user_message,
            "context": bm25_context  # BM25 결과를 문맥으로 전달
        },
        config={"configurable": {"session_id": "fhj"}}  # session_id 포함
    )

    print("***********************************")
    print(bm25_context)
    # metadata_list = [doc.metadata for doc in ai_response.get('context', [])]
    # print(metadata_list)
    print("###################################")

    
    return ai_response.get('answer')

def get_ai_response_origin(user_message):
    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain()
    # security_chain = {"input": dictionary_chain} | rag_chain

    # ai_response = security_chain.stream(
    #     {
    #         "question": user_message
    #     },
    #     config={
    #         "configurable": {"session_id": "abcd"}
    #     },
    # )

    # # dictionary_chain을 실행하여 질문을 변환합니다.
    # transformed_question = dictionary_chain.invoke({"question": user_message})

    # # 변환된 질문을 rag_chain에 전달하여 최종 응답을 받습니다.
    # ai_response = rag_chain.invoke(
    #     {"input": transformed_question},
    #     config={"configurable": {"session_id": "2"}}  # 여기서 session_id를 추가
    # )



    # 사용자 질문을 rag_chain에 직접 전달하여 최종 응답을 받습니다.
    ai_response = rag_chain.invoke(
        {"input": user_message},
        config={"configurable": {"session_id": "zxczxczxc"}}  # session_id를 여전히 포함
    )

    # print("###################################")
    # metadata_list = [doc.metadata for doc in ai_response.get('context', [])]
    # print(metadata_list)

    return ai_response.get('answer')


def get_bm25(user_message):
    ###########-START-##########
    # Pinecone에서 문서 가져오기
    retriever = get_retriever()
    # 사용자 쿼리에 기반하여 문서 검색
    pinecone_documents = retriever.invoke(user_message, k=3)  # k는 반환할 문서 수

    # 문서 내용 추출
    documents = [doc.page_content for doc in pinecone_documents]  # 문서의 내용을 리스트로 저장
    ##########-END-##########

    # 문서 토큰화
    tokenized_docs = [doc.split(" ") for doc in documents] # 공백으로 분리
    # BM25 초기화
    bm25 = BM25Okapi(tokenized_docs)

    # 사용자 쿼리
    query = user_message
    tokenized_query = query.split(" ")

    # BM25 검색
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_n = bm25_scores.argsort()[-3:][::-1]  # 상위 3개 문서 인덱스

    # BM25 결과
    bm25_results = [documents[i] for i in bm25_top_n]

    return bm25_results


def get_docs(user_message):

    # 엑셀 파일 읽기
    df = pd.read_excel('data_v0.5.xlsx')

    # 공백 제거
    df.columns = df.columns.str.strip()
    # 첫 번째 row의 모든 값이 'Unnamed'로 시작하는지 확인
    if df.columns.str.startswith('Unnamed').all():
        # 첫 번째 행을 삭제하고, 그 다음 행을 새로운 헤더로 설정
        df = pd.read_excel("./data_v0.5.xlsx", header=[1,1])
        # 모든 값이 NaN인 행 삭제
        df = df.dropna(axis=0, how='all')
        # 모든 값이 NaN인 열 삭제
        df = df.dropna(axis=1, how='all')

    # 엑셀 데이터 중 빈 값은 NaN으로 표시되는데 임베딩 중 에러로 인해 이 값을 string NaN으로 변경
    df = df.replace(np.NaN, '')
    df = df.replace('\n', ', ', regex=True)
    df = df.replace(r'\[', '(', regex=True)
    df = df.replace(r'\]', ')', regex=True)

    # 엑셀 특성 상 셀 병합으로 인해 column명이 2개의 행을 차지하는 경우가 있음, 이 경우 데이터를 읽었을 때, Unnamed로 표시됨, Unnamed 필터링
    # 엑셀 특성 상 컬럼 이름을 분류하는 상위 컬럼이 존재 할 수 있음 Ex) 분류 -> 대 중 소, 이런 컬럼들을 공백을 사이에 두고 합쳐준다. Ex) 분류 -> 대 중 소 --> 분류 대, 분류 중, 분류 소
    df.columns = [' '.join(col).strip() if not col[1].startswith('Unnamed') else col[0] for col in df.columns.values]

    # 문서 리스트 준비
    documents = []
    for index, row in df.iterrows():
        content = f"""이 자료는 롯데 그룹의 롯데 이노베이트(IT SI/SM) 회사가 협력하는 파트너사들의 정보를 정리한 문서입니다. 이 문서에는 각 파트너사의 재무 정보와 프로젝트 경험이 포함되어 있습니다.
    롯데 이노베이트는 롯데 그룹의 여러 계열사(호텔, 면세점 등)에 시스템 통합(SI) 및 시스템 관리(SM) 서비스를 제공하며, 다양한 프로젝트에 파트너사와 함께 참여하고 있습니다. 이 자료는 적합한 파트너사를 선정하는 데 활용됩니다.
    아래는 {str(row['파트너사'])} 기업의 정보입니다. 이 기업이 파트너사로 적합한지 판단하는 근거가 됩니다.


    {str(row['파트너사'])} 정보
            
    - 파트너사(회사명): {str(row['파트너사'])}
    - 주요업종: {str(row['주요업종'])}
    - 업종 구분: {str(row['구분'])}
    - 업종 구분시 대분류: {str(row['대분류'])}
    - 업종 구분시 중분류: {str(row['중분류'])}
    - 업종 구분시 소분류: {str(row['소분류'])}
    - 협약사 여부: {
        '협약사임' if str(row['협약사 여부']) == 'O' else 
        ('협약사 아님' if str(row['협약사 여부']) == 'X' else 
        '해당 정보 없음')
    }
    - 보유 솔루션명: {str(row['솔루션명']) if str(row['솔루션명']) != '' else '해당 정보 없음'}
    - 직원수: {str(row['직원수(명)'])+'명' if str(row['직원수(명)']).isdigit() else '해당 정보 없음'}
    - 2021년 매출액: {str(row['2021년 매출(백만원)'])+'00만원' if str(row['2021년 매출(백만원)']) != '' else '해당 정보 없음'}
    - 2022년 매출액: {str(row['2022년 매출(백만원)'])+'00만원' if str(row['2022년 매출(백만원)']) != '' else '해당 정보 없음'}
    - 2023년 매출액: {str(row['2023년 매출(백만원)'])+'00만원' if str(row['2023년 매출(백만원)']) != '' else '해당 정보 없음'}
    - 사업부 선호도: {str(row['사업부 선호도']) if str(row['사업부 선호도']) != '' else '해당 정보 없음'}
    - 참여한 프로젝트: {str(row['참여한 프로젝트']) if str(row['참여한 프로젝트']) != '' else '해당 정보 없음'}
    - 최근 3개년간 참여한 프로젝트의 계열사: {str(row['최근 3개년간 참여한 프로젝트 계열사']) if str(row['최근 3개년간 참여한 프로젝트 계열사']) != '' else '해당 정보 없음'}
    - SI(인력) 보유 여부: {
        '보유' if str(row['SI(인력) 기술']) == 'O' else 
        ('미보유' if str(row['SI(인력) 기술']) == 'X' else 
        '해당 정보 없음')
    }
    - SI(인력) 보유시 기술명: {str(row['SI(인력) 기술']) if str(row['SI(인력) 기술']) not in ['O', 'X'] else '해당 정보 없음'}
    - 솔루션 기술 보유 여부: {
        '보유' if str(row['솔루션 기술']) == 'O' else 
        ('미보유' if str(row['솔루션 기술']) == 'X' else 
        '해당 정보 없음')
    }
    - 솔루션 기술 보유시 기술명: {str(row['솔루션 기술']) if str(row['솔루션 기술']) not in ['O', 'X'] else '해당 정보 없음'}
    """

        metadata = {
            "info": f"""
    {str(row['파트너사'])} 정보
            
    - 파트너사(회사명): {str(row['파트너사'])}
    - 주요업종: {str(row['주요업종'])}
    - 업종 구분: {str(row['구분'])}
    - 업종 구분시 대분류: {str(row['대분류'])}
    - 업종 구분시 중분류: {str(row['중분류'])}
    - 업종 구분시 소분류: {str(row['소분류'])}
    - 협약사 여부: {
        '협약사임' if str(row['협약사 여부']) == 'O' else 
        ('협약사 아님' if str(row['협약사 여부']) == 'X' else 
        '해당 정보 없음')
    }
    - 보유 솔루션명: {str(row['솔루션명']) if str(row['솔루션명']) != '' else '해당 정보 없음'}
    - 직원수: {str(row['직원수(명)'])+'명' if str(row['직원수(명)']).isdigit() else '해당 정보 없음'}
    - 2021년 매출액: {str(row['2021년 매출(백만원)'])+'00만원' if str(row['2021년 매출(백만원)']) != '' else '해당 정보 없음'}
    - 2022년 매출액: {str(row['2022년 매출(백만원)'])+'00만원' if str(row['2022년 매출(백만원)']) != '' else '해당 정보 없음'}
    - 2023년 매출액: {str(row['2023년 매출(백만원)'])+'00만원' if str(row['2023년 매출(백만원)']) != '' else '해당 정보 없음'}
    - 사업부 선호도: {str(row['사업부 선호도']) if str(row['사업부 선호도']) != '' else '해당 정보 없음'}
    - 참여한 프로젝트: {str(row['참여한 프로젝트']) if str(row['참여한 프로젝트']) != '' else '해당 정보 없음'}
    - 최근 3개년간 참여한 프로젝트의 계열사: {str(row['최근 3개년간 참여한 프로젝트 계열사']) if str(row['최근 3개년간 참여한 프로젝트 계열사']) != '' else '해당 정보 없음'}
    - SI(인력) 보유 여부: {
        '보유' if str(row['SI(인력) 기술']) == 'O' else 
        ('미보유' if str(row['SI(인력) 기술']) == 'X' else 
        '해당 정보 없음')
    }
    - SI(인력) 보유시 기술명: {str(row['SI(인력) 기술']) if str(row['SI(인력) 기술']) not in ['O', 'X', ''] else '해당 정보 없음'}
    - 솔루션 기술 보유 여부: {
        '보유' if str(row['솔루션 기술']) == 'O' else 
        ('미보유' if str(row['솔루션 기술']) == 'X' else 
        '해당 정보 없음')
    }
    - 솔루션 기술 보유시 기술명: {str(row['솔루션 기술']) if str(row['솔루션 기술']) not in ['O', 'X', ''] else '해당 정보 없음'}
    """
        }
        document = Document(page_content=content, metadata=metadata)  # Document 객체 생성
        documents.append(document)
    
    
    # nltk.download('punkt')  # 필요한 리소스 다운로드
    # nltk.download('punkt_tab')
    nltk.data.find('C:\\Users\\LDCC\\AppData\\Roaming\\nltk_data\\tokenizers\\punkt')

    # 문서 내용 추출 및 토큰화
    tokenized_docs = [word_tokenize(doc.page_content) for doc in documents if isinstance(doc, Document)]  # Document 객체에서 내용 추출
    
    # BM25 초기화, k1과 b 값 조정
    k1 = 1.2  # k1 값 조정
    b = 0.75  # b 값 조정

    # BM25 초기화
    bm25 = BM25Okapi(tokenized_docs, k1=k1, b=b)

    # 사용자 쿼리
    query = user_message
    tokenized_query = query.split(" ")

    # BM25 검색
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_n = bm25_scores.argsort()[-3:][::-1]  # 상위 3개 문서 인덱스

    # BM25 결과
    bm25_results = [documents[i] for i in bm25_top_n]

    # 결과 통합
    final_results = {
        "bm25_results": bm25_results
    }

    formatted_documents = [doc.page_content for doc in bm25_results]
    # 최종 결과 출력
    # print(final_results)
    return formatted_documents