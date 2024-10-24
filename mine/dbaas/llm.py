## 자연어 처리

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_upstage import UpstageEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_upstage import ChatUpstage
from langchain.utilities import SQLDatabase
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from config import answer_examples

store = {}

# 세션 채팅 이력
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


## text-to-sql
def get_db_table_retriever(input):
    llm = get_llm()
    #llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_KEY)
    db = SQLDatabase.from_uri("mysql+pymysql://root:0000@127.0.0.1:3306/dbaas_be")
    chain = create_sql_query_chain(llm, db)

    question = "다음 질문을 보고 필요한 데이터베이스 테이블을 먼저 추출해줘: " + input
    response = chain.invoke({"question": input }) 
    
    print("################################")
    print(response)

    # "Answer" 부분 추출
    start_index = response.find("Answer:")  # "Answer:"의 시작 인덱스 찾기
    if start_index != -1:
        answer = response[start_index + len("Answer:"):].strip()  # "Answer:" 이후의 내용을 추출하고 공백 제거
    else:
        answer = "No answer found."  # "Answer:"가 없을 경우 기본값

    return answer  # 기본값 설정



## Vector DB로부터 retreiver 수행 
def get_retriever():
    embeddings = UpstageEmbeddings(model='solar-embedding-1-large-query')
    index_name = "dbaas-index"
    # 이미 저장된 데이터를 사용할 때 
    database = Chroma(collection_name='chroma-dbaas', persist_directory="./chroma", embedding_function=embeddings)
    retriever = database.as_retriever(search_kwargs={'k': 3})

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
    llm = ChatUpstage(model="solar-1-mini-chat")
    return llm

# 키워드 사전
def get_dictionary_chain():
    #dictionary = ["비밀번호를 나타내는 표현 -> 패스워드"]
    dictionary = ["가용성을 나타내는 표헌 -> ha_type", "디비를 나타내는 표현 -> db_service"]
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
        "당신은 DBaaS 전문가입니다. 사용자의 질문에 답변해주세요"
        "아래에 제공된 문서를 활용해서 답변해주시고 답변을 알 수 없다면 모른다고 답변해주세요"
        "2-3 문장정도의 짧은 내용의 답변을 원합니다"
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"), #나중에 들어가는 값
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = get_history_retriever()
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick('answer')

    return conversational_rag_chain

 
def get_ai_response(user_message):
    # dictionary_chain = get_dictionary_chain()
    # rag_chain = get_rag_chain()
    # security_chain = {"input": dictionary_chain} | rag_chain
    
    # ai_response = security_chain.stream(
    #     {
    #         "question": user_message
    #     },
    #     config={
    #         "configurable": {"session_id": "abc123"}
    #     },
    # )
    
    ai_response = get_db_table_retriever(user_message)

    return ai_response