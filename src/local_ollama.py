from langchain.tools.retriever import create_retriever_tool
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from seed_data import seed_milvus, connect_to_milvus
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


def get_retriever(collection_name: str = "data_test") -> EnsembleRetriever:
    try:
        vectorstore = connect_to_milvus(
            'http://localhost:19530', collection_name)
        milvus_retriever = vectorstore.as_retriever(
            search_kwargs={"k": 4}, search_type="similarity")

        documents = [
            Document(page_content=doc.page_content, metadata=doc.metadata)
            for doc in vectorstore.similarity_search("", k=100)
        ]

        if not documents:
            raise ValueError(
                f"No documents found in collection '{collection_name}.'")

        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 4

        ensemble_retriever = EnsembleRetriever(
            retrievers=[milvus_retriever, bm25_retriever], weights=[0.7, 0.3])
        return ensemble_retriever
    except Exception as e:
        print(f"Error in get_retriever: {e}")
        default_doc = [
            Document(page_content="An error occurred while connecting to the database.", metadata={
                     "source": "error"})

        ]
        return BM25Retriever.from_documents(default_doc)


def get_llm_agent(retriever):
    tool = create_retriever_tool(
        retriever,
        "find_documents",
        "Searches and returns documents regarding the question.",
    )

    llm = ChatOllama(
        model="llama2:7b-chat",
        temperature=0,
        streaming=True)
    tools = [tool]

    # Prompt Template
    system = """You are CTU Bot, an AI assistant specialized in answering questions about the regulations, 
    policies, and general information of Can Tho University. Your task is to provide clear, 
    concise, and accurate answers to help students with their inquiries. Feel free to ask anything.\n\n"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Create agent
    agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)


retriever = get_retriever()
agent_executor = get_llm_agent(retriever)
