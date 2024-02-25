from langchain.globals import set_verbose
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ChatMessageHistory
from langchain_community.document_loaders import DirectoryLoader, PythonLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain.chains import ConversationChain
from langchain.vectorstores import Chroma

from langchain.agents import Tool
from langchain.agents import initialize_agent

from langchain.chains import ConversationalRetrievalChain

from langchain.prompts import ChatPromptTemplate

from langchain.schema.runnable import RunnablePassthrough

from langchain.agents import AgentType
from langchain.agents import tool


set_verbose(True)

def create_index() -> VectorStoreIndexWrapper:
    loader = DirectoryLoader("./src/", glob="**/*.py", loader_cls=PythonLoader)
    return VectorstoreIndexCreator(embedding=OpenAIEmbeddings()).from_loaders([loader])


def create_tools(llm, index: VectorStoreIndexWrapper):
    return [
        Tool(
            name = "udemy-langchain source code",
            func=lambda query: str(index.query(query, llm)),
            description="Source code of application named udemy-langchain",
            return_direct=True
        ),
    ]


def chat(message: str, history: ChatMessageHistory, index: VectorStoreIndexWrapper) -> str:
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    tools = create_tools(llm, index)

    memory = ConversationBufferMemory(
        chat_memory=history, memory_key="chat_history", return_messages=True
    )

    agent_chain = initialize_agent(
        tools=tools, 
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
        verbose=True, 
        memory=memory
    )

    return agent_chain.run(input=message)


def old_chat(message: str, history: ChatMessageHistory, index: VectorStoreIndexWrapper) -> str:
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # prompt = ChatPromptTemplate.from_template(
    #     "以下のcontextだけに基づいて回答してください。\n{context}\n質問: {question}"
    # )

    # retriever = index.vectorstore.as_retriever()

    # chain = (
    #   {"context": retriever, "question": RunnablePassthrough()}
    #   | prompt
    #   | llm
    # )

    # result = chain.invoke(message)

    # return result.content


    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # chat = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

    # result = chat.run({"question": message, "chat_history": history})

    # return result

    tools = [
        Tool(
            name = "udemy-langchain source code",
            func=lambda q: str(index.query(question=q, llm=llm)),
            description="Source code of application named udemy-langchain",
            return_direct=True
        ),
    ]

    memory = ConversationBufferMemory(
        chat_memory=history, memory_key="chat_history", return_messages=True
    )

    agent_chain = initialize_agent(
        tools=tools, 
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
        verbose=True, 
        memory=memory
    )

    return agent_chain.run(input=message)

    # convesation = ConversationChain(llm=llm, memory=ConversationBufferMemory())

    # messages = history.messages
    # messages.append(HumanMessage(content=message))
    # ai_message = index.query(question=message, llm=llm)
    # messages.append(AIMessage(ai_message))

