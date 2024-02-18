from langchain.globals import set_verbose
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory import ChatMessageHistory
from langchain.schema import HumanMessage
from langchain.chains import ConversationChain

set_verbose(True)

def chat(message: str, history: ChatMessageHistory) -> str:
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    convesation = ConversationChain(llm=llm, memory=ConversationBufferMemory())

    messages = history.messages
    messages.append(HumanMessage(content=message))

    return convesation.predict(input=messages)
