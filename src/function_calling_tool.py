from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9)
tools = load_tools(["terminal"], llm=llm)

agent = initialize_agent(
  tools=tools,
  llm=llm,
  agent=AgentType.OPENAI_FUNCTIONS
)

result = agent.run("srcディレクトリを一覧表示してください。")
print(result)