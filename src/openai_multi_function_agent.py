import langchain
import openai
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

langchain.debug = True
langchain.verbose = True
openai.log = "info"

tools = load_tools(["ddg-search"])
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_MULTI_FUNCTIONS)

result = agent.run(
  "猫宮ひなたの前世や年齢について教えてください。"
)

print(f"""=== 結果 ===
{result}
""")