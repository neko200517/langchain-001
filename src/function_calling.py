import json

from langchain.agents import AgentType, initialize_agent, Tool
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

def get_current_weatcher(location, unit = "fahrenheit"):
  """Get the current weather in a given location"""
  weather_info = {
    "location": location,
    "temperature": "72",
    "unit": unit,
    "forecast": ["sunny", "windty"],
  }
  return json.dumps(weather_info)

tools = [
  Tool(
    name = "get_current_weatcher",
    func=get_current_weatcher,
    description="Get the current weather in a given location",
  )
]

chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)\

agent = initialize_agent(
  tools=tools,
  llm=chat,
  agent=AgentType.OPENAI_FUNCTIONS
)

result = agent.run("ボストンの天気を教えてください。")
print(result)