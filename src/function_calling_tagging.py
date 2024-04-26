import json
from langchain.chains import create_tagging_chain
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

# Schema
schema = {
  "properties": {
    "sentiment": {"type": "string"},
    "aggressiveness": {"type": "integer"},
    "language": {"type": "string"},
  }
}

# LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
chain = create_tagging_chain(schema, llm)

inp = "私はあなたにとても腹を立てています！"
result = chain.run(inp)

print(f"""=== 結果 ===
{json.dumps(result, indent=2, ensure_ascii=False)}
===""")