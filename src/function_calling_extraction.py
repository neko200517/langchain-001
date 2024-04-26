import json
from langchain_openai import ChatOpenAI
from langchain.chains import create_extraction_chain

from dotenv import load_dotenv

load_dotenv()

# Schema
schema = {
  "properties": {
    "person_name": {"type": "string"},
    "person_height": {"type": "integer"},
    "person_hair_color": {"type": "string"},
    "dog_name": {"type": "string"},
    "dog_breed": {"type": "string"},
  },
  "required": ["person_name", "person_height"],
}
inp = """
アレックスの身長は5フィートでブロンドです。彼の犬フロスティはラブラドールで、かくれんぼが好きです。
クローディアはアレックスより1フィート背が高く、彼よりも高くジャンプします。クローディアはブルネットです。
"""

# LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
chain = create_extraction_chain(schema, llm)

result = chain.run(inp)
print(f"""=== 結果 ===
{json.dumps(result, indent=2, ensure_ascii=False)}
===""")