from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# PromptTemplate
prompt = PromptTemplate.from_template("""料理のレシピを教えてください。

料理名: {dish}""")

# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Chain
chain = prompt | llm

# Result
result = chain.invoke({"dish": "カレー"})
print(result.content)