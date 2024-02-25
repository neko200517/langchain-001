import gradio as gr
from dotenv import load_dotenv
from chatbot_engine import chat, create_index
from langchain.memory import ChatMessageHistory

import os

def ai_response(message, chat_history):
    history = ChatMessageHistory()
    for [user_message, ai_message] in chat_history:
       history.add_user_message(user_message)
       history.add_ai_message(ai_message)

    bot_message = chat(message, history, index)
    
    return bot_message

demo = gr.ChatInterface(ai_response)

if __name__ == "__main__":
  load_dotenv()

  app_env = os.environ.get("APP_ENV", "production")

  # 開発環境がproductionの場合のみ認証がかかる
  if app_env == "production":
     username = os.environ["GRADIO_USERNAME"]
     password = os.environ["GRADIO_PASSWORD"]
     auth = (username, password)
  else:
     auth = None

  index = create_index()

  demo.launch(auth=auth)