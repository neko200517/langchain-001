import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv
from chatbot_engine import chat, create_index
from langchain.memory import ChatMessageHistory

load_dotenv()

index = create_index()

# ボットトークンとソケットモードハンドラーを使ってアプリを初期化します
app = App(token=os.environ.get("SLACK_BOT_TOKEN"))


def fetch_history(channel: str) -> ChatMessageHistory:
    # AIのユーザーIDを取得
    bot_user_id = app.client.auth_test()["user_id"]

    conversations_history = app.client.conversations_history(channel=channel, limit=3)

    history = ChatMessageHistory()

    # 会話の流れを正確にするためにreversedで配列を逆にする
    for messages in reversed(conversations_history["messages"]):
        text = messages["text"]

        if messages["user"] == bot_user_id:
            history.add_ai_message(text)
        else:
            history.add_user_message(text)
        
    return history


@app.event("app_mention")
def handle_mention(event, say):
    channel = event["channel"]
    history = fetch_history(channel)

    message = event["text"]
    bot_message = chat(message, history, index)
    say(bot_message)


# アプリを起動します
if __name__ == "__main__":
    app_env = os.environ.get("APP_ENV", "production")

    if app_env == "production":
        # productionの場合、待ち受けモードで起動
        app.start(port=int(os.environ.get("PORT", 3000)))
    else:
        # production以外の場合、SocketModeで起動
        SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()


