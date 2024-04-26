# LangChain を使ったサンプルアプリケーション

## ドキュメント

https://github.com/neko200517/obsidian/tree/main/Notes/develop/LangChanin%E3%81%AB%E3%82%88%E3%82%8B%E5%A4%A7%E8%A6%8F%E6%A8%A1%E8%A8%80%E8%AA%9E%E3%83%A2%E3%83%87%E3%83%AB%EF%BC%88LLM%EF%BC%89%E3%82%A2%E3%83%97%E3%83%AA%E3%82%B1%E3%83%BC%E3%82%B7%E3%83%A7%E3%83%B3%E3%81%AE%E9%96%8B%E7%99%BA

## 実行

```bash
poetry run python src/gradio_app.py
```

## ホットリロード

```bash
poetry run gradio src/gradio_app.py
```

# ローカル開発環境

## Ubuntuにログインし、以下のインストールを行う

https://asdf-vm.com/guide/getting-started.html

### 必要なパッケージをインストール

```bash
sudo apt-get update && sudo apt dist-upgrade -y && sudo apt autoremove -y
sudo apt install -y curl git
```

### asdfをインストール 

```bash
git clone https://github.com/asdf-vm/asdf.git ~/.asdf --branch v0.14.0
echo -e "\n. $HOME/.asdf/asdf.sh" >> ~/.bash_profile
echo -e '\n. $HOME/.asdf/completions/asdf.bash' >> ~/.bash_profile
source ~/.bashrc
```

## Ubuntuに依存関係をインストール 

https://github.com/pyenv/pyenv/wiki#suggested-build-environment

```bash
sudo apt update; sudo apt install build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

## asdfにpythonプラグインを追加する

```bash
asdf plugin-add python
```

### インストールできるpythonの確認

```bash
asdf list all python
```

## PythonとPoetryをインストールする

```bash
cd {インストールしたいプロジェクトの場所}
echo "python 3.11.3" >> .tool-versions
echo "poetry 1.1.14" >> .tool-versions
asdf install
```

### .tool-versions 

```
python 3.11.3
poetry 1.1.14
```

## 確認 

```bash
python --version
poetry --version
```
