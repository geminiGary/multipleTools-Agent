"""集中读取环境变量。其它模块统一从这里取配置。"""
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# 长期记忆的作用主体：所有对话窗口共享一份「用户记忆」，按此 key 存储（与每个窗口的 session_id 分离）
USER_ID = os.getenv("USER_ID", "local")
