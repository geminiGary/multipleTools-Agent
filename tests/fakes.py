"""测试用替身：不联网即可测试 Agent / 短期记忆等逻辑。"""
from tools.base import Tool


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class FakeLLM:
    """可编排的假 LLM。

    turns: 按顺序消费的回合列表，每个回合是：
        {"content": "文本"}                              -> 流式回复该文本，无工具调用
        {"tool_calls": [{"id","name","arguments"}]}       -> 请求工具调用
    summary: chat() 的固定返回（短期记忆摘要会用到）。
    """

    def __init__(self, turns=None, summary="摘要"):
        self.turns = list(turns or [])
        self._summary = summary
        self.seen_messages = []  # 记录每次 stream_turn 收到的 messages，供测试断言

    def chat(self, messages, tools=None, tool_choice="auto"):
        return _Resp(self._summary)

    def stream_turn(self, messages, tools=None):
        self.seen_messages.append(list(messages))
        turn = self.turns.pop(0) if self.turns else {"content": ""}
        for ch in (turn.get("content") or ""):
            yield ch
        return turn.get("tool_calls", [])

    def embed(self, texts):
        return [[0.0, 0.0, 0.0, 0.0] for _ in texts]


class EchoTool(Tool):
    name = "echo"
    description = "回显输入文本"
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def run(self, text: str) -> str:
        return f"echo: {text}"
