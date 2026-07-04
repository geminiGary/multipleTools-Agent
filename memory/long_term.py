"""【学生任务】长期记忆：跨会话记住用户事实/偏好，落盘到 JSON。

采用最简单的"关键词召回 / 全量注入"策略（进阶可改向量召回，见 README）。
实现完成后把模块级 ENABLED 改为 True，factory 会自动启用它。
"""
import json
import os

from memory.base import LongTermMemory
from llm import LLMClient

# 实现完成后改为 True
ENABLED = True


class FileLongTermMemory(LongTermMemory):
    def __init__(self, user_id: str = "local", path: str = "data/memory_store.json"):
        self.user_id = user_id
        self.path = path
        self._llm = LLMClient()
        self.facts: list[str] = self._load()

    def _read_all(self) -> dict:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except (OSError, ValueError):   
            return {}

    def _load(self) -> list[str]:
        """从 JSON 文件读取本 user 的事实列表（文件不存在时返回 []）。

        TODO: 读取 self.path，结构建议 {user_id: [fact, ...]}，
        返回 data.get(self.user_id, [])。
        """
        data = self._read_all()
        facts = data.get(self.user_id, [])
        if not isinstance(facts, list):
            return []
        return [f.strip() for f in facts if isinstance(f, str) and f.strip()]

    def _save(self) -> None:
        """把 self.facts 写回 JSON 文件（保留其它 user 的数据）。

        TODO: 读出整体 dict，更新 self.user_id 对应项，写回文件。
        """
        dir_name = os.path.dirname(self.path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        data = self._read_all()
        data[self.user_id] = self.facts
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def recall(self, query: str) -> list[str]:
        """召回与 query 相关的事实。

        TODO（关键词策略）：返回 query 与事实有词重叠的项；
        事实较少时可直接全量返回。
        """
        self.facts = self._load()
        if not self.facts:
            return []
        if len(self.facts) <= 20:
            return list(self.facts)

        query_words = {c for c in query.lower() if c.isalnum()}
        scored = []
        for fact in self.facts:
            fact_words = {c for c in fact.lower() if c.isalnum()}
            score = len(query_words & fact_words)
            if score > 0:
                scored.append((score, fact))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [fact for _, fact in scored[:8]]

    def remember(self, user_msg: str, reply: str) -> None:
        """从一轮对话中抽取值得长期记住的事实并落盘。

        TODO:
        1. 用 self._llm.chat() 让模型从对话中抽取"用户的稳定事实/偏好"，
           没有则返回空。
        2. 把新事实去重后加入 self.facts，调用 self._save()。
        """
        try:
            resp = self._llm.chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你负责抽取用户的长期记忆。"
                            "只提取稳定事实或偏好，例如姓名、专业、长期喜好、常用要求。"
                            "不要提取临时问题、一次性任务、寒暄、助手自己的内容。"
                            "只返回 JSON 数组，例如 [\"用户叫小明\"]；没有则返回 []。"
                        ),
                    },
                    {"role": "user", "content": f"用户说: {user_msg}\n助手说: {reply}"},
                ],
            )
        except Exception as e:
            print(f"LLM 调用失败: {e}")
            return
        content = (resp.choices[0].message.content or "").strip()
        if content.startswith("```"):
            lines = content.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = [line.strip("-*• \t") for line in content.splitlines()]

        if isinstance(parsed, str):
            parsed = [parsed]
        if not isinstance(parsed, list):
            return
        
        empty_words = {"", "none", "无", "没有", "无可奉告", "[]", "null", "nothing", "n/a", "不记得了", "不清楚"}
        self.facts = self._load()
        existing = {f.strip() for f in self.facts}
        changed = False

        for item in parsed:
            if not isinstance(item, str):
                continue
            fact = item.strip()
            if not fact or fact.lower() in empty_words:
                continue
            if fact not in existing:
                self.facts.append(fact)
                existing.add(fact)
                changed = True
        if changed:
            self._save()

    def all_facts(self) -> list[str]:
        self.facts = self._load()
        return list(self.facts)
