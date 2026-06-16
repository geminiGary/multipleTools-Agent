"""【学生任务】长期记忆：跨会话记住用户事实/偏好，落盘到 JSON。

采用最简单的"关键词召回 / 全量注入"策略（进阶可改向量召回，见 README）。
实现完成后把模块级 ENABLED 改为 True，factory 会自动启用它。
"""
import json
import os

from memory.base import LongTermMemory
from llm import LLMClient

# 实现完成后改为 True
ENABLED = False


class FileLongTermMemory(LongTermMemory):
    def __init__(self, user_id: str = "local", path: str = "data/memory_store.json"):
        self.user_id = user_id
        self.path = path
        self._llm = LLMClient()
        self.facts: list[str] = self._load()

    def _load(self) -> list[str]:
        """从 JSON 文件读取本 user 的事实列表（文件不存在时返回 []）。

        TODO: 读取 self.path，结构建议 {user_id: [fact, ...]}，
        返回 data.get(self.user_id, [])。
        """
        raise NotImplementedError("TODO: 实现读取持久化")

    def _save(self) -> None:
        """把 self.facts 写回 JSON 文件（保留其它 user 的数据）。

        TODO: 读出整体 dict，更新 self.user_id 对应项，写回文件。
        """
        raise NotImplementedError("TODO: 实现写入持久化")

    def recall(self, query: str) -> list[str]:
        """召回与 query 相关的事实。

        TODO（关键词策略）：返回 query 与事实有词重叠的项；
        事实较少时可直接全量返回。
        """
        raise NotImplementedError("TODO: 实现召回")

    def remember(self, user_msg: str, reply: str) -> None:
        """从一轮对话中抽取值得长期记住的事实并落盘。

        TODO:
        1. 用 self._llm.chat() 让模型从对话中抽取"用户的稳定事实/偏好"，
           没有则返回空。
        2. 把新事实去重后加入 self.facts，调用 self._save()。
        """
        raise NotImplementedError("TODO: 实现事实抽取与落盘")

    def all_facts(self) -> list[str]:
        return list(self.facts)
