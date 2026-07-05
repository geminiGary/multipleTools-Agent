"""【学生任务】长期记忆：跨会话记住用户事实/偏好，落盘到 JSON。

采用"向量召回 + 关键词兜底"策略；同时支持用户主动更新/删除记忆。
实现完成后把模块级 ENABLED 改为 True，factory 会自动启用它。
"""
import json
import math
import os
import re

from memory.base import LongTermMemory
from llm import LLMClient

# 实现完成后改为 True
ENABLED = True
RECALL_TOP_K = 8
MIN_RECALL_SCORE = 0.08

DELETE_MARKERS = ("忘记", "删除", "清除", "别记", "不要记", "不再记", "取消记忆")
UPDATE_MARKERS = ("现在", "改成", "换成", "更新", "不再", "以后", "更改", "修改", "变成", "变为", "变了")
BROAD_TOPIC_MARKERS = ("名字", "姓名", "叫什么", "年龄", "几岁", "多少岁", "职业", "工作", "岗位", "专业", "爱好", "兴趣", "偏好")
STOP_CHARS = set("用户我的你他她它们请帮把将关于记住记得忘记删除清除不要再已经还是什么多少哪些一下一个一些以及和与的是了的吗呢吧啊呀")

TOPIC_MARKERS = {
    "__name__": ("名字", "姓名", "叫什么", "叫", "称呼", "name"),
    "__age__": ("年龄", "几岁", "多少岁", "岁"),
    "__occupation__": (
        "职业",
        "工作",
        "岗位",
        "专业",
        "工程师",
        "程序员",
        "开发",
        "后端",
        "前端",
        "产品经理",
        "学生",
        "老师",
        "教师",
        "医生",
        "律师",
        "python",
        "java",
        "go",
        "c++",
    ),
    "__preference__": ("喜欢", "不喜欢", "讨厌", "爱好", "兴趣", "偏好", "习惯", "常用"),
    "__response_language__": ("中文", "英文", "英语", "语言", "回复", "回答"),
    "__location__": ("城市", "住在", "来自", "所在地", "家在"),
    "__constraint__": ("过敏", "不能吃", "忌口", "限制", "要求"),
}


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

        优先使用 embedding 做语义召回；当 embedding 不可用或分数过低时，
        再用关键词重叠兜底，避免"职业"与"工程师"这类字面不一致的问题。
        """
        self.facts = self._load()
        if not self.facts:
            return []

        recalled = self._vector_recall(query)
        if recalled:
            return recalled
        return self._keyword_recall(query)

    def remember(self, user_msg: str, reply: str) -> None:
        """从一轮对话中抽取值得长期记住的事实并落盘。

        TODO:
        1. 用 self._llm.chat() 让模型从对话中抽取"用户的稳定事实/偏好"，
           没有则返回空。
        2. 把新事实去重后加入 self.facts，调用 self._save()。
        """
        if self._handle_forget_request(user_msg):
            return

        try:
            resp = self._llm.chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你负责抽取用户的长期记忆。"
                            "只根据用户原话提取稳定事实或偏好，例如姓名、专业、长期喜好、常用要求。"
                            "不要提取临时问题、一次性任务、寒暄、用户问过什么、比较过什么、助手自己的内容或助手推断。"
                            "如果用户只是提问，例如询问产品创始人、价格、对比、搜索资料，不要记录。"
                            "只返回 JSON 数组，例如 [\"用户叫小明\"]；没有则返回 []。"
                        ),
                    },
                    {"role": "user", "content": f"用户说: {user_msg}"},
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

        new_facts = self._clean_facts(parsed)
        if not new_facts:
            return

        self.facts = self._load()
        existing = {f.strip() for f in self.facts}
        changed = False

        for fact in new_facts:
            if self._remove_conflicts_for_new_fact(fact, user_msg):
                existing = {f.strip() for f in self.facts}
                changed = True
            if fact not in existing:
                self.facts.append(fact)
                existing.add(fact)
                changed = True
        if changed:
            self._save()

    def all_facts(self) -> list[str]:
        self.facts = self._load()
        return list(self.facts)

    def _clean_facts(self, items: list) -> list[str]:
        empty_words = {"", "none", "无", "没有", "无可奉告", "[]", "null", "nothing", "n/a", "不记得了", "不清楚"}
        cleaned = []
        for item in items:
            if not isinstance(item, str):
                continue
            fact = item.strip()
            if not fact or fact.lower() in empty_words:
                continue
            if any(bad in fact for bad in ("用户询问", "用户问过", "用户比较", "助手", "知简笔记的创始人")):
                continue
            cleaned.append(self._normalize_fact(fact))
        return cleaned

    def _normalize_fact(self, fact: str) -> str:
        compact = re.sub(r"\s+", "", fact)
        age_match = re.search(r"(?:年龄|今年)?(?:改为|改成|变为|变成|更改为|修改为|是|为)?(\d{1,3})岁", compact)
        if age_match and ("岁" in compact or "年龄" in compact):
            return f"用户年龄是{age_match.group(1)}岁"
        return fact

    def _vector_recall(self, query: str) -> list[str]:
        try:
            texts = [self._embedding_text(query)] + [self._embedding_text(fact) for fact in self.facts]
            embeddings = self._llm.embed(texts)
        except Exception as e:  # noqa: BLE001
            print(f"记忆向量召回失败，改用关键词召回: {e}")
            return []

        if len(embeddings) != len(self.facts) + 1:
            return []

        query_vec = embeddings[0]
        scored = []
        for fact, fact_vec in zip(self.facts, embeddings[1:]):
            score = self._cosine(query_vec, fact_vec)
            score += min(0.25, self._keyword_score(query, fact) * 0.06)
            if score >= MIN_RECALL_SCORE:
                scored.append((score, fact))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [fact for _, fact in scored[:RECALL_TOP_K]]

    def _keyword_recall(self, query: str) -> list[str]:
        scored = []
        for fact in self.facts:
            score = self._keyword_score(query, fact)
            if score > 0:
                scored.append((score, fact))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [fact for _, fact in scored[:RECALL_TOP_K]]

    def _embedding_text(self, text: str) -> str:
        tags = self._topic_tags(text)
        if not tags:
            return text
        return text + "\n" + " ".join(sorted(tags))

    def _topic_tags(self, text: str) -> set[str]:
        lowered = text.lower()
        tags = set()
        for topic, markers in TOPIC_MARKERS.items():
            if any(marker.lower() in lowered for marker in markers):
                tags.add(topic)
        return tags

    def _primary_topic(self, text: str) -> str | None:
        tags = self._topic_tags(text)
        for topic in ("__name__", "__age__", "__occupation__", "__response_language__", "__location__", "__constraint__", "__preference__"):
            if topic in tags:
                return topic
        return None

    def _cosine(self, left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)

    def _keyword_score(self, query: str, fact: str) -> float:
        query_tokens = self._tokens(query)
        fact_tokens = self._tokens(fact)
        if not query_tokens or not fact_tokens:
            return 0.0
        overlap = query_tokens & fact_tokens
        return len(overlap) / max(len(query_tokens), 1)

    def _keyword_overlap_count(self, query: str, fact: str) -> int:
        return len(self._tokens(query) & self._tokens(fact))

    def _tokens(self, text: str) -> set[str]:
        lowered = text.lower()
        tokens = set(re.findall(r"[a-z0-9+#]+", lowered))
        for ch in lowered:
            if ch.isalnum() and not ch.isascii() and ch not in STOP_CHARS:
                tokens.add(ch)
        return tokens

    def _handle_forget_request(self, user_msg: str) -> bool:
        if not any(marker in user_msg for marker in DELETE_MARKERS):
            return False

        self.facts = self._load()
        kept = []
        removed = []
        for fact in self.facts:
            if self._should_forget_fact(user_msg, fact):
                removed.append(fact)
            else:
                kept.append(fact)

        if removed:
            self.facts = kept
            self._save()
        return True

    def _should_forget_fact(self, user_msg: str, fact: str) -> bool:
        requested_topics = self._requested_topics(user_msg)
        fact_topics = self._topic_tags(fact)
        broad_request = any(marker in user_msg for marker in BROAD_TOPIC_MARKERS)
        if broad_request and requested_topics and requested_topics & fact_topics:
            return True
        return self._keyword_overlap_count(user_msg, fact) >= 2

    def _requested_topics(self, text: str) -> set[str]:
        return self._topic_tags(text)

    def _remove_conflicts_for_new_fact(self, fact: str, user_msg: str) -> bool:
        topic = self._primary_topic(fact)
        if not topic:
            return False

        replace_topics = {"__name__", "__age__", "__occupation__", "__response_language__", "__location__"}
        should_replace_topic = topic in replace_topics
        preference_update = topic == "__preference__" and any(marker in user_msg for marker in UPDATE_MARKERS)

        kept = []
        removed = []
        for old_fact in self.facts:
            if old_fact == fact:
                kept.append(old_fact)
                continue
            old_topic = self._primary_topic(old_fact)
            if should_replace_topic and old_topic == topic:
                removed.append(old_fact)
            elif preference_update and old_topic == topic and self._keyword_overlap_count(user_msg, old_fact) >= 2:
                removed.append(old_fact)
            else:
                kept.append(old_fact)

        if removed:
            self.facts = kept
            return True
        return False
