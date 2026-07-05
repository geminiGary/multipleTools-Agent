import json

from memory.long_term import FileLongTermMemory
from tests.fakes import FakeLLM


class TopicEmbedder:
    def embed(self, texts):
        vectors = []
        for text in texts:
            vectors.append([
                1.0 if "__occupation__" in text else 0.0,
                1.0 if "__name__" in text else 0.0,
                1.0 if "__preference__" in text else 0.0,
                1.0 if "__age__" in text else 0.0,
            ])
        return vectors


def test_load_missing_file_starts_empty(tmp_path):
    m = FileLongTermMemory(user_id="u1", path=str(tmp_path / "missing.json"))
    assert m.all_facts() == []


def test_save_preserves_other_users(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text(json.dumps({"other": ["其他用户事实"]}, ensure_ascii=False), encoding="utf-8")

    m = FileLongTermMemory(user_id="u1", path=str(path))
    m.facts = ["用户叫小明"]
    m._save()

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["other"] == ["其他用户事实"]
    assert data["u1"] == ["用户叫小明"]


def test_remember_extracts_and_persists_fact(tmp_path):
    path = tmp_path / "memory.json"
    m = FileLongTermMemory(user_id="u1", path=str(path))
    # FakeLLM.chat 不接受 max_tokens；这个测试可防止 remember 再传入多余参数。
    m._llm = FakeLLM(summary='["用户叫小明"]')

    m.remember("我叫小明", "好的，我记住了。")

    assert m.all_facts() == ["用户叫小明"]
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["u1"] == ["用户叫小明"]


def test_recall_refreshes_facts_saved_by_another_instance(tmp_path):
    path = tmp_path / "memory.json"
    first = FileLongTermMemory(user_id="u1", path=str(path))
    second = FileLongTermMemory(user_id="u1", path=str(path))

    first.facts = ["用户叫小明"]
    first._save()

    assert "用户叫小明" in second.recall("我叫什么")


def test_recall_many_facts_uses_keyword_overlap(tmp_path):
    path = tmp_path / "memory.json"
    m = FileLongTermMemory(user_id="u1", path=str(path))
    m.facts = [f"事实{i}" for i in range(25)] + ["用户喜欢Python"]
    m._save()

    assert "用户喜欢Python" in m.recall("Python怎么写")


def test_recall_many_facts_uses_vector_semantics(tmp_path):
    path = tmp_path / "memory.json"
    m = FileLongTermMemory(user_id="u1", path=str(path))
    m._llm = TopicEmbedder()
    m.facts = [f"用户的偏好{i}是示例{i}" for i in range(25)] + ["用户是 Python 后端工程师"]
    m._save()

    recalled = m.recall("我的职业是什么")

    assert recalled[0] == "用户是 Python 后端工程师"


def test_forget_request_removes_matching_fact(tmp_path):
    path = tmp_path / "memory.json"
    m = FileLongTermMemory(user_id="u1", path=str(path))
    m.facts = ["用户喜欢打篮球", "用户常用中文回答"]
    m._save()

    m.remember("请忘记我喜欢打篮球", "好的，已忘记。")

    assert m.all_facts() == ["用户常用中文回答"]


def test_update_replaces_conflicting_occupation(tmp_path):
    path = tmp_path / "memory.json"
    m = FileLongTermMemory(user_id="u1", path=str(path))
    m.facts = ["用户是 Python 后端工程师"]
    m._save()
    m._llm = FakeLLM(summary='["用户是 Go 后端工程师"]')

    m.remember("我现在是 Go 后端工程师", "好的，我记住了。")

    assert m.all_facts() == ["用户是 Go 后端工程师"]


def test_update_preference_removes_old_preference_when_user_says_no_longer(tmp_path):
    path = tmp_path / "memory.json"
    m = FileLongTermMemory(user_id="u1", path=str(path))
    m.facts = ["用户喜欢打篮球"]
    m._save()
    m._llm = FakeLLM(summary='["用户喜欢游泳"]')

    m.remember("我现在喜欢游泳，不再喜欢打篮球", "好的，我记住了。")

    assert m.all_facts() == ["用户喜欢游泳"]


def test_update_age_replaces_old_age_fact(tmp_path):
    path = tmp_path / "memory.json"
    m = FileLongTermMemory(user_id="u1", path=str(path))
    m.facts = ["用户今年20岁"]
    m._save()
    m._llm = FakeLLM(summary='["用户年龄改为21岁"]')

    m.remember("更改我的年龄为21岁", "好的，我记住了。")

    assert m.all_facts() == ["用户年龄是21岁"]


def test_forget_age_removes_age_fact_by_topic(tmp_path):
    path = tmp_path / "memory.json"
    m = FileLongTermMemory(user_id="u1", path=str(path))
    m.facts = ["用户今年20岁", "用户喜欢打篮球"]
    m._save()

    m.remember("请忘记我的年龄", "好的，已忘记。")

    assert m.all_facts() == ["用户喜欢打篮球"]
