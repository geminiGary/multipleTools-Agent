# 智能多工具助手（作业项目）

一个基于大模型 API 的多工具助手智能体。老师提供可运行的引擎与两个完整范例，
你需要**模仿范例**补全其它能力模块。

> 第一次上手，先看带思路引导和自检步骤的教程 [`docs/任务指导.md`](docs/任务指导.md)。本 README 作为速查手册。

## 快速开始

### 环境要求

- Python 3.10+
- 任意 OpenAI 兼容的大模型服务（key / base_url / 模型名）

### 安装与配置

```bash
pip install -r requirements.txt
cp .env.example .env   # 填入 OpenAI 兼容服务的 key / base_url / 模型名
```

### 两种运行方式

**命令行对话**（最快验证，单条会话、历史不落盘）：

```bash
python main.py
```

**后端服务 + 网页界面**（推荐，支持多会话与历史持久化）：

```bash
uvicorn server:app --reload
```

启动后浏览器打开 **http://localhost:8000**（`server.py` 已托管 `frontend/index.html`）。界面分三栏：

- **左侧会话栏**：新建 / 切换 / 重命名 / 删除多个对话。每条对话的历史会自动落盘到 `data/conversations/`，**重启服务后仍在**。
- **中间对话区**：与助手流式对话。
- **右侧仪表栏**：实时显示助手调用了哪个工具、引用了哪些来源、跨对话记住了你什么，也能看到已启用哪些能力模块。

> 也可直接双击打开 `frontend/index.html`，在界面右上角把「后端」地址指向你的服务即可（后端已开启 CORS）。

未填 key 也能跑测试：`pytest -v`

## 你会学到

- 工具调用（function calling）
- 短期记忆与长期记忆
- 检索增强生成（RAG，进阶可选）

## 架构

- `llm.py` 模型客户端 / `agent.py` 主循环（事件流）/ `server.py` SSE 接口与会话管理 —— 已提供，无需改动。
- `tools/` 工具（`Tool` 接口）/ `rag/` 检索 / `memory/` 记忆。
- **对话持久化与多会话**：`memory/conversation_store.py` 把每条对话落盘为 `data/conversations/<id>.json`，服务端提供 `GET /api/conversations`、`GET / PATCH / DELETE /api/conversations/{id}` 等接口。这部分是已做好的基础设施，**你只需专注实现长期记忆**。
- **优雅降级**：未实现的模块会被自动隐藏，整体仍能正常对话。运行时可通过 `GET /api/capabilities` 或 CLI 启动行查看已启用能力。

## 范例

- `tools/weather.py`、`tools/calculator.py`：一个工具从定义到被调用的完整流程。
- `tests/test_weather.py` 是工具测试的范例。
- `memory/short_term.py`：一个记忆模块的完整范例。

## 你的任务（两个主任务，互不阻塞，做完即见效）

| # | 文件 | 做什么 | 启用方式 |
|---|---|---|---|
| 1 | `tools/web_search.py` | 实现 `run()` | 把 `is_available()` 改为 `True` |
| 2 | `memory/long_term.py` | 实现长期记忆 | 把模块里的 `ENABLED` 改为 `True` |

> RAG（本地知识库检索）作为进阶可选，见下方「进阶」。

每实现一个模块，照着 `tests/test_weather.py` / `tests/test_short_term.py` 为它补测试。

> 小贴士：
> - 骨架文件里有一些「未使用的 import」（如 `numpy`、`STORE`、`json`）是**故意预置**的，你填完 TODO 就会用到，不是框架问题。
> - **长期记忆是全局的**：所有对话窗口共享一份「用户记忆」，按固定的 `user_id` 落盘到 `data/memory_store.json`（构造参数已是 `user_id`，与每个窗口的 `session_id` 分开）。所以在一个窗口告诉助手「我叫小明」，换个窗口它也能想起来——这正是任务 2 做完后的效果。

## 评分

总分 100 分。每项「怎么算拿到分」的判定细则见 [`docs/任务指导.md`](docs/任务指导.md) 第 9 节。

| 模块 | 分值 |
|---|---|
| 任务 1 · 工具调用 `web_search` | 30 |
| 任务 2 · 长期记忆 | 50 |
| 进阶 · RAG（本地知识库检索） | 10 |
| 其余各类优化 | 10 |

提交两部分：**代码源码**（不含模型，如 RAG 的 embedding 权重）和**大作业报告**（含各任务执行截图，做了优化还要有优化介绍）。详见任务指导第 10 节。

## 进阶

- **RAG（本地知识库检索）**：完整的进阶任务。仓库保留了 `rag/` 与 `tools/knowledge_base.py` 骨架、`data/docs/` 语料、[`docs/rag-verification.md`](docs/rag-verification.md) 验收清单；实现思路留给你自行探索。
- 长期记忆改用向量召回。
- RAG 做完后可再加重排序（rerank）或多路召回。
- 工具调用失败自动重试；ReAct 反思自纠。
- 真实网络搜索 API 接入。
