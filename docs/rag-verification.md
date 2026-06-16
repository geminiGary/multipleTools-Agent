# RAG 实现与验证指南

> 这份文档分两部分：
> 前半（一～四节）教你怎么实现 RAG：思路、三个文件怎么协作、关键算法的伪代码、容易踩的坑；
> 后半（五～六节）是一份验收清单，实现完照着逐条核对，确认检索真的生效了。
>
> 配套阅读：整体上手看 `docs/任务指导.md`，原理看 `docs/背景知识.md`。RAG 本质上就是第 1 节讲的那种「工具」——一个去查本地资料的工具；第 2.5 节也点过，长期记忆的「召回」其实就是一次小型 RAG。这两段先扫一眼，下面会顺很多。

---

## 一、RAG 是什么，为什么需要它

回到 `背景知识.md` 第 1.1 节那句话：大模型只会**续写文字**。它没学过的东西——你的私有文档、公司内部资料、训练截止之后的新知识——硬问它，它只会一本正经地编（这就是幻觉）。

RAG（检索增强生成）就是来补这个缺口的：先准备一个可检索的资料库，每次用户提问时，先从库里把最相关的几段查出来、塞进上下文，再让模型照着这些资料回答。这样模型就从「凭记忆瞎猜」变成「照着手头资料说话」，答得准，也能交代出处。

<details>
<summary>📖 概念：为什么不直接把所有文档塞进 prompt？</summary>
两个原因，都在 `背景知识.md` 第 2.2 节讲过：上下文窗口装不下（文档可能很长），而且塞得越多越慢越贵。所以不能全塞，只能按这一问挑最相关的几段。怎么挑，是 RAG 真正在做的事：靠的不是关键词匹配，而是把文字转成向量（embedding），按语义接近程度来找。这样「价格」和「多少钱」「收费」字面不同，也能凭意思接近被检索到。
</details>

这个项目里，知识库语料刻意限定成一个虚构产品「知简笔记」（`data/docs/知简-*.md`）。模型预训练时绝不可能见过它，所以域内问题只要答对了，就说明检索确实生效了。这让「RAG 有没有跑通」一眼可辨，也是后半那份验收清单的设计思路。

---

## 二、三个文件怎么协作

RAG 拆在三个文件里，不少人卡住，是因为没先理清它们的关系。其实就一句话：两个阶段，共用中间一个库。

```
入库阶段（服务启动时，自动跑一次）
  data/docs/知简-*.md
        │  ① 读取文本
        ▼
  rag/ingest.py
    chunk_text() 切成小块  →  _llm.embed() 把每块转成向量
        │  ② STORE.add(块文本, 块向量, 来源)
        ▼
  ┌──────────────────────────────────────┐
  │  STORE  ← 模块级共享实例（rag/vector_store.py） │
  │  texts[] / embeddings[] / metadatas[]          │
  └──────────────────────────────────────┘
        ▲  ③ STORE.search(问题向量, top_k)
        │
检索阶段（每次用户提问时跑）
  tools/knowledge_base.py
    _llm.embed([query]) 把问题转成向量  →  STORE.search 取最相似的几块
        │  ④ 拼成一段文字返回给模型；把来源写进 last_sources
        ▼
  模型据此组织回答  +  右侧仪表栏显示「引用来源」
```

对着图记住几点：

- `STORE` 是中枢，而且自始至终是同一个。`vector_store.py` 顶部 `STORE = VectorStore()` 建好的这个共享实例，入库的 `ingest.py` 往里写、检索的 `knowledge_base.py` 从里读，访问的必须是同一个对象。这是最容易翻车的地方（见第四节）。
- 入库只在启动时跑一次，检索每次提问都跑。所以你改完代码得重启服务，库里才会有数据。
- 三个文件各管一摊：`vector_store.py` 是地基（存 + 算相似度），`ingest.py` 负责把文档喂进去，`knowledge_base.py` 把检索包装成一个模型能调的工具。建议就按这个顺序做。

---

## 三、动手顺序与算法指导

下面的伪代码讲的是算法思路：每一步该做什么、为什么这么做。它不是能直接粘贴的最终代码，你得照着思路翻译成自己的实现，这个翻译的过程才是真正学懂的地方。骨架里每个函数都写好了签名和分步 TODO，对照着看。

### 3.1 先做地基：`rag/vector_store.py`

向量库就干两件事：把（文本块, 向量）存起来；给一个问题向量，找出最像的几个块。

**`add`：往库里追加**

```
add(texts, embeddings, metadatas=None):
    if metadatas is None:
        metadatas = [{} for _ in texts]      # 没给来源就用空 dict 占位，长度要和 texts 对齐
    self.texts.extend(texts)                  # 三个列表同步追加，靠相同下标对应同一个块
    self.embeddings.extend(embeddings)
    self.metadatas.extend(metadatas)
```

**`search`：余弦相似度 + 取 top_k**

相似度用余弦相似度：两个向量夹角越小越相似，公式 `cos = a·b / (|a|·|b|)`，值域 -1~1，越接近 1 越相关。下面用 numpy 一次算出问题跟库里所有块的相似度（向量化，比 for 循环快得多）：

```
search(query_embedding, top_k=3):
    if len(self) == 0:                        # ① 空库先挡掉，否则下面在空数组上算会出错
        return []

    q = np.array(query_embedding)             # 形状 (dim,)
    M = np.array(self.embeddings)             # 形状 (N, dim)：N 个块，每块一行

    # ② 一次算出 q 与每个块的余弦相似度
    dots  = M @ q                             # (N,) 每个块和 q 的点积 a·b
    norms = np.linalg.norm(M, axis=1) * np.linalg.norm(q)   # 每个块的 |a| 乘上 |q|
    scores = dots / (norms + 1e-8)            # +1e-8 防止除以零

    # ③ 取分数最高的 top_k 个块的下标
    order = np.argsort(scores)[::-1][:top_k]  # argsort 升序 → 反转成降序 → 切前 top_k

    return [
        {"text": self.texts[i], "score": float(scores[i]), "metadata": self.metadatas[i]}
        for i in order
    ]                                         # 已按 score 降序
```

> 想验地基对不对，不用等整条链路通：临时手写两三个简单向量塞进去 `search` 一下，看返回的结构和排序对不对就行。

### 3.2 把文档喂进去：`rag/ingest.py`

**`chunk_text`：把长文切成带重叠的小块**

为什么要切块？整篇文档太长，而且检索粒度太粗会不准——你想要的是「价格那一段」，不是整篇概览。为什么要重叠？防止一个完整的事实被从中间切断（比如「专业版 ¥18/月」刚好落在两块的边界上），相邻块重叠几十个字，跨块的信息就丢不了。

```
chunk_text(text, chunk_size=300, overlap=50):
    step = chunk_size - overlap               # 每次前进的步长，必须 > 0（否则死循环）
    chunks = []
    start = 0
    while start < len(text):
        piece = text[start : start + chunk_size]
        if piece.strip():                     # 跳过纯空白的块
            chunks.append(piece)
        start += step                         # 前进 step，相邻两块自然重叠了 overlap 个字
    return chunks
```

**`ingest_file` / `ingest_dir`：切块、嵌入、入库**

```
ingest_file(path):
    text   = 读取 path 的全部文本
    chunks = chunk_text(text)
    if not chunks:
        return 0
    embs   = _llm.embed(chunks)               # 一次把整批块嵌入成向量
    metas  = [{"doc": path} for _ in chunks]  # 记下每块来自哪个文件，检索时好显示来源
    STORE.add(chunks, embs, metas)
    return len(chunks)

ingest_dir(dir_path="data/docs"):
    total = 0
    for path in 遍历 dir_path 下所有 .md / .txt 文件:
        total += ingest_file(path)
    return total
```

### 3.3 包成一个工具：`tools/knowledge_base.py`

最后一步，把检索做成一个模型能在 function calling 里调的工具（不熟工具循环就回看 `背景知识.md` 第 1.5 节）。它干的事是：把问题转成向量 → 检索 → 把命中的片段拼成一段返回给模型，同时把来源记到 `last_sources`（右侧仪表栏的「引用来源」就是读它画出来的）。

```
run(query):
    q_emb = self._llm.embed([query])[0]       # embed 收 list 返回 list，单个问题要包成 [query] 再取 [0]
    hits  = STORE.search(q_emb, top_k=3)
    if not hits:
        self.last_sources = []
        return "知识库里没有找到相关资料。"     # 查不到也要返回一句话，别抛异常
    # 把来源写进 last_sources，仪表栏靠它显示引用了哪些文档
    self.last_sources = [
        {"doc": h["metadata"].get("doc"), "score": h["score"], "snippet": h["text"][:50]}
        for h in hits
    ]
    # 把命中的几段拼成一段喂回给模型。这是给模型读的，信息够清楚即可，不必排版漂亮
    return "\n\n".join(h["text"] for h in hits)
```

`is_available` 是这个工具的**开关**，也是整个 RAG 的开关：

```
is_available():
    return len(STORE) > 0        # 库里有数据才启用；空库时自动隐藏（优雅降级）
```

### 3.4 打开开关，让它真正跑起来

把上面三处填完后，启用链路是这样自动接通的：

```
实现 add/search/chunk_text/ingest/run + is_available 改成 len(STORE)>0
        │  重启服务
        ▼
启动时自动调 ingest_dir() 把 data/docs 入库  →  STORE 非空
        ▼
is_available() 返回 True  →  knowledge_base 进入「可用工具」清单
        ▼
GET /api/capabilities 里 "rag_enabled": true，仪表栏出现该工具
```

所以填完 TODO 一定要重启 `uvicorn`（或重跑 `python main.py`）：入库只在启动时跑一次，不重启的话库一直是空的，`rag_enabled` 也就一直是 false。

---

## 四、易踩坑

- `ingest.py` 和 `knowledge_base.py` 里都得 `from rag.vector_store import STORE`，用那个共享实例，千万别自己再 `VectorStore()` 新建一个。各建各的，就成了入库往 A 库写、检索去 B 库查，永远查不到。这是最高频的翻车点。
- `embed` 的入参和返回都是 list：入库时 `embed(chunks)` 传整批；检索时单个问题要写成 `embed([query])[0]`，先包成单元素 list，再取第 0 个。
- 空库要先挡掉。`search` 在 `len(STORE)==0` 时直接返回 `[]`，否则 numpy 在空数组上算余弦会报错。
- 切块的步长必须为正，也就是 `chunk_size - overlap > 0`，否则 `start` 不前进，`while` 会死循环。顺手用 `strip()` 滤掉纯空白块。
- 改完代码必须重启。入库只在启动时跑一次（见 3.4），代码对了却忘了重启，库还是空的，容易让你误以为没写对。
- `is_available` 别忘了改。骨架里它先返回 `False`，工具被隐藏、模型根本看不到；实现完才改成 `len(STORE) > 0`。
- 入库和检索要用同一个 embedding 模型，维度才对得齐，余弦才算得了。`.env` 里的 `EMBEDDING_MODEL` 别中途换。
- `run` 里要把异常接住。检索失败也好、查无结果也好，都返回一句说明（比如「没找到相关资料」），别让异常把整个回合打断。这就是 `背景知识.md` 第 1.7 节说的优雅降级。

---

## 五、验证清单：域内 / 域外问题

知识库已**限定**为虚构产品「知简笔记」，语料是 `data/docs/` 下的四篇 `知简-*.md`。因为这些事实模型不可能预训练已知，所以**域内问题只有靠检索才能答对**——这让 RAG 是否生效一目了然。

> 注意：本文件**刻意不放在 `data/docs/` 内**，否则会被当作语料一起入库、污染检索结果。

### 怎么用这份清单

1. 按上面一～四节实现并启用 RAG，重启服务让 `data/docs/` 入库（见 3.4）。
2. 在网页界面或 CLI 逐条提问下面的「域内问题」，核对：
   - **答案**是否与「预期答案」一致；
   - 右侧仪表栏「引用来源」是否命中「预期出处」文档。
3. 再提「域外问题」，确认助手**不会凭空编造**知识库里没有的事实。

### 域内问题（应能答对 + 命中正确出处）

| # | 问题 | 预期答案 | 预期出处 |
| --- | --- | --- | --- |
| 1 | 知简专业版多少钱？ | ¥18/月 或 ¥168/年 | 知简-价格与套餐 |
| 2 | 免费版最多能存多少篇笔记？ | 100 篇 | 知简-价格与套餐 |
| 3 | 全局搜索的快捷键是什么？ | Ctrl+Shift+F | 知简-功能与快捷键 |
| 4 | 团队版的版本历史保留多久？ | 1 年 | 知简-价格与套餐 |
| 5 | 知简能导入哪些来源的笔记？ | Evernote（.enex）和 Notion 导出包 | 知简-功能与快捷键 |
| 6 | 知简的退款政策是怎样的？ | 7 天无理由退款 | 知简-客服与隐私 |
| 7 | 免费版最多能在几台设备上同步？ | 5 台 | 知简-价格与套餐 / 知简-客服与隐私 |
| 8 | 知简的数据会用于训练 AI 模型吗？存在哪里？ | 不会用于训练；存于「华东节点」（上海） | 知简-客服与隐私 |
| 9 | 知简当前最新版本号是多少？ | v4.2（2026-03） | 知简-产品概览 |

### 域外问题（库里没有，检验是否会编造）

| # | 问题 | 期望行为 |
| --- | --- | --- |
| A | 知简笔记的创始人/CEO 是谁？ | 知识库未提及——助手应表示不确定，**不应编造** |
| B | 知简和印象笔记哪个更好？ | 无对比信息——属主观/范围外，应说明无法从知识库回答 |
| C | 北京今天天气怎么样？ | 与知简无关——应由天气工具处理，或说明不在知识库范围 |

---

## 六、进阶检验建议

- 改写问法（如把「多少钱」换成「价格」「收费」），看检索是否仍命中——检验嵌入语义匹配而非关键词匹配。
- 比较 `top_k` 取不同值时的命中情况。
- 对域外问题，观察检索分数是否明显偏低（可作为「是否进入知识库范围」的判据）。
- 加重排序（rerank）或多路召回，看域内命中率和答案质量有没有提升（属优化加分项）。
