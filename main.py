"""命令行入口：不开前端也能与助手对话，方便快速测试。

用法：python main.py  （需先在 .env 配好 key）
"""
from factory import build_agent


def main():
    agent = build_agent("cli")
    # 把 data/docs 下的种子文档入库；RAG 未实现时静默跳过
    try:
        from rag.ingest import ingest_dir
        ingest_dir()
    except (ImportError, NotImplementedError):
        pass
    print("已启用工具:", agent.registry.available_names())
    print("输入 exit 退出。\n")
    while True:
        try:
            user = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("exit", "quit"):
            break
        print("助手 > ", end="", flush=True)
        for ev in agent.chat_stream(user):
            if ev["type"] == "token":
                print(ev["delta"], end="", flush=True)
            elif ev["type"] == "tool_call":
                print(f"\n  [调用工具 {ev['name']} 参数={ev['arguments']}]\n助手 > ", end="", flush=True)
            elif ev["type"] == "sources" and ev["items"]:
                docs = [s.get("doc") for s in ev["items"]]
                print(f"\n  [来源: {docs}]", end="")
            elif ev["type"] == "error":
                print(f"\n  [错误: {ev['message']}]", end="")
        print("\n")


if __name__ == "__main__":
    main()
