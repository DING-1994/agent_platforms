"""
LLM Agent Platform — 可视化拖拽创建 Agent，建立关系，进行对话
Usage: python app.py
"""

import json, os, uuid, io, threading, textwrap, tempfile
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import networkx as nx
from openai import OpenAI

# ── 每用户独立状态 ──────────────────────────────────────
# 每个浏览器 session 通过 gr.State 持有自己的 store，彼此完全不可见：
#   store = {
#       "agents":        { id: {id, name, role, prompt, model, color, x, y} },
#       "edges":         [ {id, source, target} ],
#       "conversations": { conv_id: [messages] },
#   }
# Gradio 会为每个 session 深拷贝一份默认值，所以 new_store() 返回的只是模板。
# 注意：store 里不能放 threading.Lock —— deepcopy 复制不了锁对象。
def new_store() -> dict:
    return {"agents": {}, "edges": [], "conversations": {}}

# 同一个 session 自己也可能并发触发事件（例如连点按钮），用一把模块级锁保护写操作。
# 不同 session 操作的是各自的 store，这点争用可以忽略。
_lock = threading.Lock()

_active_users: set[str] = set()   # 在线 session_hash，仅用于人数显示

COLORS = [
    "#e94560", "#3498db", "#2ecc71", "#9b59b6",
    "#f39c12", "#1abc9c", "#e67e22", "#00cec9",
]

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")

# ── LLM 调用 ───────────────────────────────────────────
def llm_reply(agent: dict, history: list[dict], visible_ids: set[str] | None = None) -> str:
    """调用 OpenAI 兼容 API 生成回复。
    每个 agent 可指定自己的 model（含 fine-tuned），未指定则用全局 DEFAULT_MODEL。
    visible_ids: 该 agent 能"听到"的其他 agent id 集合（基于连线拓扑）。
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not api_key:
        return "[请设置环境变量 OPENAI_API_KEY]"
    client = OpenAI(api_key=api_key, base_url=base_url)
    model = agent.get("model") or DEFAULT_MODEL
    messages = [{"role": "system", "content": agent["prompt"]}]
    for m in history:
        if m["agent_id"] == agent["id"]:
            messages.append({"role": "assistant", "content": m["content"]})
        elif visible_ids is None or m["agent_id"] in visible_ids:
            messages.append({"role": "user", "content": f'[{m["name"]}]: {m["content"]}'})
    try:
        r = client.chat.completions.create(
            model=model,
            messages=messages, max_tokens=512, temperature=0.8,
        )
        return r.choices[0].message.content or "(empty)"
    except Exception as e:
        return f"[LLM Error ({model}): {e}]"


# ── 画布渲染 ───────────────────────────────────────────
def render_canvas(store: dict,
                  bubbles: dict[str, str] | None = None,
                  speaking_id: str | None = None) -> str:
    """用 matplotlib 绘制 agent 节点、连线和对话气泡，返回高清 PNG 路径。
    store:       调用方 session 自己的状态
    bubbles:     {agent_id: 最新发言文本}  — 只显示 speaking_id 的气泡
    speaking_id: 当前正在说话的 agent id — 高亮其边框并显示气泡
    """
    agents, edges = store["agents"], store["edges"]
    fig, ax = plt.subplots(figsize=(10, 7), dpi=200)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(0, 800)
    ax.set_ylim(-60, 540)
    ax.set_aspect("equal")
    ax.axis("off")

    if not agents:
        ax.text(400, 250, "No agents yet.\nClick 'Add Agent' to start!",
                ha="center", va="center", fontsize=14, color="#888")
    else:
        pos = {}
        for a in agents.values():
            pos[a["id"]] = (a["x"], a["y"])

        # 画连线
        for e in edges:
            if e["source"] in pos and e["target"] in pos:
                x0, y0 = pos[e["source"]]
                x1, y1 = pos[e["target"]]
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle="<->", color="#bbb", lw=2))

        # 画节点
        for a in agents.values():
            is_speaking = (speaking_id == a["id"])
            ec = "#ffd700" if is_speaking else "#666"
            lw = 3.5 if is_speaking else 2
            circle = plt.Circle((a["x"], a["y"]), 38,
                                 color=a["color"], ec=ec, lw=lw, zorder=5)
            ax.add_patch(circle)
            ax.text(a["x"], a["y"] + 8, a["name"], ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white", zorder=6)
            ax.text(a["x"], a["y"] - 6, a["role"], ha="center", va="center",
                    fontsize=7, color="#eee", zorder=6)
            m_label = a.get("model") or DEFAULT_MODEL
            if len(m_label) > 16:
                m_label = m_label[:14] + ".."
            ax.text(a["x"], a["y"] - 18, m_label, ha="center", va="center",
                    fontsize=5, color="#ddd", zorder=6)

        # 只画当前发言者的气泡
        if bubbles and speaking_id and speaking_id in bubbles and speaking_id in pos:
            aid = speaking_id
            text = bubbles[aid]
            bx, by = pos[aid]
            short = text[:120] + ("..." if len(text) > 120 else "")
            wrapped = textwrap.fill(short, width=22)
            bubble_y = by + 58
            ax.text(bx, bubble_y, wrapped, ha="center", va="bottom",
                    fontsize=6, color="#222", zorder=10,
                    bbox=dict(boxstyle="round,pad=0.4",
                              fc=agents[aid]["color"] + "22",
                              ec=agents[aid]["color"], lw=1.2))

    plt.tight_layout()
    # 导出高清 PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.write(buf.read())
    tmp.close()
    return tmp.name


# ── Agent CRUD ─────────────────────────────────────────
def add_agent(store: dict, name: str, role: str, prompt: str, model: str):
    agents = store["agents"]
    with _lock:
        if not name.strip():
            name = f"Agent-{len(agents)+1}"
        aid = str(uuid.uuid4())[:8]
        col = len(agents) % 4
        row = len(agents) // 4
        agents[aid] = {
            "id": aid, "name": name.strip(),
            "role": role.strip() or "Assistant",
            "prompt": prompt.strip() or f"You are {name}, a helpful assistant.",
            "model": model.strip() or "",
            "color": COLORS[len(agents) % len(COLORS)],
            "x": 120 + col * 180, "y": 400 - row * 150,
        }
    return (
        render_canvas(store),
        agent_dropdown_choices(store),
        agent_dropdown_choices(store),
        agent_dropdown_choices(store),
        agent_checkbox_choices(store),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=""),
    )

def delete_agent(store: dict, selection: str):
    with _lock:
        aid = _parse_id(selection)
        if aid and aid in store["agents"]:
            del store["agents"][aid]
            _remove_edges_for(store, aid)
    return (
        render_canvas(store),
        agent_dropdown_choices(store),
        agent_dropdown_choices(store),
        agent_dropdown_choices(store),
        agent_checkbox_choices(store),
    )

def _remove_edges_for(store: dict, aid: str):
    store["edges"] = [e for e in store["edges"]
                      if e["source"] != aid and e["target"] != aid]

def _get_neighbors(store: dict, aid: str) -> set[str]:
    """根据 store 里的 edges 返回与 aid 直接相连的所有 agent id"""
    neighbors = set()
    for e in store["edges"]:
        if e["source"] == aid:
            neighbors.add(e["target"])
        elif e["target"] == aid:
            neighbors.add(e["source"])
    return neighbors

def _parse_id(s: str) -> str | None:
    if not s:
        return None
    # format: "name (id)"
    if "(" in s and s.endswith(")"):
        return s.rsplit("(", 1)[1][:-1].strip()
    return None

def agent_dropdown_choices(store: dict):
    return gr.update(choices=[f'{a["name"]} ({a["id"]})'
                              for a in store["agents"].values()])

def agent_checkbox_choices(store: dict):
    return gr.update(choices=[f'{a["name"]} ({a["id"]})'
                              for a in store["agents"].values()])

def edge_dropdown_choices(store: dict):
    agents = store["agents"]
    labels = []
    for e in store["edges"]:
        s = agents.get(e["source"], {}).get("name", "?")
        t = agents.get(e["target"], {}).get("name", "?")
        labels.append(f'{s} <-> {t} ({e["id"]})')
    return gr.update(choices=labels)


# ── 关系(Edge) ─────────────────────────────────────────
def add_edge(store: dict, src_sel: str, tgt_sel: str):
    with _lock:
        src = _parse_id(src_sel)
        tgt = _parse_id(tgt_sel)
        if not src or not tgt or src == tgt:
            return render_canvas(store), edge_dropdown_choices(store)
        for e in store["edges"]:
            if {e["source"], e["target"]} == {src, tgt}:
                return render_canvas(store), edge_dropdown_choices(store)
        eid = str(uuid.uuid4())[:8]
        store["edges"].append({"id": eid, "source": src, "target": tgt})
    return render_canvas(store), edge_dropdown_choices(store)

def delete_edge(store: dict, selection: str):
    with _lock:
        eid = _parse_id(selection)
        if eid:
            store["edges"] = [e for e in store["edges"] if e["id"] != eid]
    return render_canvas(store), edge_dropdown_choices(store)


# ── 对话 ───────────────────────────────────────────────
def _build_bubbles(history: list[dict]) -> dict[str, str]:
    """从 history 中提取每个 agent 的最新发言"""
    latest: dict[str, str] = {}
    for m in history:
        latest[m["agent_id"]] = m["content"]
    return latest

def start_conversation(store: dict, agent_selections: list[str],
                       topic: str, turns: int):
    """支持 N 个 agent 的群聊，按连线拓扑决定每个 agent 能听到谁。
    每轮同时 yield (画布, 文本日志) 让气泡实时显示在节点旁。
    """
    if not agent_selections or len(agent_selections) < 2:
        yield render_canvas(store), "Please select at least 2 agents to start a conversation."
        return

    # 解析选中的 agent
    participants = []
    for sel in agent_selections:
        aid = _parse_id(sel)
        if aid and aid in store["agents"]:
            participants.append(store["agents"][aid])
    if len(participants) < 2:
        yield render_canvas(store), "Need at least 2 valid agents."
        return

    participant_ids = {p["id"] for p in participants}

    # 检查连通性
    isolated = []
    for p in participants:
        neighbors = _get_neighbors(store, p["id"]) & participant_ids
        if not neighbors:
            isolated.append(p["name"])
    if isolated:
        msg = (f"**Cannot start**: {', '.join(isolated)} ha{'s' if len(isolated)==1 else 've'} "
               f"no connections to other selected agents.\n\n"
               f"Please go to **Relations** tab and connect them first.")
        yield render_canvas(store), msg
        return

    # 构建每个 agent 的可见集（邻居 ∩ 参与者）
    visibility: dict[str, set[str]] = {}
    topo_lines = []
    for p in participants:
        visible = _get_neighbors(store, p["id"]) & participant_ids
        visibility[p["id"]] = visible
        visible_names = [store["agents"][vid]["name"]
                         for vid in visible if vid in store["agents"]]
        topo_lines.append(f"- **{p['name']}** hears: {', '.join(visible_names)}")
    topo_header = "**Topology:**\n" + "\n".join(topo_lines) + "\n\n---\n\n"

    history: list[dict] = []
    # 第一个 agent 开场
    opener = participants[0]
    opening = topic.strip() if topic.strip() else "Hello! Let's have a conversation."
    history.append({"agent_id": opener["id"], "name": opener["name"], "content": opening})
    yield (render_canvas(store, _build_bubbles(history), opener["id"]),
           topo_header + format_log(history))

    # 从第二个 agent 开始，轮流发言
    n = len(participants)
    for turn in range(int(turns) - 1):
        speaker = participants[(turn + 1) % n]
        reply = llm_reply(speaker, history, visible_ids=visibility[speaker["id"]])
        history.append({"agent_id": speaker["id"], "name": speaker["name"], "content": reply})
        yield (render_canvas(store, _build_bubbles(history), speaker["id"]),
               topo_header + format_log(history))

    conv_id = str(uuid.uuid4())[:8]
    store["conversations"][conv_id] = history
    # 最终帧：无高亮
    yield (render_canvas(store, _build_bubbles(history)),
           topo_header + format_log(history) + "\n\n--- Conversation finished ---")

def format_log(history: list[dict]) -> str:
    lines = []
    for m in history:
        lines.append(f'**{m["name"]}**: {m["content"]}')
    return "\n\n".join(lines)


# ── 刷新 / 在线人数 ─────────────────────────────────────
def _refresh_all(store: dict):
    """重绘画布并刷新所有下拉列表（手动 Refresh 按钮用）"""
    return (
        render_canvas(store),
        agent_dropdown_choices(store),
        agent_dropdown_choices(store),
        agent_dropdown_choices(store),
        agent_checkbox_choices(store),
        edge_dropdown_choices(store),
    )

def _user_count_text() -> str:
    n = len(_active_users)
    return (f"**{n}** user{'s' if n != 1 else ''} online "
            f"— each with a private workspace")

def _session_id(request: gr.Request) -> str | None:
    """用 Gradio 的 session_hash 标识会话。不要用 id(request)：那是内存地址，
    对象被回收后会被复用，人数统计会漂。"""
    return getattr(request, "session_hash", None)

def _on_connect(request: gr.Request):
    sid = _session_id(request)
    if sid:
        with _lock:
            _active_users.add(sid)
    return _user_count_text()

def _on_disconnect(request: gr.Request):
    sid = _session_id(request)
    if sid:
        with _lock:
            _active_users.discard(sid)


# ── Gradio UI ──────────────────────────────────────────
with gr.Blocks(title="LLM Agent Platform") as app:
    # 每个 session 一份独立状态，Gradio 为每个 session 深拷贝一份默认值
    store = gr.State(new_store())

    gr.Markdown("# LLM Agent Platform\nCreate agents, connect them, and let them talk!\n\n"
                "**Your workspace is private** — the agents, relations and conversations "
                "you create are visible only to you. Note that reloading the page starts "
                "a brand-new empty workspace.")
    status_bar = gr.Markdown(value=_user_count_text())

    with gr.Row():
        # ─ 左侧: 画布 ─
        with gr.Column(scale=3):
            canvas = gr.Image(value=lambda: render_canvas(new_store()),
                              label="Agent Canvas", type="filepath")

        # ─ 右侧: 控制面板 ─
        with gr.Column(scale=2):
            with gr.Tab("Add Agent"):
                a_name  = gr.Textbox(label="Name", placeholder="e.g. Socrates")
                a_role  = gr.Textbox(label="Role", placeholder="e.g. Philosopher")
                a_prompt = gr.Textbox(label="System Prompt", lines=3,
                           placeholder="You are Socrates, the Greek philosopher...")
                a_model = gr.Textbox(label="Model (optional)",
                           placeholder="e.g. gpt-4o, ft:gpt-4o-mini-2024-07-18:my-org:xxx")
                add_btn = gr.Button("Add Agent", variant="primary")

            with gr.Tab("Relations"):
                src_dd = gr.Dropdown(label="Agent A", choices=[])
                tgt_dd = gr.Dropdown(label="Agent B", choices=[])
                link_btn = gr.Button("Connect", variant="primary")
                edge_dd = gr.Dropdown(label="Existing Relations", choices=[])
                unlink_btn = gr.Button("Delete Relation", variant="stop")

            with gr.Tab("Delete"):
                del_dd = gr.Dropdown(label="Select Agent", choices=[])
                del_btn = gr.Button("Delete Agent", variant="stop")

            refresh_btn = gr.Button("Refresh Canvas", variant="secondary", size="sm")

    gr.Markdown("---")
    gr.Markdown("### Agent Conversation (Multi-Agent)")
    with gr.Row():
        conv_agents = gr.CheckboxGroup(label="Select Agents (pick 2+)", choices=[])
        with gr.Column():
            conv_topic = gr.Textbox(label="Opening topic / first message", placeholder="Let's discuss AI ethics...")
            conv_turns = gr.Slider(2, 20, value=6, step=1, label="Turns")
    conv_btn = gr.Button("Start Conversation", variant="primary")
    conv_log = gr.Markdown(value="*Select 2 or more agents and click Start...*")

    # ── Events ─────────────────────────────────────────
    add_btn.click(
        add_agent, [store, a_name, a_role, a_prompt, a_model],
        [canvas, src_dd, tgt_dd, del_dd, conv_agents, a_name, a_role, a_prompt, a_model],
    )
    del_btn.click(
        delete_agent, [store, del_dd],
        [canvas, src_dd, tgt_dd, del_dd, conv_agents],
    )
    link_btn.click(add_edge, [store, src_dd, tgt_dd], [canvas, edge_dd])
    unlink_btn.click(delete_edge, [store, edge_dd], [canvas, edge_dd])

    conv_btn.click(start_conversation,
                   [store, conv_agents, conv_topic, conv_turns],
                   [canvas, conv_log])

    # 手动重绘画布 / 刷新下拉列表
    refresh_btn.click(
        _refresh_all, [store],
        [canvas, src_dd, tgt_dd, del_dd, conv_agents, edge_dd],
    )

    # 这里原本有一个每 5 秒触发的 gr.Timer，用来把别人的改动同步过来。
    # 状态改成每 session 独立之后没有"别人的改动"了，轮询只会让服务端
    # 为每个在线用户每 5 秒重绘一张 200 DPI 的 PNG，纯属浪费，已移除。

    # 用户连接/断开时更新在线人数
    app.load(_on_connect, [], [status_bar])
    app.unload(_on_disconnect)


if __name__ == "__main__":
    share = os.getenv("SHARE", "false").lower() in ("1", "true", "yes")
    port = int(os.getenv("PORT", "7860"))

    if share:
        print("\n=== Starting with public link (SHARE=true) ===")
        print("If the Gradio tunnel fails, you can also use:")
        print("  ngrok http 7860")
        print("  or: ssh -R 80:localhost:7860 serveo.net\n")

    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
        show_error=True,
    )
