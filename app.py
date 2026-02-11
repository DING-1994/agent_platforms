"""
LLM Agent Platform — 可视化拖拽创建 Agent，建立关系，进行对话
Usage: python app.py
"""

import json, os, uuid, io, threading
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from openai import OpenAI

# ── 共享状态（多用户并发访问，需要锁保护） ──────────────
# agents  = { id: {id, name, role, prompt, color, x, y} }
# edges   = [ {id, source, target} ]
_lock = threading.Lock()
agents: dict[str, dict] = {}
edges: list[dict] = []
conversations: dict[str, list] = {}   # conv_id -> [messages]
_active_users: set[str] = set()       # 在线 session id

COLORS = [
    "#e94560", "#3498db", "#2ecc71", "#9b59b6",
    "#f39c12", "#1abc9c", "#e67e22", "#00cec9",
]

# ── LLM 调用 ───────────────────────────────────────────
def llm_reply(agent: dict, history: list[dict], visible_ids: set[str] | None = None) -> str:
    """调用 OpenAI 兼容 API 生成回复。
    visible_ids: 该 agent 能"听到"的其他 agent id 集合（基于连线拓扑）。
    为 None 时表示能听到所有人（向后兼容）。
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not api_key:
        return f"[请设置环境变量 OPENAI_API_KEY]"
    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = [{"role": "system", "content": agent["prompt"]}]
    for m in history:
        if m["agent_id"] == agent["id"]:
            messages.append({"role": "assistant", "content": m["content"]})
        elif visible_ids is None or m["agent_id"] in visible_ids:
            # 只接收能"听到"的 agent 的消息
            messages.append({"role": "user", "content": f'[{m["name"]}]: {m["content"]}'})
        # else: 没有连线 → 这条消息对当前 agent 不可见
    try:
        r = client.chat.completions.create(
            model=os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo"),
            messages=messages, max_tokens=512, temperature=0.8,
        )
        return r.choices[0].message.content or "(empty)"
    except Exception as e:
        return f"[LLM Error: {e}]"


# ── 画布渲染 ───────────────────────────────────────────
def render_canvas() -> plt.Figure:
    """用 matplotlib 绘制 agent 节点和关系连线"""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.set_xlim(0, 800)
    ax.set_ylim(0, 500)
    ax.set_aspect("equal")
    ax.axis("off")

    if not agents:
        ax.text(400, 250, "No agents yet.\nClick 'Add Agent' to start!",
                ha="center", va="center", fontsize=14, color="#888")
        return fig

    pos = {}
    for a in agents.values():
        pos[a["id"]] = (a["x"], a["y"])

    # 画连线
    for e in edges:
        if e["source"] in pos and e["target"] in pos:
            x0, y0 = pos[e["source"]]
            x1, y1 = pos[e["target"]]
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="<->", color="#555", lw=2))

    # 画节点
    for a in agents.values():
        circle = plt.Circle((a["x"], a["y"]), 38, color=a["color"], ec="white", lw=2, zorder=5)
        ax.add_patch(circle)
        ax.text(a["x"], a["y"] + 2, a["name"], ha="center", va="center",
                fontsize=9, fontweight="bold", color="white", zorder=6)
        ax.text(a["x"], a["y"] - 14, a["role"], ha="center", va="center",
                fontsize=7, color="#ccc", zorder=6)

    plt.tight_layout()
    return fig


# ── Agent CRUD ─────────────────────────────────────────
def add_agent(name: str, role: str, prompt: str):
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
            "color": COLORS[len(agents) % len(COLORS)],
            "x": 120 + col * 180, "y": 400 - row * 150,
        }
    return (
        render_canvas(),
        agent_dropdown_choices(),
        agent_dropdown_choices(),
        agent_dropdown_choices(),
        agent_checkbox_choices(),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=""),
    )

def delete_agent(selection: str):
    with _lock:
        aid = _parse_id(selection)
        if aid and aid in agents:
            del agents[aid]
            _remove_edges_for(aid)
    return (
        render_canvas(),
        agent_dropdown_choices(),
        agent_dropdown_choices(),
        agent_dropdown_choices(),
        agent_checkbox_choices(),
    )

def _remove_edges_for(aid: str):
    global edges
    edges = [e for e in edges if e["source"] != aid and e["target"] != aid]

def _get_neighbors(aid: str) -> set[str]:
    """根据 edges 返回与 aid 直接相连的所有 agent id"""
    neighbors = set()
    for e in edges:
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

def agent_dropdown_choices():
    return gr.update(choices=[f'{a["name"]} ({a["id"]})' for a in agents.values()])

def agent_checkbox_choices():
    return gr.update(choices=[f'{a["name"]} ({a["id"]})' for a in agents.values()])

def edge_dropdown_choices():
    labels = []
    for e in edges:
        s = agents.get(e["source"], {}).get("name", "?")
        t = agents.get(e["target"], {}).get("name", "?")
        labels.append(f'{s} <-> {t} ({e["id"]})')
    return gr.update(choices=labels)


# ── 关系(Edge) ─────────────────────────────────────────
def add_edge(src_sel: str, tgt_sel: str):
    with _lock:
        src = _parse_id(src_sel)
        tgt = _parse_id(tgt_sel)
        if not src or not tgt or src == tgt:
            return render_canvas(), edge_dropdown_choices()
        for e in edges:
            if {e["source"], e["target"]} == {src, tgt}:
                return render_canvas(), edge_dropdown_choices()
        eid = str(uuid.uuid4())[:8]
        edges.append({"id": eid, "source": src, "target": tgt})
    return render_canvas(), edge_dropdown_choices()

def delete_edge(selection: str):
    with _lock:
        eid = _parse_id(selection)
        if eid:
            global edges
            edges = [e for e in edges if e["id"] != eid]
    return render_canvas(), edge_dropdown_choices()


# ── 对话 ───────────────────────────────────────────────
def start_conversation(agent_selections: list[str], topic: str, turns: int):
    """支持 N 个 agent 的群聊，按连线拓扑决定每个 agent 能听到谁"""
    if not agent_selections or len(agent_selections) < 2:
        return "Please select at least 2 agents to start a conversation."

    # 解析选中的 agent
    participants = []
    for sel in agent_selections:
        aid = _parse_id(sel)
        if aid and aid in agents:
            participants.append(agents[aid])
    if len(participants) < 2:
        return "Need at least 2 valid agents."

    participant_ids = {p["id"] for p in participants}

    # 检查连通性：找出没有任何连线（在参与者中）的 agent
    isolated = []
    for p in participants:
        neighbors = _get_neighbors(p["id"]) & participant_ids
        if not neighbors:
            isolated.append(p["name"])
    if isolated:
        return (f"**Cannot start**: {', '.join(isolated)} ha{'s' if len(isolated)==1 else 've'} "
                f"no connections to other selected agents.\n\n"
                f"Please go to **Relations** tab and connect them first.")

    # 构建每个 agent 的可见集（邻居 ∩ 参与者）
    visibility: dict[str, set[str]] = {}
    topo_lines = []
    for p in participants:
        visible = _get_neighbors(p["id"]) & participant_ids
        visibility[p["id"]] = visible
        visible_names = [agents[vid]["name"] for vid in visible if vid in agents]
        topo_lines.append(f"- **{p['name']}** hears: {', '.join(visible_names)}")
    topo_header = "**Topology:**\n" + "\n".join(topo_lines) + "\n\n---\n\n"

    history: list[dict] = []
    # 第一个 agent 开场
    opener = participants[0]
    opening = topic.strip() if topic.strip() else "Hello! Let's have a conversation."
    history.append({"agent_id": opener["id"], "name": opener["name"], "content": opening})
    yield topo_header + format_log(history)

    # 从第二个 agent 开始，轮流发言
    n = len(participants)
    for turn in range(int(turns) - 1):
        speaker = participants[(turn + 1) % n]
        reply = llm_reply(speaker, history, visible_ids=visibility[speaker["id"]])
        history.append({"agent_id": speaker["id"], "name": speaker["name"], "content": reply})
        yield topo_header + format_log(history)

    conv_id = str(uuid.uuid4())[:8]
    conversations[conv_id] = history
    yield topo_header + format_log(history) + "\n\n--- Conversation finished ---"

def format_log(history: list[dict]) -> str:
    lines = []
    for m in history:
        lines.append(f'**{m["name"]}**: {m["content"]}')
    return "\n\n".join(lines)


# ── 同步刷新 ───────────────────────────────────────────
def _refresh_all():
    """返回最新画布 + 所有下拉列表，供多人实时同步"""
    return (
        render_canvas(),
        agent_dropdown_choices(),
        agent_dropdown_choices(),
        agent_dropdown_choices(),
        agent_checkbox_choices(),
        edge_dropdown_choices(),
    )

def _user_count_text() -> str:
    n = len(_active_users)
    return f"**{n}** user{'s' if n != 1 else ''} online"

def _on_connect(request: gr.Request):
    sid = str(id(request))
    _active_users.add(sid)
    return _user_count_text()

def _on_disconnect(request: gr.Request):
    sid = str(id(request))
    _active_users.discard(sid)


# ── Gradio UI ──────────────────────────────────────────
with gr.Blocks(title="LLM Agent Platform") as app:
    gr.Markdown("# LLM Agent Platform\nCreate agents, connect them, and let them talk!\n\n"
                "Share this link with others — everyone sees the same canvas in real time.")
    status_bar = gr.Markdown(value=_user_count_text())

    with gr.Row():
        # ─ 左侧: 画布 ─
        with gr.Column(scale=3):
            canvas = gr.Plot(value=render_canvas, label="Agent Canvas")

        # ─ 右侧: 控制面板 ─
        with gr.Column(scale=2):
            with gr.Tab("Add Agent"):
                a_name  = gr.Textbox(label="Name", placeholder="e.g. Socrates")
                a_role  = gr.Textbox(label="Role", placeholder="e.g. Philosopher")
                a_prompt = gr.Textbox(label="System Prompt", lines=3,
                           placeholder="You are Socrates, the Greek philosopher...")
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
        add_agent, [a_name, a_role, a_prompt],
        [canvas, src_dd, tgt_dd, del_dd, conv_agents, a_name, a_role, a_prompt],
    )
    del_btn.click(
        delete_agent, [del_dd],
        [canvas, src_dd, tgt_dd, del_dd, conv_agents],
    )
    link_btn.click(add_edge, [src_dd, tgt_dd], [canvas, edge_dd])
    unlink_btn.click(delete_edge, [edge_dd], [canvas, edge_dd])

    conv_btn.click(start_conversation, [conv_agents, conv_topic, conv_turns], [conv_log])

    # 手动刷新：拉取其他用户的最新更改
    refresh_btn.click(
        _refresh_all, [],
        [canvas, src_dd, tgt_dd, del_dd, conv_agents, edge_dd],
    )

    # 自动定时刷新画布（每 5 秒），让多人协作保持同步
    _timer = gr.Timer(value=5)
    _timer.tick(
        _refresh_all, [],
        [canvas, src_dd, tgt_dd, del_dd, conv_agents, edge_dd],
    )

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
