"""
LLM Agent Platform — 可视化拖拽创建 Agent，建立关系，进行对话
Usage: python app.py
"""

import json, os, uuid, io
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from openai import OpenAI

# ── 状态 ────────────────────────────────────────────────
# agents  = { id: {id, name, role, prompt, color, x, y} }
# edges   = [ {id, source, target} ]
agents: dict[str, dict] = {}
edges: list[dict] = []
conversations: dict[str, list] = {}   # edge_id -> [messages]

COLORS = [
    "#e94560", "#3498db", "#2ecc71", "#9b59b6",
    "#f39c12", "#1abc9c", "#e67e22", "#00cec9",
]

# ── LLM 调用 ───────────────────────────────────────────
def llm_reply(agent: dict, history: list[dict]) -> str:
    """调用 OpenAI 兼容 API 生成回复（支持多 agent 上下文）"""
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not api_key:
        return f"[请设置环境变量 OPENAI_API_KEY]"
    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = [{"role": "system", "content": agent["prompt"]}]
    for m in history:
        if m["agent_id"] == agent["id"]:
            messages.append({"role": "assistant", "content": m["content"]})
        else:
            # 多 agent 时用名字前缀区分不同发言者
            messages.append({"role": "user", "content": f'[{m["name"]}]: {m["content"]}'})
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
    if not name.strip():
        name = f"Agent-{len(agents)+1}"
    aid = str(uuid.uuid4())[:8]
    # 自动排列位置
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
    aid = _parse_id(selection)
    if aid and aid in agents:
        del agents[aid]
        # 删除相关连线
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
    src = _parse_id(src_sel)
    tgt = _parse_id(tgt_sel)
    if not src or not tgt or src == tgt:
        return render_canvas(), edge_dropdown_choices()
    # 避免重复
    for e in edges:
        if {e["source"], e["target"]} == {src, tgt}:
            return render_canvas(), edge_dropdown_choices()
    eid = str(uuid.uuid4())[:8]
    edges.append({"id": eid, "source": src, "target": tgt})
    return render_canvas(), edge_dropdown_choices()

def delete_edge(selection: str):
    eid = _parse_id(selection)
    if eid:
        global edges
        edges = [e for e in edges if e["id"] != eid]
    return render_canvas(), edge_dropdown_choices()


# ── 对话 ───────────────────────────────────────────────
def start_conversation(agent_selections: list[str], topic: str, turns: int):
    """支持 N 个 agent 的群聊，按选中顺序轮流发言"""
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

    history: list[dict] = []
    # 第一个 agent 开场
    opener = participants[0]
    opening = topic.strip() if topic.strip() else "Hello! Let's have a conversation."
    history.append({"agent_id": opener["id"], "name": opener["name"], "content": opening})
    yield format_log(history)

    # 从第二个 agent 开始，轮流发言
    n = len(participants)
    for turn in range(int(turns) - 1):
        speaker = participants[(turn + 1) % n]
        reply = llm_reply(speaker, history)
        history.append({"agent_id": speaker["id"], "name": speaker["name"], "content": reply})
        yield format_log(history)

    conv_id = str(uuid.uuid4())[:8]
    conversations[conv_id] = history
    yield format_log(history) + "\n\n--- Conversation finished ---"

def format_log(history: list[dict]) -> str:
    lines = []
    for m in history:
        lines.append(f'**{m["name"]}**: {m["content"]}')
    return "\n\n".join(lines)


# ── Gradio UI ──────────────────────────────────────────
with gr.Blocks(title="LLM Agent Platform") as app:
    gr.Markdown("# 🤖 LLM Agent Platform\nCreate agents, connect them, and let them talk!")

    with gr.Row():
        # ─ 左侧: 画布 ─
        with gr.Column(scale=3):
            canvas = gr.Plot(value=render_canvas, label="Agent Canvas")

        # ─ 右侧: 控制面板 ─
        with gr.Column(scale=2):
            with gr.Tab("➕ Add Agent"):
                a_name  = gr.Textbox(label="Name", placeholder="e.g. Socrates")
                a_role  = gr.Textbox(label="Role", placeholder="e.g. Philosopher")
                a_prompt = gr.Textbox(label="System Prompt", lines=3,
                           placeholder="You are Socrates, the Greek philosopher...")
                add_btn = gr.Button("Add Agent", variant="primary")

            with gr.Tab("🔗 Relations"):
                src_dd = gr.Dropdown(label="Agent A", choices=[])
                tgt_dd = gr.Dropdown(label="Agent B", choices=[])
                link_btn = gr.Button("Connect", variant="primary")
                edge_dd = gr.Dropdown(label="Existing Relations", choices=[])
                unlink_btn = gr.Button("Delete Relation", variant="stop")

            with gr.Tab("🗑️ Delete"):
                del_dd = gr.Dropdown(label="Select Agent", choices=[])
                del_btn = gr.Button("Delete Agent", variant="stop")

    gr.Markdown("---")
    gr.Markdown("### 💬 Agent Conversation (Multi-Agent)")
    with gr.Row():
        conv_agents = gr.CheckboxGroup(label="Select Agents (pick 2+)", choices=[])
        with gr.Column():
            conv_topic = gr.Textbox(label="Opening topic / first message", placeholder="Let's discuss AI ethics...")
            conv_turns = gr.Slider(2, 20, value=6, step=1, label="Turns")
    conv_btn = gr.Button("▶ Start Conversation", variant="primary")
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


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860,
                theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="purple"))
