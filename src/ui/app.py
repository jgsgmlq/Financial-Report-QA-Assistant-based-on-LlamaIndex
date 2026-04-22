import os
import re
import gradio as gr
import gradio_client.utils as gradio_client_utils
from pathlib import Path


def _patch_gradio_client_schema():
    """Pydantic v2 can emit additionalProperties as bool; gradio_client 1.3 assumes dict."""
    _orig = gradio_client_utils._json_schema_to_python_type

    def _wrap(schema, defs):
        if not isinstance(schema, dict):
            return "Any"
        return _orig(schema, defs)

    gradio_client_utils._json_schema_to_python_type = _wrap


_patch_gradio_client_schema()
from src.utils.config import load_config
from src.ingest.pdf_parser import load_pdf_documents
from src.ingest.chunker import get_chunks
from src.ingest.indexer import build_or_load_index, remove_document_from_index
from src.retrieval.retriever import get_hybrid_retriever
from src.retrieval.reranker import get_reranker
from src.generation.pipeline import create_chat_engine
from src.generation.overview import generate_document_overview
from src.generation.workspace import generate_comparison_table

class AppState:
    def __init__(self):
        self.chat_engine = None
        self.index = None
        self.nodes = None
        self.doc_map = {}
        self.data_dir = load_config()['storage']['data_dir']
        os.makedirs(self.data_dir, exist_ok=True)

state = AppState()

def initialize_system(pdf_files=None, user_memory=""):
    if pdf_files or not state.index:
        documents = load_pdf_documents(state.data_dir)
        state.doc_map = {doc.metadata['source']: doc.doc_id for doc in documents}
        state.nodes = get_chunks(documents)
        state.index = build_or_load_index(state.nodes)
            
    retriever = get_hybrid_retriever(state.index, state.nodes)
    reranker = get_reranker()
    state.chat_engine = create_chat_engine(retriever, reranker, user_memory=user_memory)
    
    overview = generate_document_overview(state.nodes)
    return "系统就绪", overview['summary'], overview['questions']

def chat_response(message, history, user_memory=""):
    if not state.chat_engine:
        initialize_system(user_memory=user_memory)
    chat_response_obj = state.chat_engine.stream_chat(message)
    partial_message = ""
    for token in chat_response_obj.response_gen:
        partial_message += token
        yield partial_message, chat_response_obj.source_nodes

# ... (此处必须包含前面缺失的两个重要函数) ...

def update_pdf_viewer(doc_name, page_num):
    if not doc_name:
        return "请在对话中点击引用来源，或在左侧选择文档以在此处预览。"
    abs_path = os.path.abspath(os.path.join(state.data_dir, doc_name))
    page_anchor = f"#page={page_num}" if page_num else ""
    html_content = f"""
    <iframe 
        src="/file={abs_path}{page_anchor}" 
        width="100%" 
        height="800px" 
        style="border: 1px solid #ccc; border-radius: 8px;">
    </iframe>
    """
    return html_content

def format_citations_to_html(source_nodes):
    if not source_nodes:
        return "无引用来源。", []
        
    html = "<div style='font-size: 0.9em;'>"
    choices = []
    for i, node in enumerate(source_nodes):
        doc = node.metadata.get("source", "未知文档")
        page = node.metadata.get("page_label", "1")
        content = node.get_content()[:150].replace('\n', ' ') + "..."
        is_summary = node.metadata.get("is_summary", False)
        badge = "<span style='background-color:#e0f2fe; color:#0369a1; padding:2px 6px; border-radius:4px; font-size:0.8em;'>宏观摘要</span>" if is_summary else f"<span style='background-color:#fef08a; color:#b45309; padding:2px 6px; border-radius:4px; font-size:0.8em;'>第 {page} 页</span>"
        
        html += f"<div style='margin-bottom: 10px; padding: 10px; background-color: #f8fafc; border-left: 4px solid #3b82f6;'>"
        html += f"<strong>📄 {doc}</strong> {badge}<br/>"
        html += f"<span style='color: #475569;'>{content}</span>"
        html += "</div>"
        
        if not is_summary:
            choices.append(f"{doc} (页码: {page})")
            
    html += "</div>"
    return html, list(set(choices))

def bot_msg(history, user_memory=""):
    user_message = history[-1][0]
    gen = chat_response(user_message, history[:-1], user_memory=user_memory)
    
    history[-1][1] = ""
    source_nodes = []
    for partial_resp, nodes in gen:
        history[-1][1] = partial_resp
        source_nodes = nodes
        yield history, gr.update(visible=False), gr.update(choices=[])

    citations_html, jump_choices = format_citations_to_html(source_nodes)
    
    default_viewer = "在此预览文档..."
    if jump_choices:
        first_doc, first_page = jump_choices[0].split(" (页码: ")
        first_page = first_page.rstrip(")")
        default_viewer = update_pdf_viewer(first_doc, first_page)
        
    yield history, gr.update(value=citations_html, visible=True), gr.update(choices=jump_choices, value=jump_choices[0] if jump_choices else None)

def process_upload(files, user_memory=""):
    names = []
    for f in files:
        dest = Path(state.data_dir) / Path(f.name).name
        with open(f.name, "rb") as src, open(dest, "wb") as dst:
            dst.write(src.read())
        names.append(dest.name)
    
    status, summary, qs = initialize_system(pdf_files=names, user_memory=user_memory)
    viewer_html = update_pdf_viewer(names[0], 1) if names else "请上传文档。"
    
    return (
        f"✅ 索引完成: {status}", 
        gr.update(choices=list(state.doc_map.keys())), 
        summary, 
        qs,
        viewer_html
    )

def handle_jump_selection(selection):
    if not selection:
        return gr.update()
    try:
        doc_name, page_str = selection.split(" (页码: ")
        page_num = page_str.rstrip(")")
        return update_pdf_viewer(doc_name, page_num)
    except:
        return gr.update()

def update_memory_prompt(memory_text):
    """更新内存并强制重建 ChatEngine"""
    if state.index:
        retriever = get_hybrid_retriever(state.index, state.nodes)
        reranker = get_reranker()
        state.chat_engine = create_chat_engine(retriever, reranker, user_memory=memory_text)
    return "✅ 偏好已注入，新提问将遵循该指令。"

# --- 新增的 Workspace 工具函数 ---
def pin_to_notepad(history, current_notepad):

    """提取历史中最后一条系统回答，追加到记事本中。"""
    if not history or not history[-1][1]:
        return current_notepad
    
    answer = history[-1][1]
    new_entry = f"---\n\n📝 **已保存回答 (提取自对话)**:\n{answer}\n\n"
    
    if current_notepad:
        return current_notepad + new_entry
    return new_entry

def generate_table(selected_docs, dimension, current_notepad):
    """生成多篇文档对比表，并尝试直接追加到灵感库。"""
    if not state.index:
        return current_notepad, "⚠️ 错误：系统尚未构建索引。"
        
    table_md = generate_comparison_table(state.index, selected_docs, dimension)
    
    # 无论当前有没有笔记，都把对比结果放进灵感库
    new_entry = f"---\n\n{table_md}\n\n"
    if current_notepad:
        new_content = current_notepad + new_entry
    else:
        new_content = new_entry
        
    return new_content, "✅ 对比表已生成，请查看下方的【灵感库】。"


# --- 构建 Gradio 三栏 UI (Vibe Polish: 高级沉浸质感) ---
custom_css = """
    body { background-color: #f8fafc; font-family: 'Inter', -apple-system, sans-serif; }
    .container { max-width: 95%; margin: auto; padding: 20px; }
    .chat-box { height: 50vh !important; background-color: #ffffff; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }
    .notepad-box { background-color: #fffbeb !important; border: 1px solid #fde68a; border-radius: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}
    .panel-box { background-color: #ffffff; border-radius: 12px; padding: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 15px; }
    .pdf-container iframe { border-radius: 12px; box-shadow: inset 0 2px 4px rgba(0,0,0,0.1); }
"""

with gr.Blocks(title="Insight | 金融研报分析引擎", css=custom_css, theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")) as demo:
    gr.Markdown("# 📈 **Insight** · 机构级研报分析工作台")
    
    with gr.Row():
        # --- 左栏：文档底座 (Width: 20%) ---
        with gr.Column(scale=2, elem_classes="panel-box"):
            gr.Markdown("### 📂 数据底座")
            file_upload = gr.File(label="导入研报 (支持 PDF 深度解析)", file_count="multiple")
            upload_btn = gr.Button("🚀 启动深度索引 (LlamaParse)", variant="primary")
            doc_list = gr.Dropdown(label="已激活的工作区", multiselect=True)
            
            gr.Markdown("---")
            gr.Markdown("### 💡 宏观脉络")
            summary_box = gr.Textbox(label="全局摘要 (RAPTOR 树聚合)", lines=5, show_copy_button=True)
            qs_box = gr.Textbox(label="推荐探索维度", lines=5)
            
            gr.Markdown("---")
            gr.Markdown("### 🧠 分析师记忆设定")
            memory_input = gr.Textbox(label="设定您的偏好", placeholder="例如：我重点关注现金流；请用表格输出...", lines=3)
            memory_btn = gr.Button("注入记忆偏好")
            memory_status = gr.Markdown()
            
            status_text = gr.Markdown("*尚未建立连接...*")

        # --- 中栏：洞察与合成 (Width: 40%) ---
        # ... (保持原样)
        with gr.Column(scale=4):
            gr.Markdown("### 💬 智能洞察助手")
            chatbot = gr.Chatbot(elem_classes="chat-box", show_copy_button=True, bubble_full_width=False, avatar_images=(None, "https://cdn-icons-png.flaticon.com/512/8649/8649595.png"))
            msg_input = gr.Textbox(label="提出您的分析诉求...", placeholder="例如：详细对比这几份报告中提到的23年毛利率波动。")
            
            with gr.Row():
                submit_btn = gr.Button("生成洞察", variant="primary")
                pin_btn = gr.Button("📌 摘录至灵感库", variant="secondary")
                clear_btn = gr.Button("清除会话")
                
            gr.Markdown("#### 🔍 溯源证据链")
            cite_html = gr.HTML(label="引用溯源", visible=False)
            jump_dropdown = gr.Dropdown(label="🎯 沉浸验证 (点击自动跳转右侧阅读器)", choices=[])
            
            gr.Markdown("---")
            gr.Markdown("### 📝 知识聚合")
            with gr.Row():
                dim_input = gr.Textbox(label="跨文档分析维度", placeholder="例如：三季度营收增速...", scale=3)
                table_btn = gr.Button("生成多维对比矩阵", variant="secondary", scale=1)
                
            table_status = gr.Markdown()
            notepad_area = gr.Textbox(label="📌 分析师灵感库 (Notepad)", lines=8, elem_classes="notepad-box", show_copy_button=True)

        # --- 右栏：沉浸式阅读 (Width: 40%) ---
        with gr.Column(scale=4, elem_classes="panel-box"):
            gr.Markdown("### 📖 原文追溯视图")
            pdf_viewer = gr.HTML(value="<div style='text-align:center; padding:100px; color:#94a3b8; font-style: italic;'>👈 暂无激活文档。<br/>请在左侧导入研报，或点击证据链跳转。</div>", elem_classes="pdf-container")

    # --- 绑定事件逻辑 ---
    memory_btn.click(update_memory_prompt, inputs=[memory_input], outputs=[memory_status])
    
    upload_btn.click(
        process_upload, 
        [file_upload, memory_input], 
        [status_text, doc_list, summary_box, qs_box, pdf_viewer]
    )
    
    submit_btn.click(lambda x, h: ("", h + [[x, None]]), [msg_input, chatbot], [msg_input, chatbot], queue=False).then(
        bot_msg, [chatbot, memory_input], [chatbot, cite_html, jump_dropdown]
    ).then(
        handle_jump_selection, jump_dropdown, pdf_viewer
    )
    
    msg_input.submit(lambda x, h: ("", h + [[x, None]]), [msg_input, chatbot], [msg_input, chatbot], queue=False).then(
        bot_msg, [chatbot, memory_input], [chatbot, cite_html, jump_dropdown]
    ).then(
        handle_jump_selection, jump_dropdown, pdf_viewer
    )
    
    jump_dropdown.change(handle_jump_selection, jump_dropdown, pdf_viewer)
    
    pin_btn.click(pin_to_notepad, inputs=[chatbot, notepad_area], outputs=[notepad_area])
    table_btn.click(generate_table, inputs=[doc_list, dim_input, notepad_area], outputs=[notepad_area, table_status])
    
    clear_btn.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    config = load_config()
    data_dir_abs = os.path.abspath(config['storage']['data_dir'])
    # 127.0.0.1: Gradio's post-launch localhost probe fails with 0.0.0.0 on many setups.
    # Use env GRADIO_SERVER_NAME=0.0.0.0 when you need LAN binding.
    _host = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    _port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.queue().launch(server_name=_host, server_port=_port, allowed_paths=[data_dir_abs])
