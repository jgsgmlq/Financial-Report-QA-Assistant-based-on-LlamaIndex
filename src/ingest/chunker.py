from typing import List
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.core.schema import Document, BaseNode
from llama_index.core import Settings
from src.utils.config import load_config


def _get_or_init_embed_model(config):
    """Avoid triggering LlamaIndex default OpenAI embed resolver."""
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    embed_model = getattr(Settings, "_embed_model", None)
    if embed_model is None:
        embed_model = HuggingFaceEmbedding(
            model_name=config["embedding"]["model"],
            device=config["embedding"]["device"],
        )
        Settings.embed_model = embed_model
    return embed_model


def get_chunks(documents: List[Document]) -> List[BaseNode]:
    """
    根据配置选择分块策略：语义分块 (Semantic) 或 长度分块 (Fixed)。
    """
    config = load_config()
    strategy = config.get("chunking", {}).get("strategy", "fixed")
    
    if strategy == "semantic":
        print("💡 [分块策略] 启用语义分块 (Semantic Chunking)...")
        # 需要一个嵌入模型来计算句间相似度，优先复用全局模型避免重复加载
        embed_model = _get_or_init_embed_model(config)
        
        splitter = SemanticSplitterNodeParser(
            buffer_size=config['chunking'].get('semantic_buffer_size', 1),
            breakpoint_percentile_threshold=config['chunking'].get('semantic_breakpoint_percentile', 95),
            embed_model=embed_model
        )
    else:
        print(f"⚙️ [分块策略] 启用固定长度分块 (Fixed-Size Chunking): {config['chunking']['chunk_size']} tokens")
        splitter = SentenceSplitter(
            chunk_size=config['chunking'].get('chunk_size', 512),
            chunk_overlap=config['chunking'].get('chunk_overlap', 51),
            include_metadata=True,
            include_prev_next_rel=True
        )

    nodes = splitter.get_nodes_from_documents(documents)
    print(f"📦 [分块结果] 从 {len(documents)} 个文档中生成了 {len(nodes)} 个块。")
    return nodes
