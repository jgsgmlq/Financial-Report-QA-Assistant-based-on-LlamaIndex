import os
import chromadb
import pickle
from typing import List
from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import BaseNode
from src.utils.config import load_config
from llama_index.core import SummaryIndex

# --- RAPTOR 逻辑 (层次摘要树构建) ---
def _generate_raptor_nodes(nodes: List[BaseNode], embed_model, llm) -> List[BaseNode]:
    """
    生成层级摘要节点 (简化的 RAPTOR 策略实现)。
    将一组基础节点进行聚类，并使用 LLM 生成每个簇的摘要，形成新的父节点。
    """
    config = load_config()
    max_levels = config.get("raptor", {}).get("max_levels", 3)
    
    print(f"🌲 [RAPTOR] 开始构建层次摘要树 (最大 {max_levels} 层)...")
    
    # 获取原始节点
    all_nodes = list(nodes)
    current_level_nodes = list(nodes)
    
    # 模拟聚类并生成摘要的层级逻辑 (这里使用最简单的相邻分组)
    # 在生产环境中可以采用 KMeans 或 UMAP 聚类
    cluster_size = 5
    
    for level in range(1, max_levels):
        print(f"  > 构建第 {level} 层摘要...")
        next_level_nodes = []
        
        # 将当前层的节点划分为聚簇
        for i in range(0, len(current_level_nodes), cluster_size):
            cluster = current_level_nodes[i:i+cluster_size]
            if len(cluster) == 1 and level > 1:
                # 避免孤立节点无限向上传递
                next_level_nodes.append(cluster[0])
                continue
                
            # 基于簇生成摘要
            cluster_text = "\n\n".join([n.get_content() for n in cluster])
            summary_prompt = f"请简明扼要地总结以下包含 {len(cluster)} 段文本的综合信息，提取最关键的论点、数据和结论：\n\n{cluster_text}"
            
            try:
                # 尝试调用 LLM 获取摘要
                response = llm.complete(summary_prompt)
                summary_text = str(response)
                
                # 创建一个包含摘要内容的新父节点
                from llama_index.core.schema import TextNode
                parent_node = TextNode(
                    text=summary_text,
                    metadata={"raptor_level": level, "is_summary": True}
                )
                next_level_nodes.append(parent_node)
            except Exception as e:
                 print(f"  [错误] 摘要生成失败: {e}")
                 
        if not next_level_nodes or len(next_level_nodes) == len(current_level_nodes):
            break # 没有更多聚类或收敛
            
        all_nodes.extend(next_level_nodes)
        current_level_nodes = next_level_nodes
        
    print(f"🌲 [RAPTOR] 树构建完成，总节点数从 {len(nodes)} 扩充至 {len(all_nodes)}。")
    return all_nodes
# -----------------------------------

def _get_or_init_embed_model(config):
    """Avoid triggering LlamaIndex default OpenAI embed resolver."""
    embed_model = getattr(Settings, "_embed_model", None)
    if embed_model is None:
        embed_model = HuggingFaceEmbedding(
            model_name=config["embedding"]["model"],
            device=config["embedding"]["device"],
        )
        Settings.embed_model = embed_model
    return embed_model


def build_or_load_index(nodes: List[BaseNode] = None) -> VectorStoreIndex:
    """
    加载持久化或从头构建包含 RAPTOR 支持的向量索引。
    """
    config = load_config()
    persist_dir = config['storage']['chroma_persist_dir']
    node_cache_path = os.path.join(persist_dir, "nodes_cache.pkl")
    collection_name = "financial_reports"
    use_raptor = config.get("raptor", {}).get("use_raptor", False)
    
    # 确保嵌入模型可用（避免默认回退到 OpenAI 依赖）
    embed_model = _get_or_init_embed_model(config)
    
    db = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    if nodes:
        # 新构建
        os.makedirs(persist_dir, exist_ok=True)
        
        final_nodes = nodes
        if use_raptor:
             # 获取专门用于生成摘要的 LLM (小模型)
             from llama_index.llms.ollama import Ollama
             summary_llm = Ollama(
                 model=config['raptor']['summary_model'],
                 base_url=config['llm']['ollama_base_url'],
                 request_timeout=300.0
             )
             # 执行层次扩充
             final_nodes = _generate_raptor_nodes(nodes, embed_model, summary_llm)
             
        # 保存节点以供 BM25 使用 (包含树结构的所有节点)
        with open(node_cache_path, "wb") as f:
            pickle.dump(final_nodes, f)
            
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(final_nodes, storage_context=storage_context)
    else:
        # 仅加载
        index = VectorStoreIndex.from_vector_store(vector_store)
        
    return index

def get_persisted_nodes():
    """获取所有缓存节点，确保包含 RAPTOR 生成的层级。"""
    config = load_config()
    node_cache_path = os.path.join(config['storage']['chroma_persist_dir'], "nodes_cache.pkl")
    if os.path.exists(node_cache_path):
        with open(node_cache_path, "rb") as f:
            return pickle.load(f)
    return None

def remove_document_from_index(index: VectorStoreIndex, doc_id: str):
    """
    增量删除（RAPTOR 树构建后删除父节点的代价较高，P0 只处理叶子文档引用）。
    """
    index.delete_ref_doc(doc_id, delete_from_docstore=True)
    return index
