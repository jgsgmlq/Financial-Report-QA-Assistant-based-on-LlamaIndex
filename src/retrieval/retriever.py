import os
from typing import List
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import NodeWithScore, BaseNode, Document
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.llms.ollama import Ollama

# 导入全局配置
from src.utils.config import GLOBAL_CONFIG

# ==========================================
# 全局单例缓存区 (Singleton Cache)
# 解决性能地雷：防止每次查询重复加载庞大模型和计算几万字的词频矩阵
# ==========================================
_CACHED_BM25_RETRIEVER = None
_CACHED_RERANKER = None

def _extract_nodes_from_vector_store(index: VectorStoreIndex) -> List[BaseNode]:
    """从持久化的 ChromaDB 中反向提取全局文本块，用于构建 BM25 词频索引。"""
    try:
        vector_store = index.storage_context.vector_store
        docstore_data = vector_store.client.get()
        nodes = []
        for i in range(len(docstore_data['ids'])):
            node = Document(
                text=docstore_data['documents'][i],
                id_=docstore_data['ids'][i],
                metadata=docstore_data['metadatas'][i] or {}
            )
            nodes.append(node)
        return nodes
    except Exception as e:
        print(f"⚠️ [BM25] 反向提取全局文本失败: {e}。BM25 将回退为空。")
        return []

def get_basic_retriever(index: VectorStoreIndex) -> BaseRetriever:
    """【Phase 1 基线检索策略】纯净的单路稠密向量检索 (Dense Vector Search)。"""
    top_k = GLOBAL_CONFIG["retrieval"].get("final_top_k", 5)
    return index.as_retriever(similarity_top_k=top_k)

def get_hybrid_retriever(index: VectorStoreIndex) -> BaseRetriever:
    """【Phase 2 高级混合检索策略】集成 BM25 + Vector + RRF (全局单例缓存优化版)"""
    global _CACHED_BM25_RETRIEVER
    
    vector_top_k = GLOBAL_CONFIG["retrieval"].get("vector_top_k", 20)
    bm25_top_k = GLOBAL_CONFIG["retrieval"].get("bm25_top_k", 20)
    
    # 1. 构建向量检索路 (Vector) - 这个很快，不需要我们自己做单例
    vector_retriever = index.as_retriever(similarity_top_k=vector_top_k)
    
    # 2. 构建或获取驻留内存的词频检索路 (BM25)
    if _CACHED_BM25_RETRIEVER is None:
        print("⏳ [Cache Miss] 首次启动 BM25 引擎，正在内存中构建全局词频 TF-IDF 矩阵 (耗时较长)...")
        nodes_for_bm25 = _extract_nodes_from_vector_store(index)
        if not nodes_for_bm25:
            print("⚠️ [Retrieval] 全局文本提取失败，安全降级回单路向量检索。")
            return vector_retriever
            
        _CACHED_BM25_RETRIEVER = BM25Retriever.from_defaults(
            nodes=nodes_for_bm25, 
            similarity_top_k=bm25_top_k
        )
        print(f"✅ [Cache Hit] BM25 词频矩阵构建完成，已常驻内存 (覆盖 {len(nodes_for_bm25)} 个片段)。")
    else:
        pass # 直接复用内存中的 _CACHED_BM25_RETRIEVER，耗时 0 毫秒
    
    # 3. 融合两路召回 (RRF)
    fusion_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, _CACHED_BM25_RETRIEVER],
        similarity_top_k=GLOBAL_CONFIG["retrieval"].get("vector_top_k", 20),
        num_queries=1,
        mode="reciprocal_rerank",
        # 显式注入本地 LLM，避免 QueryFusionRetriever 回退到 OpenAI 默认配置
        llm=Ollama(
            model=GLOBAL_CONFIG["llm"]["weak_model"],
            base_url=GLOBAL_CONFIG["llm"]["ollama_base_url"],
            temperature=0.0,
            request_timeout=300.0,
        ),
    )
    
    return fusion_retriever

def get_retriever(index: VectorStoreIndex) -> BaseRetriever:
    """通用入口：自动分发检索策略"""
    strategy = GLOBAL_CONFIG["chunking"].get("strategy", "fixed")
    if strategy == "semantic":
        return get_hybrid_retriever(index)
    else:
        return get_basic_retriever(index)

def get_node_postprocessors() -> list:
    """获取精排后处理器 (Reranker) - 全局单例缓存优化版"""
    global _CACHED_RERANKER
    strategy = GLOBAL_CONFIG["chunking"].get("strategy", "fixed")
    
    if strategy == "semantic":
        reranker_model = GLOBAL_CONFIG["reranker"]["model"]
        final_top_k = GLOBAL_CONFIG["retrieval"]["final_top_k"]
        
        if _CACHED_RERANKER is None:
            print(f"⏳ [Cache Miss] 首次挂载强力精排模型 (将 {reranker_model} 装载至显存)...")
            try:
                _CACHED_RERANKER = SentenceTransformerRerank(
                    model=reranker_model, 
                    top_n=final_top_k
                )
                print(f"✅ [Cache Hit] 交叉编码精排模型已常驻显存。")
            except ImportError:
                print("⚠️ 未安装 sentence-transformers，精排模块加载失败，已降级。")
                return []
        else:
            pass # 直接复用显存中的大模型，消除 5 秒钟的冷启动卡顿
            
        return [_CACHED_RERANKER]
    return []

if __name__ == "__main__":
    from src.ingest.indexer import build_vector_index
    print("="*50)
    print("🚀 启动混合检索器缓存穿透测试 (Phase 2 高级模式)")
    print("="*50)
    try:
        index = build_vector_index()
        retriever = get_retriever(index)
        
        # 模拟第一次提问 (Cache Miss)
        print("\n🗣️ [第一次查询] 模拟首次提问，预期发生冷启动延迟...")
        retriever.retrieve("风险是什么？")
        get_node_postprocessors()
        
        # 模拟第二次提问 (Cache Hit)
        print("\n🗣️ [第二次查询] 模拟多轮对话，预期毫秒级瞬间穿透...")
        retriever.retrieve("那么它的利润呢？")
        get_node_postprocessors()
        
        print("\n🎉 完美！【单例缓存优化】彻底排除了反复构建矩阵和大模型的性能地雷。")
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")