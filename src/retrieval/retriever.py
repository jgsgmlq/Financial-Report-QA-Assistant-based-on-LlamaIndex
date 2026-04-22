from typing import List
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import BaseNode
from src.utils.config import load_config
from src.ingest.indexer import get_persisted_nodes
from src.generation.llm_backend import init_llm

def get_hybrid_retriever(index: VectorStoreIndex, nodes: List[BaseNode] = None):
    config = load_config()
    
    # 向量检索
    vector_retriever = index.as_retriever(similarity_top_k=config['retrieval']['vector_top_k'])
    
    # 优先从缓存获取节点，解决 BM25 性能问题
    if nodes is None:
        nodes = get_persisted_nodes()
        
    if nodes:
        node_count = len(nodes)
        bm25_top_k = min(config['retrieval']['bm25_top_k'], node_count)
        final_top_k = min(config['retrieval']['final_top_k'], node_count)
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes, 
            similarity_top_k=bm25_top_k
        )
    else:
        # Fallback to vector only if no nodes available
        return vector_retriever
    
    return QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=final_top_k,
        mode="reciprocal_rerank",
        # In Gradio worker threads (Python 3.9), nested async often fails with no event loop.
        use_async=False,
        llm=init_llm(),
    )
