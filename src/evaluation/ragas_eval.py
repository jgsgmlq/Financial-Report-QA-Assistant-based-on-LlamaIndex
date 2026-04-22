import json
import os
import pandas as pd
from pathlib import Path
from datasets import Dataset

# LlamaIndex & Ollama
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine

# Ragas 相关包
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
from ragas.run_config import RunConfig
# 如果本地运行 Ragas 报错，需要引入以下包装器 (取决于 Ragas 的具体版本)
from ragas.llms import LlamaIndexLLMWrapper
from ragas.embeddings import LlamaIndexEmbeddingsWrapper

# 项目内部模块
from src.utils.config import GLOBAL_CONFIG
from src.ingest.indexer import get_index
from src.retrieval.retriever import get_retriever
from src.generation.pipeline import QA_PROMPT_TEMPLATE

def load_test_set(file_path: str | Path) -> list:
    """加载测试用例"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到测试集文件: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_evaluation():
    """
    执行 Ragas 本地评估闭环
    """
    # 1. 读取配置
    judge_model_name = GLOBAL_CONFIG["evaluation"]["judge_model"]
    base_url = GLOBAL_CONFIG["llm"]["ollama_base_url"]
    embed_model_name = GLOBAL_CONFIG["embedding"]["model"]
    device = GLOBAL_CONFIG["embedding"]["device"]
    test_set_path = GLOBAL_CONFIG["evaluation"]["test_set_path"]
    
    print(f"⚖️ [Evaluation] 启动本地无情裁判: {judge_model_name}")
    
    # 2. 实例化裁判模型 (Judge LLM) 和裁判向量 (Judge Embedding)
    # 这里我们使用强大的 72B 模型作为 Ragas 的裁判
    judge_llm = Ollama(model=judge_model_name, base_url=base_url, request_timeout=600.0)
    judge_embedding = HuggingFaceEmbedding(model_name=embed_model_name, device=device)
    
    # 将 LlamaIndex 的模型包装为 Ragas 可识别的格式
    ragas_llm = LlamaIndexLLMWrapper(judge_llm)
    ragas_emb = LlamaIndexEmbeddingsWrapper(judge_embedding)

    # 3. 初始化待测系统的 RAG 管线 (使用单次查询引擎，防止测试题之间上下文污染)
    print("⚙️ [Pipeline] 准备待测系统引擎...")
    index = get_index()
    retriever = get_retriever(index)
    
    # 使用基础生成模型 (如 7B) 作为答题选手
    answer_llm = Ollama(
        model=GLOBAL_CONFIG["llm"]["weak_model"], 
        base_url=base_url, 
        temperature=0.0
    )
    
    # 构建 Query Engine
    query_engine = RetrieverQueryEngine.from_args(
        retriever,
        llm=answer_llm,
        text_qa_template=PromptTemplate(QA_PROMPT_TEMPLATE)
    )

    # 4. 加载测试集并生成回答
    print(f"📂 [Dataset] 加载测试集: {test_set_path}")
    test_cases = load_test_set(test_set_path)
    
    data_dict = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    print("🤖 [Running] 开始生成测试题答案...")
    for idx, case in enumerate(test_cases, 1):
        q = case["question"]
        gt = case.get("ground_truth", "")
        print(f"  [{idx}/{len(test_cases)}] 正在回答: {q}")
        
        # 运行问答引擎
        response = query_engine.query(q)
        
        # 提取上下文 (contexts) 文本列表
        contexts = [node.get_content() for node in response.source_nodes]
        
        # 填充评估数据集
        data_dict["question"].append(q)
        data_dict["answer"].append(response.response)
        data_dict["contexts"].append(contexts)
        data_dict["ground_truth"].append(gt)

    # 转换为 HuggingFace Dataset 格式，供 Ragas 消费
    dataset = Dataset.from_dict(data_dict)

    # 5. 执行 Ragas 评估
    print("\n⚖️ [Evaluation] 裁判 72B 正在打分 (这一步需要大量算力，请耐心等待)...")
    # 稳定性优先：显式注入本地裁判模型，并降低 answer_relevancy 的多生成采样压力
    metrics = [
        ContextPrecision(llm=ragas_llm),                       # 检索精度 (相关上下文是否排在前面)
        Faithfulness(llm=ragas_llm),                           # 忠实度 (回答是否有幻觉，是否完全基于上下文)
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb, strictness=1),  # 回答相关性
    ]
    
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_emb,
        run_config=RunConfig(timeout=600, max_retries=3, max_workers=1),
        batch_size=1,
        raise_exceptions=False # 防止个别题目打分失败导致整个流程崩溃
    )

    # 6. 动态生成实验报告名称与路径
    strategy = GLOBAL_CONFIG["chunking"].get("strategy", "fixed")
    is_phase2 = (strategy == "semantic")
    
    phase_name = "Phase 2A (Advanced Semantic + RAPTOR)" if is_phase2 else "Phase 1 (Fixed-256 Baseline)"
    report_filename = "phase2_evaluation_report.csv" if is_phase2 else "phase1_evaluation_report.csv"
    report_path = f"data/{report_filename}"

    print("\n" + "="*50)
    print(f"🏆 {phase_name} 评估报告出炉")
    print("="*50)
    print(result)
    
    # 导出到 CSV 供进一步分析，确保不同 Phase 的消融实验数据不被覆盖
    df = result.to_pandas()
    df.to_csv(report_path, index=False)
    print(f"\n📁 详细评分明细已保存至: {report_path}")
    print("="*50)


if __name__ == "__main__":
    try:
        run_evaluation()
    except Exception as e:
        print(f"\n❌ 评估脚本运行失败: {str(e)}")
        print("请检查:")
        print("1. Ollama 服务是否启动，且已下载 72B 模型。")
        print("2. data/test_set.json 是否存在。")
        print("3. 是否已运行过 indexer.py 存入至少一篇研报。")
