# Insight · 机构级金融研报智能工作台

<p align="center">
  <img src="https://img.shields.io/badge/Status-Beta-brightgreen" alt="Status" />
  <img src="https://img.shields.io/badge/LLM-Qwen2.5--7B-blue" alt="LLM" />
  <img src="https://img.shields.io/badge/Framework-LlamaIndex-purple" alt="Framework" />
</p>

**Insight** 是对标 **NotebookLM** 体验的 RAG（检索增强生成）分析工作台，面向复杂排版的金融研报（PDF）。从底层解析、索引到前台交互均针对研报场景优化：不仅定位原文，还能理解表格、提炼宏观观点并合成可用知识。

---

## 核心特性

### 数据底座：深度解析与索引

| 能力 | 说明 |
|------|------|
| **云端解析 (LlamaParse)** | 穿透复杂 PDF，稳定抽取财务表格、双栏正文，输出高保真 Markdown。 |
| **语义分块** | 基于 BGE 句向量切分语义边界，避免生硬按字数截断，保持段落与表格逻辑完整。 |
| **混合检索 (Hybrid RRF)** | `BM25` 关键词 + `bge-m3` 语义双路召回，RRF（倒数排序融合）合并结果。 |
| **二次精排** | `bge-reranker-base` 降噪，将高质量证据送入 LLM。 |

### 认知升级：宏观视角（RAPTOR）

入库时自底向上构建层次摘要树（Level 1～3），细粒度事实与行业级趋势均可被稳定检索。

### 沉浸式工作台（类 NotebookLM）

| 能力 | 说明 |
|------|------|
| **三分栏** | 左侧数据、中间对话与灵感区、右侧原生 PDF 阅读器。 |
| **证据链联动** | 生成内容带引用卡片；点击引用，PDF 预览**跳转至对应页码**。 |
| **跨文档合成** | 一键生成多文档对比表，并可写入「分析师灵感库 (Notepad)」，便于拼装正式报告。 |

---

## 快速启动与配置

以下步骤默认你在**项目根目录**（含 `configs/`、`src/`、`requirements.txt` 的目录）操作。配置文件与 `.env` 均相对该目录加载；若从其他路径启动，可能导致找不到 `configs/config.yaml`。

### 硬件与环境

| 档位 | 建议配置 |
|------|----------|
| **最低** | Apple Silicon (M1/M2) 或 x86_64 CPU，**16GB** 内存。可跑 7B 量化模型，速度较慢，不适合重度跨文档分析。 |
| **推荐** | Apple Silicon (M1/M2/M3 Pro/Max) **≥32GB** 统一内存；或 **NVIDIA GPU**（如 RTX 3060/4060，**≥12GB** 显存）的 Windows/Linux 主机。 |

- **Python**：建议 **3.10+**（与当前依赖兼容）。
- **网络**：首次运行会从 Hugging Face 拉取 `embedding` / `reranker` 等模型，需能访问外网。

### 步骤 1：Python 虚拟环境与依赖

```bash
cd /path/to/CS6496-group

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 步骤 2：Ollama（本地 LLM）

1. 安装并启动 [Ollama](https://ollama.com/)（默认服务地址 `http://localhost:11434`）。
2. 拉取与配置一致的对话模型（须与 `configs/config.yaml` 中 `llm.strong_model` / `weak_model` 一致，默认如下）：

```bash
ollama pull qwen2.5:7b
```

3. 可选自检：`ollama list` 中能看到 `qwen2.5:7b`。若你修改了 YAML 中的模型名，请 `pull` 同名模型。

### 步骤 3：环境变量 `.env`（LlamaParse）

在项目根目录创建或编辑 `.env`，填入 [LlamaCloud API Key](https://cloud.llamaindex.ai/)：

```env
LLAMA_CLOUD_API_KEY=llx-你的密钥
```

- **无 API Key**：在 `configs/config.yaml` 中将 `parser.use_llamaparse` 设为 `false`，将使用本地基础解析（`pymupdf` 等）；复杂报表版式可能不如云端解析稳定。
- **有 API Key**：保持 `parser.use_llamaparse: true`（默认），即可获得高保真表格与双栏解析。

### 步骤 4：`configs/config.yaml`（常用项）

| 配置项 | 作用 |
|--------|------|
| `llm.ollama_base_url` | Ollama 地址，本机默认 `http://localhost:11434`。 |
| `llm.strong_model` / `weak_model` | 须与 `ollama pull` 的模型名一致（默认 `qwen2.5:7b`）。 |
| `embedding.device` | Apple Silicon 用 `mps`；NVIDIA 用 `cuda`；无 GPU 可改为 `cpu`（会较慢）。 |
| `parser.use_llamaparse` | 为 `true` 时需配置 `LLAMA_CLOUD_API_KEY`；否则请改为 `false`。 |
| `storage.chroma_persist_dir` / `data_dir` | 向量库与业务数据目录，默认 `./chroma_db`、`./data`。 |

其余项（分块策略、RAPTOR、检索 `top_k` 等）可参考 [技术规格说明](docs/specs/technical_specification.md) 与 `docs/technical_design.md`。

### 步骤 5：启动 Web 工作台

```bash
# 确认已激活虚拟环境，且当前目录为项目根目录
python -m src.ui.app
```

浏览器打开：**http://localhost:7860**

### 启动前检查清单

| 检查项 | 说明 |
|--------|------|
| 工作目录 | 运行 `python -m src.ui.app` 时 cwd 为项目根目录。 |
| Ollama | 服务已启动，且已 `pull` 与 YAML 一致的模型。 |
| 解析 | 使用 LlamaParse 时已配置 `.env`；否则已将 `use_llamaparse` 设为 `false`。 |
| 设备 | `embedding.device` 与当前机器匹配（`mps` / `cuda` / `cpu`）。 |

---

## 文档与评估

- **技术细节**（RAPTOR、检索链路、数据流等）：[技术规格说明](docs/specs/technical_specification.md)

项目集成 **Ragas** 与基于正则的 **Citation Audit**。运行基准评估：

```bash
python -m src.evaluation.ragas_eval
```
