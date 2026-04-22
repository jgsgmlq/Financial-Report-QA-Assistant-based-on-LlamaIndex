# 📊 中期项目进展报告：消融实验与评估操作规范 (Progress Report Guide)

> **文档目的**：提供一份可直接照做的实验手册，指导你在本地完成 Phase 1（基线）与 Phase 2A（高级架构）的对比实验，稳定产出报告需要的日志、截图和量化 CSV。

---

## 0) 统一约定（先看）

- 所有命令默认在**项目根目录**执行。
- 默认 shell 为 `zsh`，默认 Python 虚拟环境为 `.venv`。
- 中期报告正式数据必须使用 **20 题**测试集（5 题仅用于本地冒烟，不可用于最终对比结论）。
- 两轮实验必须使用**同一批 PDF**与**同一份测试集**，只改变配置变量。
- 若你需要先快速验证流程，可使用“**单 PDF 极速模式**”（见第 0.1 节），但在正式提交中需明确说明实验范围与局限性。

---

## 0.1) 单 PDF 极速模式（可选，用于赶进度）

> 适用场景：时间紧、先跑通流程、先产出一版对比结果。  
> 本模式核心是：`data/` 根目录只保留 1 份 PDF，显著缩短索引构建与调试时间。

### 快速操作步骤

1. 在 `data/` 下临时收纳其它 PDF（不删除）：

```bash
mkdir -p data/_pdf_hold
mv data/H301_AP202604201821330558_1.pdf data/_pdf_hold/ 2>/dev/null
mv data/H301_AP202604211821369664_1.pdf data/_pdf_hold/ 2>/dev/null
ls data/*.pdf
```

2. 确认仅保留目标文件：
   - `data/H301_AP202604211821369599_1.pdf`

3. 完成实验后如需恢复多文档场景：

```bash
mv data/_pdf_hold/*.pdf data/
```

### 报告写作提示（必须写）

- 本次为**单文档实验**，用于快速验证 Phase 1 与 Phase 2A 的相对差异。  
- 结论对多研报场景的外推能力有限，后续建议补充多文档复现实验。

---

## 1) 环境准备与核查（Prerequisites）

### 1.1 数据与测试集

1. 将 1-3 份真实金融研报 PDF 放入 `data/`。  
2. 编辑 `data/test_set.json`，准备 20 个测试题（遵循 `docs/02_evaluation_metrics.md`）。

### 1.2 本地模型与服务

确保 Ollama 服务已启动，并已拉取模型：

```bash
ollama pull qwen2.5:7b
ollama pull qwen2.5:72b
ollama list
```

### 1.3 Python 环境

```bash
source .venv/bin/activate
python -V
pip -V
```

### 1.4 启动前快速检查命令（建议截图）

```bash
pwd
ls data
python -m src.utils.config
```

📸 **截图 S0（环境截图）建议包含**：
- 当前工作目录
- `data/` 下 PDF 文件列表
- 配置摘要输出（尤其 `chunking.strategy`、`raptor.use_raptor`、`evaluation.judge_model`）

---

## 2) 实验一：Phase 1 基线（Fixed-256）

目标：得到基线性能与基线报告文件 `data/phase1_evaluation_report.csv`。

### Step 1：注入基线配置

编辑 `configs/config.yaml`，确认关键项如下：

```yaml
chunking:
  strategy: "fixed"
  chunk_size: 256
  chunk_overlap: 25

raptor:
  use_raptor: false

storage:
  chroma_persist_dir: "./chroma_db/phase1_baseline"

evaluation:
  test_set_path: "./data/test_set.json"
  judge_model: "qwen2.5:72b"
```

📸 **截图 P1-1（配置截图）**：
- `chunking` / `raptor` / `storage.chroma_persist_dir` / `evaluation.judge_model`

### Step 2：执行基线索引构建

```bash
export PYTHONPATH=.
source .venv/bin/activate
python src/ingest/indexer.py
```

📸 **截图 P1-2（索引构建日志）建议包含**：
- `[Step 1] 解析原始研报 (PDF Parser)`
- `[Step 2 & 3] ... (Indexer & Chunker)` 关键日志
- 最后的构建成功提示

### Step 3：执行基线评估

```bash
export PYTHONPATH=.
source .venv/bin/activate
python src/evaluation/ragas_eval.py
```

⏳ 20 题 + 72B 裁判通常耗时较长（15-30 分钟或更久）。

📸 **截图 P1-3（评估日志）建议包含**：
- 裁判模型启动信息（`qwen2.5:72b`）
- 评估结束汇总输出（含指标）
- `phase1_evaluation_report.csv` 保存路径日志

### Step 4：基线产物核对与归档

```bash
ls -lh data/phase1_evaluation_report.csv
```

建议备份一份防误操作：

```bash
cp data/phase1_evaluation_report.csv data/phase1_evaluation_report.backup.csv
```

---

## 3) 实验二：Phase 2A 高级架构（Semantic + RAPTOR）

目标：在同一数据和测试集下，生成 `data/phase2_evaluation_report.csv` 用于与基线对比。

### Step 1：注入高级配置

编辑 `configs/config.yaml`，仅修改实验变量（其余保持一致）：

```yaml
chunking:
  strategy: "semantic"

raptor:
  use_raptor: true
  summary_model: "qwen2.5:7b"

storage:
  chroma_persist_dir: "./chroma_db/phase2_raptor"

evaluation:
  test_set_path: "./data/test_set.json"
  judge_model: "qwen2.5:72b"
```

📸 **截图 P2-1（配置截图）**：
- `chunking.strategy=semantic`
- `raptor.use_raptor=true`
- `storage.chroma_persist_dir=./chroma_db/phase2_raptor`

### Step 2：执行高级索引构建

```bash
export PYTHONPATH=.
source .venv/bin/activate
python src/ingest/indexer.py
```

📸 **截图 P2-2（高级构建日志）建议包含**：
- `[Step 2 & 3] 语义向量化与树状摘要构建`
- RAPTOR 相关日志（如摘要构建、入库完成）

📸 **截图 P2-3（性能开销截图，可选但强烈建议）**：
- Mac 活动监视器中的 CPU/内存峰值
- 与终端构建日志同屏或并排展示

### Step 3：执行高级评估

```bash
export PYTHONPATH=.
source .venv/bin/activate
python src/evaluation/ragas_eval.py
```

📸 **截图 P2-4（评估日志）建议包含**：
- 裁判模型与评估启动信息
- 评估完成摘要
- `phase2_evaluation_report.csv` 保存路径日志

### Step 4：高级产物核对

```bash
ls -lh data/phase2_evaluation_report.csv
```

---

## 4) 对比分析与报告填表

1. 打开两份结果文件：
   - `data/phase1_evaluation_report.csv`
   - `data/phase2_evaluation_report.csv`
2. 对比核心指标（建议至少）：
   - `context_precision`
   - `faithfulness`
   - `answer_relevancy`
3. 将结果回填到 `docs/02_evaluation_metrics.md` 的实验记录表。

建议在报告中额外补充：
- 同一问题在 Phase 1 与 Phase 2A 的检索上下文差异示例（1-2 题）
- 引用溯源质量变化（页码命中/引用可点击验证）
- 若采用单 PDF 极速模式：补充“实验范围说明与局限性”小节（单文档 vs 多文档）。

---

## 5) 最终交付清单（Checklist）

- [ ] 环境核查截图（S0）
- [ ] Phase 1 配置截图（P1-1）
- [ ] Phase 1 索引日志截图（P1-2）
- [ ] Phase 1 评估日志截图（P1-3）
- [ ] `data/phase1_evaluation_report.csv`
- [ ] Phase 2A 配置截图（P2-1）
- [ ] Phase 2A 构建日志截图（P2-2）
- [ ] 性能开销截图（P2-3，可选但建议）
- [ ] Phase 2A 评估日志截图（P2-4）
- [ ] `data/phase2_evaluation_report.csv`
- [ ] `docs/02_evaluation_metrics.md` 回填完成
- [ ] （若使用单 PDF 极速模式）报告中已明确写明单文档实验范围与局限性

---

## 6) 常见问题与排查（FAQ）

- **Q1：提示找不到模型？**  
  先执行 `ollama list`，确认 `qwen2.5:7b`、`qwen2.5:72b` 已下载。

- **Q2：提示找不到测试集？**  
  检查 `evaluation.test_set_path` 是否为 `./data/test_set.json`，并确认文件存在且 JSON 格式合法。

- **Q3：Phase2 结果覆盖了 Phase1？**  
  当前脚本按 `chunking.strategy` 自动区分 `phase1_evaluation_report.csv` 与 `phase2_evaluation_report.csv`。若仍担心，按本指南 Step 4 先手动备份。

- **Q4：72B 太慢或跑不动？**  
  可先用 7B 做冒烟验证，再按课程文档降级策略执行并在报告里明确说明。

- **Q5：`zsh: command not found: python`？**  
  说明当前终端未激活虚拟环境。先执行 `source .venv/bin/activate`，再运行本指南中的 `python ...` 命令；或临时改用 `python3`。