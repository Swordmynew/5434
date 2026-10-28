# COMP5434 问题 2：评论评分预测

本项目实现 Spark 并行数据处理、自定义文本评分模型、自定义层级用户商品关系模型，以及基于校准集的融合。输出连续的 1–5 分评分，以 RMSE 评价。

项目针对 **Ubuntu 22.04 / Python 3.12 / PyTorch 2.5.1 / CUDA 12.4 / 单张 RTX 3090 24 GB / 14 vCPU / 60 GB 内存 / 50 GB 数据盘** 配置。PyTorch 使用服务器镜像已有版本，安装项目依赖不会主动替换它。

当前代码不附带全量训练权重。合成数据的测试结果只证明流程可运行，不能作为比赛准确率或服务器速度。真实指标由服务器运行后自动生成。

## 1. 服务器安装与磁盘安排

把本项目代码及 `comp-5434-20261-project-task-2.zip` 上传到数据盘，例如 `/root/autodl-tmp/5434_pro1`。不要把训练缓存放在 30 GB 系统盘中。

```bash
cd /root/autodl-tmp/5434_pro1
mkdir -p .tmp .hf-cache runs
export TMPDIR="$PWD/.tmp"
export SPARK_LOCAL_DIRS="$PWD/.tmp/spark"
mkdir -p "$SPARK_LOCAL_DIRS"
export HF_HOME="$PWD/.hf-cache"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYSPARK_PYTHON="$(command -v python)"
export PYSPARK_DRIVER_PYTHON="$PYSPARK_PYTHON"

# 先确认镜像中的 PyTorch 和 CUDA 工作正常
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"

# Spark 使用 Java 17；如果镜像尚未安装 Java，以 root 执行：
apt-get update
apt-get install -y openjdk-17-jdk-headless
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"

python -m pip install --no-cache-dir -e '.[test]'
python -m task2 doctor
```

请在以后每次登录时重新设置这些环境变量，或把它们加入自己的启动脚本。程序启动前检查数据盘可用空间，默认至少保留 20 GiB。Spark 中间文件可通过 `SPARK_LOCAL_DIRS` 指向数据盘。

存储包括原始 CSV、清洗 Parquet、特征 Parquet、压缩 token 缓存和模型工件。仅保留每个模型的最佳权重和最近一次可续训检查点；不会保存每个 epoch 的大权重。多个试验会增加占用，应按 `du -sh runs/* .hf-cache data` 检查。程序不会自动删除原始数据或模型。

## 2. 先验证小数据流程

```bash
python -m pytest -q -m 'not spark'
python -m task2 fixture --output runs/smoke
python -m task2 --config runs/smoke/config.json run
python -m pytest -q -m spark
```

合成数据包含逗号、引号、真实换行、HTML、空正文、重复文本及未知实体。其 tokenizer 和小型 BERT 从零初始化，不下载预训练权重。样例提交的 ID 顺序特意与测试数据相反，用于检查输出按样例顺序对齐。

默认正式数据划分为训练部分、50,000 条校准集和 50,000 条独立验证集。小数据配置将后两者改为各 50 条。训练 ID、split、fold 保存在 `clean/` 和特征工件中。

## 3. 正式运行

先在 Kaggle 竞赛规则中确认是否允许通用预训练权重。指导允许第三方库，但本项目没有把它等同于竞赛对外部权重的授权；默认配置不会下载权重。

正式实验推荐使用单机 Standalone：一个 Master、一个 8 核 Worker，按照每 Executor 4 核分配两个执行进程。

```bash
bash scripts/spark_cluster.sh

# 已确认允许通用预训练权重时运行
python -m task2 \
  --set spark.master=spark://127.0.0.1:7077 \
  --set text.pretrained_allowed=true \
  run
```

完成安装后，也可用 `bash scripts/run_server.sh --pretrained-approved` 一次设置缓存路径、检查 GPU、启动集群并执行流程。长时间训练可在项目根目录执行 `mkdir -p runs` 后，用 `nohup bash scripts/run_server.sh --pretrained-approved > runs/training.log 2>&1 &` 保持后台运行；查看该日志获取进度。从零训练对应 `--scratch`。

若规则禁止外部预训练权重，可采用同一流程、从项目训练文本学习词表并从零训练文本编码器：

```bash
python -m task2 \
  --set spark.master=spark://127.0.0.1:7077 \
  --set text.mode=scratch \
  run
```

开发或排查问题时默认 `local[8]` 也能运行 Spark 多线程任务。正式报告应如实填写采用的模式；不要把单机并行或伪分布式实验描述为多台物理服务器。

集群日志与 PID 位于 `runs/cluster/`。完成实验后先查看 PID 文件和相应进程，再结束自己启动的 Worker 与 Master；脚本不会批量结束其他 Java 进程。

### 逐步执行与恢复

全流程 `run` 适用于全新 `run_dir`。已有准备工件时，按阶段继续，不要重跑 `prepare` 覆盖划分。

```bash
python -m task2 --set text.pretrained_allowed=true prepare
python -m task2 train ridge
python -m task2 train relation
python -m task2 --set text.pretrained_allowed=true pilot

# pilot 产生的配置包含选定损失和根据吞吐估算的训练规模
python -m task2 --config runs/main/selected_config.json train text
python -m task2 --config runs/main/selected_config.json train text --resume
python -m task2 --config runs/main/selected_config.json evaluate --models mean ridge text relation
python -m task2 --config runs/main/selected_config.json predict
python -m task2 --config runs/main/selected_config.json report
```

如果采用 Standalone，逐步命令也要传入相同的 `spark.master` 覆盖，或直接把它改到自己的配置中。配置可复制后修改，命令行 `--set a.b=value` 的覆盖优先。

`last.pt` 包括模型、优化器、AMP scaler、随机数状态、epoch、批次位置和已用时间。`--resume` 从最近持久化检查点重放确定的数据顺序；不能改变随机种子、批大小或数据加载 worker 数。检查点仅加载本项目自己生成的可信工件。

若首次启动发生 CUDA OOM，程序降低单批大小并相应增加梯度累积。若已有检查点后才发生 OOM，保持已有工件，按下方故障说明处理。

## 4. 模型与自主实现

### 文本模型

`task2/models.py` 使用 DeBERTa 编码器，但评分头、池化、训练循环、损失与预算控制由本项目实现。输入采用 `[CLS] 标题 [SEP] 正文 [SEP]`；标题最多 32 tokens，总长度 192，正文超长时保留约 2/3 开头和 1/3 结尾。

把 CLS 表示与正文掩码均值拼接，通过 MLP 输出五个概率。预测为 `sum(k * p[k])`，损失为 MSE 加 `lambda * CE`。空正文使用 CLS 作为池化回退。pilot 在相同训练采样上比较 `lambda=0` 与 `0.2`，仅按校准 RMSE 选择。

### 关系模型

自主实现 `mu + b_user + b_product + b_parent + dot(user, product + parent) + MLP(metadata)`。嵌入维度默认 32。每个实体有稳定编号；0 表示未知实体，其偏置和向量为零。

实体统计包括评论数、平滑评分均值和五类评分分布。训练记录用五折交叉统计：从实体总统计减去整折统计，先验同样由其余折计算。校准、验证和测试均使用训练部分的完整历史，绝不使用它们自己的标签。

Spark 通过分布式排序与编号生成实体映射。整个训练集、用户词表或商品词表不会作为 Python 列表收集到 driver。

### 基线与融合

Spark ML 的 HashingTF、IDF、Normalizer 和 Ridge 用于对照实验，标题与正文的一元和二元词组使用独立前缀。其词频统计与回归均只在训练部分拟合。这个对照的现成模型调用不是项目的自主算法主体。

融合通过自行实现的单纯形投影梯度算法拟合非负且和为 1 的权重。所有预测先按字符串 ID 对齐。校准集拟合权重，独立验证集进行一次融合与最佳单模的接受比较。不要反复根据独立验证结果试参，否则它会失去独立性。

## 5. 预算与增强实验

默认主文本模型上限 36 小时，关系模型上限 4 小时，pilot 每个候选上限 1.5 小时。CUDA 训练阶段累计最多使用 72 小时中的 64 小时，另外 8 小时留给最终推理和复现。日志记录实际耗时，校准及检查点操作也会消耗预算；单次校准/保存的执行时间可能略超阶段停止点。

pilot 根据实测速率估计全量 epoch 时长；若预算不足，会把主模型改为确定的 ID 哈希采样。采样量是期望值，实际处理条数写入 epoch 日志。只有完整结束的轮次才标记为 `complete_epoch=true`。后续 epoch 预计超时则不启动。

需要增强时优先增加一个不同随机种子，使用同一份冻结的数据划分和 token 缓存。先检查剩余预算和磁盘：

```bash
python -m task2 --config runs/main/selected_config.json \
  --set seed=5435 --set text.max_hours=12 --set text.max_train_rows=1000000 \
  train text --name text_seed2
python -m task2 --config runs/main/selected_config.json \
  evaluate --models mean ridge text relation text_seed2
```

为避免反复查看独立验证集，上述增强实验应预先确定，再统一执行最终 `evaluate`。

## 6. 消融、重复文本与效率实验

关系模型消融可共享已经冻结的特征工件。代码会同时屏蔽实体嵌入、偏置和对应统计字段，而不是只去掉一个 ID。

```bash
python -m task2 --set relation.drop_user=true train relation --name relation_no_user
python -m task2 --set relation.drop_parent=true train relation --name relation_no_parent
python -m task2 --set baseline.drop_title=true train ridge --name ridge_no_title
```

去掉文本主模型标题会改变 token 缓存，需使用独立 run。较低成本的标题消融可先使用 `ridge_no_title`。已有 token 缓存的格式参数改变时程序会报错，防止误用旧缓存。

按相同文本分组的诊断使用独立目录，允许样本量近似目标值，但组间绝不交叉：

```bash
python -m task2 --set run_dir=runs/grouped --set split.strategy=group prepare
python -m task2 --set run_dir=runs/grouped --set split.strategy=group train ridge
python -m task2 --set run_dir=runs/grouped --set split.strategy=group train relation
python -m task2 --set run_dir=runs/grouped --set split.strategy=group evaluate --models mean ridge relation

python -m task2 benchmark
python -m task2 report
```

分组实验会另存清洗和特征文件。50 GB 数据盘运行前检查剩余空间，避免同时保留多个大型诊断 run。基于主验证集的重复文本分层分析无需建立第二份完整 run，结果已在 `evaluation.json` 中。

benchmark 在相同 Parquet 上比较 1、4、8 核的读取与历史聚合；无 Spark 缓存，操作系统页缓存不受控，因此不能把该结果冒充整个流程的严格线性加速比。若需要正式稳定数字，独立重复三次并报告中位数及缓存条件。

## 7. 工件、计时与复现

`runs/main/` 的主要文件：

| 工件 | 内容 |
|---|---|
| `manifest.json` | 数据规模、准备时配置、训练均值、实体大小、特征名 |
| `clean/`、`history/`、`features/` | 固定划分、折外统计、实体映射、模型输入 |
| `tokenizer/`、`tokens/` | 保存的 tokenizer 与压缩分词缓存 |
| `models/*/best.pt`、`last.pt` | 最佳预测权重、最近可恢复训练状态 |
| `models/*/epoch_*.json` | 完整或部分轮次、实际条数、训练耗时及吞吐 |
| `predictions/*/*.csv` | 每个组件在校准、验证、测试上的预测 |
| `evaluation.json`、`metrics.csv` | 权重、整体 RMSE、重复内容和实体频次分层指标 |
| `hardware.json`、`timings.jsonl` | 硬件及阶段时间日志 |
| `submission.csv` | 按样例顺序生成的连续评分 |
| `results.md` | 根据实际实验结果生成的报告素材 |

`run` 记录一次 `total_offline`，从原始 CSV 准备开始，包括分词、训练、校准与组件预测，属于包含额外评估开销的离线总时间。`predict` 单独测量原始测试 CSV → 特征 → 加载所有选中模型 → 推理 → 融合 → CSV 的完整时间。训练轮次时间排除中途校准耗时，并区分中断后的轮次片段。

提交前再次运行 `predict --output runs/main/submission_reproduced.csv`，比较文件。若模型目录移动到其他路径，保留目录结构并修改 `run_dir` 与 `raw_dir`。重推理使用本地保存的 tokenizer、编码器配置及权重，不需要再下载预训练模型。

固定随机种子、worker 数和依赖；Spark 的并行浮点归约及不同 GPU 库版本仍可能出现末位差异。保留已生成的历史特征和模型工件用于精确重推理，在相同环境对浮点结果以合理容差核验。`python -m pip freeze > runs/main/environment.txt` 保存完整依赖清单。

## 8. 故障处理

- **预训练权重未授权**：确认竞赛规则后设置 `text.pretrained_allowed=true`，或使用 `text.mode=scratch`。默认不会自动接受外部权重。
- **首次 CUDA OOM**：自动减小 batch 并增加 accumulation。若已经有续训检查点，不能直接改变批次配置继续重放；新建模型名、调整 batch 后重新训练，原工件保留。
- **Spark 找不到 Python**：在同一环境设置 `PYSPARK_PYTHON`，并保证 Worker 能访问同一项目及 Python 环境。
- **Java 版本错误**：Spark 3.5 配合 Java 17；不要使用系统中偶然找到的 Java 25。
- **Windows Spark 文件系统错误**：本项目正式运行目标为 Ubuntu。Windows 可能额外需要 Hadoop 原生组件；可在 Linux/WSL 中运行 Spark 集成测试，算法测试无需该组件。
- **磁盘不足**：检查缓存和诊断 run；在确认不再需要后自行归档。不要把 `.venv`、缓存或原始大文件打进最终代码压缩包。

报告结构和展示素材见 `docs/REPORT_GUIDE.md`。最终提交应包含整个项目要求的队名、姓名、学号和合计 100% 的贡献表。

## References

- [Spark CSV parsing](https://spark.apache.org/docs/3.5.8/sql-data-sources-csv.html)
- [Spark Standalone](https://spark.apache.org/docs/3.5.8/spark-standalone.html)
- [Microsoft DeBERTa-v3-base](https://huggingface.co/microsoft/deberta-v3-base)
- [DeBERTaV3 paper](https://arxiv.org/abs/2111.09543)
- [PyTorch mixed precision](https://pytorch.org/docs/stable/notes/amp_examples.html)
