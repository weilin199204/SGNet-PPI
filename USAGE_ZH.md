# SGNet-SR 中文使用指南

本目录是论文中 **SGNet-SR**（内部实验名为 Surface Residual V2）的投稿代码包。
代码仓库统一使用 `SGNet` 名称。它不包含历史 V1/V29 实验，也不包含
早期 1618/1741 样本、随机五折或 40% identity CV 结果，避免把开发阶段结果混入
论文协议。
模型下载：
model/real/best_surface_residual_v2.pt: https://drive.google.com/file/d/1IG_XOs18qixmD5IT-SvKnzYEIHjq68ID/view?usp=sharing
bmodel/est_sequence.pt: https://drive.google.com/file/d/1BEgzF-VqIqlviSkoPXSHPjGY6X58qtrZ/view?usp=sharing
## 一、投稿时应上传什么

建议把整个 `SGNet_github/` 作为独立仓库上传，其中最核心的是：

| 类别 | 文件 |
|---|---|
| 最终模型 | `sgnet/sequence.py`、`sgnet/model.py`、`sgnet/data.py` |
| 训练 | `scripts/train_sequence.py`、`scripts/train_sgnet.py` |
| 构图 | `sgnet_graph/interface_patch.py`、`scripts/build_interface_patch_graphs.py` |
| 预处理 | ESM、MSMS/MaSIF、packing 脚本 |
| 评估 | `scripts/evaluate_s166.py`、`scripts/analyze_results.py` |
| 协议 | `protocol/splits/`、`protocol/metadata/` |
| 权重 | `model/` 下 sequence、real 两个部署 checkpoint |
| 结果 | `results_snapshot/` 下逐样本预测和 OOF 预测 |
| 特征定义 | `protocol/results/interface_patch_full_metadata.json` |

不要上传以下内容：

- PDBbind、Affinity Benchmark、SKEMPI 原始结构；
- MaSIF/MSMS/APBS 的第三方二进制；
- `work/surfaces` 下的大型 PLY；
- `work/training_pkls` 下约 7 GB 的 pickle；
- Slurm 日志、`__pycache__`、临时文件和开发期历史结果；
- 带有 `/home/用户名/...` 的本机路径。

## 二、两种复现层级

### 层级 A：复核论文结果

这是审稿人最容易使用的入口，不需要原始数据、GPU 或表面软件。

```bash
cd SGNet_github
python scripts/analyze_results.py \
  --snapshot-root results_snapshot \
  --output-dir reproduced_results \
  --seed 20 \
  --sample-bootstrap 50000 \
  --cluster-bootstrap 20000
```

生成：

- `reproduced_results/main_metrics.csv`：MAE、MSE、Pearson、Spearman；
- `reproduced_results/statistical_checks.csv`：配对 bootstrap 区间；
- `reproduced_results/analysis_protocol.json`：随机种子和重采样设置。

validation、test1、test2 按样本配对重采样。S166 必须按
`base_pdb_id` 聚类重采样，因为 166 条记录实际只有 26 个 base complexes。
脚本同时输出排除 `1YCS` 后的敏感性分析。

点指标应与 `protocol/results/main_metrics.csv` 在舍入误差内一致。bootstrap
端点是 Monte Carlo 估计，重新固定脚本后的端点可能与旧归档表有轻微差异。

### 层级 B：从原始结构重建并重训

这需要：

- 获得许可的 PDBbind v2020、Affinity Benchmark v5.5 和 SKEMPI v2.0；
- `facebook/esm2_t33_650M_UR50D`；
- MaSIF、MSMS 2.6.1、APBS、Reduce 和网格依赖；
- 支持 PyTorch/PyG 的 GPU 环境；
- 足够磁盘空间保存 ESM、PLY、patch graph 和 packed pickle。

流程为：

```text
冻结 manifest
  -> 标准化 PDB 视图
  -> ESM2 residue tensors
  -> MaSIF/MSMS 完整表面
  -> interface_patch_full v5
  -> 固定 split packing
  -> Sequence baseline
  -> 5-fold OOF residual
  -> SGNet-SR real + shuffled
  -> test1/test2/S166 + bootstrap
```

## 三、安装

参考实验环境：

```text
Python             3.8.20
PyTorch            2.2.0
CUDA runtime       11.8
NumPy              1.24.4
SciPy              1.10.1
scikit-learn       1.3.2
Transformers       4.38.0
PyTorch Geometric  2.6.1
torch-scatter      2.1.2
```

示例：

```bash
conda create -n sgnet python=3.8 -y
conda activate sgnet
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter==2.1.2 \
  -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
pip install -r requirements.txt
```

先执行快速检查：

```bash
python scripts/smoke_test.py
```

成功时应显示：

```text
smoke_ok=True
cross_edge_only=True
zero_initialization_exact=True
pretrained_checkpoints_loaded=2
```

## 四、准备固定 manifest

`protocol/metadata/*.tsv` 中的 `structure_path` 已去除本机路径，使用以下 token：

- `PDBBIND_ROOT`
- `AFFINITY_BENCHMARK_ROOT`
- `SKEMPI_ROOT`
- `REPLACEMENT_ROOT`

将 token 解析为本机目录：

```bash
python scripts/resolve_manifest_paths.py \
  --manifest protocol/metadata/all.tsv \
  --pdbbind-dir "/data/PDBbind v2020" \
  --affinity-benchmark-dir "/data/Affinity Benchmark v5.5" \
  --replacement-dir "/data/replacement_structures" \
  --out work/manifests/all.tsv \
  --check

python scripts/audit_manifest.py \
  --manifest work/manifests/all.tsv \
  --report work/manifests/all.audit.json
```

`replacement_structures` 需要含 `5BXQ.pdb` 和 `7SQ2.pdb`。它们分别替代
训练条目 `1A2K` 和 `3OHM` 的结构，但保留原条目的亲和力标签。

如需从源表重建 manifest，使用 `scripts/build_manifest.py`，所有数据源路径
都必须显式传入。最终必须得到精确的 2350/80/79/81，不能重新随机划分。

## 五、生成 ESM2 表征

```bash
python scripts/prepare_structures.py \
  --manifest work/manifests/all.tsv \
  --out-dir work/raw_pdb \
  --copy

python scripts/build_sequence_artifacts.py \
  --metadata work/manifests/all.tsv \
  --pdb-dir work/raw_pdb \
  --out-dir work/sequence_graphs \
  --device cuda:0 \
  --num-shards 1 \
  --shard-index 0
```

ESM2 型号固定为 `facebook/esm2_t33_650M_UR50D`，每残基 1280 维。长链按
1000 residues 切窗，overlap 100，重叠位置取平均。最终 sequence 模型不使用
residue edge、坐标或 interface mask。

多 GPU/多任务时，设置相同的 `--num-shards N`，分别运行
`--shard-index 0 ... N-1`。

## 六、生成完整分子表面

```bash
python scripts/build_surfaces.py \
  --manifest work/manifests/all.tsv \
  --raw-pdb-dir work/raw_pdb \
  --out-dir work/surfaces \
  --masif-root /opt/masif \
  --num-shards 1 \
  --shard-index 0 \
  --strict

python scripts/audit_surfaces.py \
  --manifest work/manifests/all.tsv \
  --surface-root work/surfaces \
  --out work/surfaces/surface_index.tsv \
  --allow-missing 3
```

关键行为：

- 只使用 manifest 两组 partner chains 的并集；
- 多 MODEL PDB 只取第一个 MODEL；
- 每个任务使用独立 scratch directory；
- MSMS 按冻结 probe radius/density 组合重试；
- APBS 失败会显式留痕并将 electrostatic channel 置零；
- validation/test1/test2 不允许缺失 surface；
- 论文协议仅允许训练集 `5YWO`、`2DSQ`、`5KVE` 使用有标记 placeholder。

## 七、构建 interface_patch_full v5

```bash
python scripts/build_interface_patch_graphs.py \
  --surface-dir work/surfaces/complete_ply \
  --surface-index work/surfaces/surface_index.tsv \
  --atom-root work/sequence_graphs \
  --out-root work/patch_graphs \
  --graph-name interface_patch_full \
  --workers 4 \
  --strict
```

输出目录：

```text
work/patch_graphs/surfgraph_4/interface_patch_full
```

节点为 23 维，边为 42 维。最终 SGNet-SR 只读取 `edge_type == 1` 的跨链边。
精确特征顺序和阈值见
`protocol/results/interface_patch_full_metadata.json`，不要根据 README 手工重排。

## 八、打包固定 split

```bash
python scripts/pack_fixed_splits.py \
  --manifest work/manifests/all.tsv \
  --graph-root work/sequence_graphs \
  --patch-root work/patch_graphs/surfgraph_4/interface_patch_full \
  --out-dir work/training_pkls
```

脚本会检查：

- split 数量是否严格为 2350/80/79/81；
- 四个 split 是否有 PDB ID 重叠；
- label 是否有限；
- ESM tensor 与 residue 数是否一致；
- patch graph 是否为 v5、23/42 维并含跨链边；
- 缺失 surface 是否带显式 `surface_available=False`。

## 九、训练

Sequence baseline：

```bash
python scripts/train_sequence.py \
  --data-root work/training_pkls \
  --output-dir runs/sequence \
  --device cuda:0
```

SGNet-SR 和 shuffled negative control：

```bash
python scripts/train_sgnet.py \
  --data-root work/training_pkls \
  --sequence-checkpoint runs/sequence/best_sequence.pt \
  --output-dir runs/sgnet \
  --device cuda:0 \
  --target-mode both
```

若只训练可部署模型，使用 `--target-mode real`；论文复现必须保留
`--target-mode both`。

若不希望重新训练五个 OOF sequence fold，可复用冻结 OOF 预测：

```bash
python scripts/train_sgnet.py \
  --data-root work/training_pkls \
  --sequence-checkpoint model/best_sequence.pt \
  --output-dir runs/sgnet \
  --device cuda:0 \
  --reuse-oof results_snapshot/oof_sequence_predictions.csv \
  --target-mode both
```

训练约束：

- OOF 使用 `KFold(5, shuffle=True, random_state=20)`；
- OOF 每折固定 30 epoch，held-out label 不参与选 epoch；
- surface normalization 和 placeholder imputation 只使用 train；
- real/shuffled 除 target permutation 外保持结构、初始化和优化设置一致；
- validation 只用于 checkpoint 选择；
- test1/test2/S166 不能用于调参。

## 十、S166

构建 manifest：

```bash
python scripts/build_skempi_manifest.py \
  --ppb-csv /data/PPB-Affinity.csv \
  --pdb-dir "/data/SKEMPI v2.0" \
  --out work/skempi/skempi_s166.tsv
```

生成 mutation-aware ESM：

```bash
python scripts/build_skempi_sequence_artifacts.py \
  --manifest work/skempi/skempi_s166.tsv \
  --pdb-dir "/data/SKEMPI v2.0" \
  --out-dir work/skempi/sequence_graphs \
  --atom-root work/skempi/wt_atom_graphs \
  --device cuda:0
```

26 个 base PDB 的 WT surface 和 patch 使用与主数据相同的构建脚本。打包和评估：

```bash
python scripts/pack_skempi_s166.py \
  --manifest work/skempi/skempi_s166.tsv \
  --graph-root work/skempi/sequence_graphs \
  --patch-root work/skempi/patch_graphs/surfgraph_4/interface_patch_full \
  --out work/skempi/skempi_s166_dataset.pkl

python scripts/evaluate_s166.py \
  --data work/skempi/skempi_s166_dataset.pkl \
  --run-dir model \
  --sequence-checkpoint model/best_sequence.pt \
  --device cuda:0
```

上述命令默认只评估发布的 real 模型。只有在自行训练或另行提供 shuffled
checkpoint 后，才添加 `--include-shuffled`。

必须在论文和代码说明中保留两个限定：

1. 140 个 mutant 使用突变后的 ESM，但共享 base complex 的 WT 实验表面；
2. `1YCS` 与训练集重叠，因此 S166 不是严格零重叠外部集。

## 十一、论文结果应如何表述

允许的结论：

- OOF residual、cross-edge-only encoder 和 reliability gate 使表面分支不再被
  validation 自动关闭；
- test1 Pearson 有显著提高；
- test2 MAE 有显著恶化；
- S166 点估计总体改善，但 26-base cluster interval 跨零；
- 表面信号可检测，但具有数据集依赖性。

不应写：

- “稳定提高所有外部数据集”；
- “显著优于 shuffled control”；
- “S166 有 166 个独立结构”；
- “mutant-specific surface 被显式建模”；
- “最终推理不需要 PDB/MSMS”。

## 十二、发布前必须人工完成

代码目前已去除源码和文档中的作者机器绝对路径，并附带权重、预测和哈希清单。
公开仓库前仍需作者决定并补充：

- `LICENSE`；
- 正式论文题目、作者、DOI 和 `CITATION.cff`；
- 最终运行使用的 MaSIF commit、MSMS/APBS/Reduce 版本；
- 目标数据源的许可与可再分发边界；
- 一个带 tag 的冻结 release 和长期归档 DOI。

详见 `UPLOAD_CHECKLIST.md`。

