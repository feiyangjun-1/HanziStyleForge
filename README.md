[简体中文](README.md) | [한국어](README.ko.md) | [日本語](README.ja.md) | [English](README.en.md)

# HanziStyleForge Fusion

一个面向 Windows 的实验性汉字字体重建工具：从 `target.ttf` 学习字体风格，从 `ref.otf` 获取汉字结构，并生成可安装的 TTF 字体。

> 项目适合长时间自动运行，支持检查点恢复、安全暂停和失败重试。

## 它能做什么

- 从 `fonts/target.ttf` 学习整体与局部字体风格。
- 按 `refs/ref.otf` 的默认字形重建其覆盖的全部汉字。
- 参考字体可以是大陆、台湾、香港、日本、韩国或其他字形标准。
- 尽量保留目标字体中的拉丁字母、数字、符号、假名、谚文及主要 OpenType 数据。
- 自动完成训练、生成、候选筛选、QA、矢量化和字体构建。

## 工作方式

```text
target.ttf：提供风格
        +
ref.otf：提供汉字结构和覆盖范围
        ↓
Style Encoder → VQ → Diffusion → Refiner / Retrieval / IDS
        ↓
候选筛选 → QA → 轮廓转换 → TTF
```

程序不会自行判断哪一种地区字形“更正确”。最终汉字结构以 `ref.otf` 的默认 Unicode `cmap` 字形为准。

## 环境要求

- Windows 11 64 位
- 支持 CUDA 的 NVIDIA GPU
- Python 3.10 或更高版本
- 建议至少 150 GB 可用磁盘空间

输入字体：

```text
fonts\target.ttf
refs\ref.otf
```

建议使用静态字体。`target.ttf` 应包含 TrueType `glyf` 表；`ref.otf` 可以是静态 TrueType 或静态 CFF OTF。不要使用可变字体、TTC 或 OTC。

## 快速开始

1. 下载或克隆本仓库。
2. 将目标字体放到 `fonts\target.ttf`。
3. 将参考字体放到 `refs\ref.otf`。
4. 双击安装环境：

   ```text
   install_cuda130.bat
   ```

5. 检查项目：

   ```text
   verify_project.bat
   ```

6. 开始或继续完整流程：

   ```text
   run_months_resilient.bat
   ```

7. 安全暂停：

   ```text
   request_safe_stop.bat
   ```

8. 继续运行前清除暂停标记：

   ```text
   clear_safe_stop.bat
   ```

## 输出文件

主要输出：

```text
build\target-HanziStyleForge-Fusion.ttf
build\target-HanziStyleForge-Fusion.ttf.report.json
work_hanzistyleforge_fusion_months\qa\index.html
```

中间训练数据、检查点和生成进度保存在：

```text
work_hanzistyleforge_fusion_months\
```

不要在训练过程中删除该目录。

## 使用前须知

- 完整流程可能持续数天、数周或更久。
- 项目不包含字体文件、预训练权重或第三方字体数据集。
- 生成字体可能同时受 `target.ttf` 和 `ref.otf` 的许可证约束。
- 请仅使用你有权训练、修改和发布的字体。
- 本项目是实验性工具，正式发布字体前请检查 QA 页面并进行人工测试。

## 研究与参考来源

HanziStyleForge Fusion 是独立实现。以下项目和论文为架构设计提供了参考；本仓库不直接打包它们的源码、预训练权重或字体数据集。

| 来源 | 参考方向 |
|---|---|
| [zi2zi](https://github.com/kaonashi-tyc/zi2zi) | 汉字风格迁移、内容与风格分离 |
| [FontDiffuser](https://github.com/yeungchenwa/FontDiffuser) | 扩散生成、多尺度内容聚合、显式风格约束 |
| [HanziGen](https://github.com/wangwenho/HanziGen) | VQ 表示与条件潜空间扩散 |
| [VQ-Font](https://github.com/Yaomingshuai/VQ-Font) | 离散字体 token 与结构感知增强 |
| [LF-Font / MX-Font](https://github.com/clovaai/fewshot-font-generation) | 局部部件风格、因子分解、多专家 |
| [DeepVecFont-v2](https://github.com/yizhiwang96/deepvecfont-v2) | Transformer 矢量序列与轮廓修正 |
| [Efficient and Scalable Chinese Vector Font Generation via Component Composition](https://arxiv.org/abs/2404.06779) | 部件区域变换与大规模组合 |
| [cjkvi/cjkvi-ids](https://github.com/cjkvi/cjkvi-ids) | Unicode IDS 部件结构与局部区域提示 |

引用只表示方法层面的参考，不代表获得复制上游代码、权重、数据或字体的许可。使用任何第三方材料前，请检查其当前许可证与使用条款。

[zi2zi-JiT](https://github.com/kaonashi-tyc/zi2zi-JiT) 单独列在下面，因为它不只是架构参考——它可以作为可选的生成后端使用。

## 可选生成后端：zi2zi-JiT

生成阶段是可插拔的。默认后端是本项目自研的 Style Encoder → VQ → Diffusion → Refiner。作为替代，生成可以交给 [zi2zi-JiT](https://github.com/kaonashi-tyc/zi2zi-JiT)（一个像素空间扩散 Transformer，提供预训练权重），而 HanziStyleForge Fusion 保留其下游的全部环节：候选筛选、IDS 部件校验、QA、精修、轮廓转换和 TTF 构建。

本仓库不打包 zi2zi-JiT 的源码，也不打包它的权重。你需要自行克隆上游仓库并下载权重，后端调用的是你本地的副本。

### 怎么用

后端由 `config.json` 的 `backend` 块选择，也可以用 `--backend` 临时覆盖：

```text
hanzistyleforge.py --backend=zi2zi-jit fusion-generate
```

可选值 `native`（默认，自研生成栈）、`zi2zi-jit`、`dir`（直接读一个已生成好的图片目录，用于手工跑生成后衔接，或在不依赖任何生成器的情况下验证后处理链路）。

```json
"backend": {
  "name": "zi2zi-jit",
  "candidate_count": 3,
  "zi2zi_jit": {
    "repo_dir": "D:/zi2zi-JiT",
    "checkpoint": "D:/zi2zi-JiT/run/lora_target/checkpoint-last.pth",
    "font_label": 0
  }
}
```

`python_executable` 留空表示复用运行 HanziStyleForge 的解释器——zi2zi-JiT 的推理路径只需要 torch、numpy、opencv 和 einops，不需要它 `environment.yaml` 里钉住的那套。

### 必须先做 LoRA 微调

**zi2zi-JiT 公开的 JiT-B/16 权重是预训练产物，不能零样本使用。** 直接拿它生成没见过的字体会系统性丢笔画（上少一竖、不变成 T、个变成人）。上游 README 里每个生成示例用的都是微调后的权重。

用 `scripts/generate_font_dataset.py` 建数据集，源字体要用你的 `ref.otf`——推理时喂的就是它，对齐推理分布比对齐预训练更重要。然后跑 `lora_single_gpu_finetune_jit.py`。微调完把权重路径填进 `checkpoint`，并把 `font_label` 设成 `0`（单字体数据集的目录是 `001_<name>`，对应索引 0）；`font_label` 留空则使用 label-drop token，那只对基座权重有意义。

Windows 上还需要：`TORCHDYNAMO_DISABLE=1`（Triton 无 Windows 构建）、`PYTHONPATH` 指向仓库根（`scripts/` 下脚本的 `sys.path[0]` 是脚本目录）、`--num_workers 0`（DataLoader worker 在 Windows 上要 pickle 含 lambda 的 dataset）、去掉 `--online_eval`（它算 FID，PyPI 版 torch-fidelity 与上游用的 fork API 不一致）。

### 拓扑门槛为什么对后端另设一套

全局 `topology` 阈值是为自研生成器标定的——它被 structure-lock 拉向参考字，因此能高度贴合参考骨架。做真实风格迁移的后端按设计就会偏离，用同一套阈值会拒绝掉全部产出（实测 `topology_score` 中位数 0.14，阈值 0.06）。

所以 `backend.topology` 对非原生后端放宽了骨架相似度类指标。**没有放宽的是连通分量、孔洞和欧拉数之差**——这三项保持为 0，它们才是"生成的确实是同一个字"的保证。实测中它们的中位数本来就是 0，所以正常的风格迁移能通过，而多出或丢失一笔的字会被拦下，改用参考字兜底。

`selection.csv` 里的置信度也按同一道门槛标定——它衡量的是"离门槛还有多少余量"，所以放宽后的后端路径分数天然低于原生路径。QA 判定低置信度的阈值 `qa.low_confidence_threshold` 因此可配（默认 0.75 对应原生标定）。实测 600 字的后端分布是 p10=0.125、p50=0.258、p90=0.486，用默认值会把每个字都标记成低置信度；`config_zi2zi_production.json` 里设的是 0.12，只标记最差的约 10%。

同样的原因，**非原生后端会跳过 `refine` 阶段**。长跑精修的搜索目标是"最接近参考结构的候选"，并通过与参考兜底混合来达成——这对自研生成器的含噪输出是提纯，对风格迁移则是抹除：实测 40 个字里有 32 个被换回了参考字形。由于 `build` 优先读取 `refined/selection.csv`，不跳过的话最终字体会几乎全部由参考轮廓构成。需要强制运行可设 `backend.run_refine=true`。

> **署名义务。** zi2zi-JiT 的代码是 MIT 许可，但它的「Font Artifact License Addendum」对产物追加了条款：当你分发的字体产品中**超过 200 个字符**由它的输出构成时，必须注明出处。本工具一次正常运行重建的字数远超 200，所以只要你用了这个后端，就按需要署名处理：写明 "Created using zi2zi-JiT artifacts" 并附上上游仓库链接。使用默认后端生成的字体不受此约束。详见 `THIRD_PARTY_NOTICES.md`。

## 贡献

欢迎提交 Issue 和 Pull Request。请在提交第三方代码、数据或模型时同时说明来源与许可证。
