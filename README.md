[简体中文](README.md) | [English](README.en.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

# HanziStyleForge

用一个字体的风格，按另一个字体的字形标准，重新生成全部汉字。

工具从目标字体学笔画风格，按参考字体的结构把参考字体覆盖的每一个汉字重画一遍，打包成可安装的 `.ttf`。即使目标字体里已有这个字也会重画，这样整套字才统一。

两种典型用法：

- 扩充字数。目标字体只做了几千字，扩到两万多字
- 改字形标准。目标字体是台标（或日标、韩标）字形，换成大陆版参考字体即可按新标准重画，风格保留

> 实验性项目。完整跑一次要几天到几周，中途可随时中断续跑。

---

## 准备

### 两个字体（必需）

| 文件 | 作用 |
|---|---|
| `fonts/target.ttf` | 风格来源。程序只从它学笔画长什么样 |
| `refs/ref.otf` | 字形来源。程序按它的结构画，也由它决定生成哪些字 |

按文件名主干匹配，后缀不限：`refs/ref.ttf` 同样有效。

字形标准由 `ref` 决定。想要大陆规范就放大陆版，想要台港日韩标准就换对应的，程序不判断哪种更对。

要求静态字体，不支持可变字体、TTC、OTC。`target` 需含 `glyf` 表（TrueType），`ref` 可以是 TrueType 或 CFF/OTF。

### 同形字表（可选，但影响质量）

`data/same_form_han.txt`，仓库不含，需自己准备。列出两个字体**结构相同**（部件与布局一致、只有风格差异）的字。

有这个文件时，`prepare` 会为每个字建一条「参考结构 → 目标字形」的训练样本，这是模型唯一一次见到真实参考字体作为输入的机会。没有则只做目标字体自重建，能跑，但模型在训练中从未见过生成时实际会遇到的输入。

结构不同的字**不能**放进去——那等于教模型把一种地区字形转成另一种，与参考字体的用途相反。

```bash
python tools/same_form_review.py
```

打开一个窗口，逐字左右对照显示两个字体。字形相同按 `y`，不同按 `n`；左右方向键跳过不判定，`u` 撤销当前字的判定。**每按一次键立即写盘**，关窗、断电最多丢一个按键，重开从第一个未判定的字继续。

`--only-undecided` 跳过已判定的字，`--sort suspicious` 把拓扑差异最大的排前面，先清掉明显不同的。

已有字表但没有进度文件时，会把表里的字当作"已判定为相同"载入并留一份 `.bak`，是在原有基础上继续，不会覆盖。

格式见 [data/README.md](data/README.md)。

---

## 硬件

必须有 NVIDIA 显卡，没有 CUDA 跑不动训练。

| 平台 | 训练 |
|---|---|
| Windows / Linux + NVIDIA | 可以 |
| macOS | 不行。无 NVIDIA 显卡，Apple MPS 未支持 |
| 无独显 | 不行 |

Python 3.10-3.14，建议 150 GB 空闲磁盘。显存 12 GB 够用，默认配置按此调校。

Mac 上仍可安装、自检、检查字体，或把已生成的字形图片打包成字体，训练需另找 N 卡机器。

---

## 开始

### Windows

| 步骤 | 双击 | 说明 |
|---|---|---|
| 1 | `install.bat` | 装环境，只需一次 |
| 2 | — | 字体放进 `fonts\` 和 `refs\` |
| 3 | `verify.bat` | 检查 |
| 4 | `run.bat` | 开跑 |

`stop.bat` 在下一个存档点安全退出，再双击 `run.bat` 接着跑。

### Linux / macOS

```bash
./install.sh          # 装环境
./verify.sh           # 放好字体后检查
./run.sh              # 开跑；./stop.sh 暂停
```

提示 `Permission denied` 时先 `chmod +x *.sh`。

---

## 产出

```text
build/target-HanziStyleForge.ttf              成品字体
build/target-HanziStyleForge.ttf.report.json  构建报告
work/qa/index.html                            质检报告
```

装字体前先看质检报告，里面有逐字对照图，能看出哪些字生成得好、哪些回退成了参考字形。

训练数据、模型存档和生成进度都在 `work/`，几十 GB，跑的过程中别删。

---

## 显存配置

默认 `config.json` 按 12 GB 显存调校，12 GB 的卡直接用。以下是 RTX 5070 Ti Laptop（11.9 GB）实测的单步峰值：

| 阶段 | 默认 batch | 峰值显存 |
|---|---|---|
| vq256 | 4 | 3.1 GB |
| vq384 | 2 | 3.4 GB |
| vq512 | 1 | 3.0 GB |
| direct256 | 6 | 4.6 GB |
| direct384 | 3 | 5.2 GB |
| latent256 | 6 | 5.4 GB |
| latent384 | 4 | 7.8 GB |
| latent512 | 4 | 11.2 GB |

### 推荐值

| 显存 | 怎么改 |
|---|---|
| 16 GB+ | 不用改 |
| 12 GB | 不用改 |
| 10 GB | `latent512` 的 `batch_size` 4→2，`gradient_accumulation` 1→2 |
| 8 GB | 同上，另加 `latent384` 4→2 / 1→2 |
| 6 GB | 所有阶段 `batch_size` 减半、`gradient_accumulation` 翻倍 |

规则：`batch_size × gradient_accumulation` 的乘积要保持不变，否则等效批量变了，学习率不再匹配。

显存更大不会明显更快。瓶颈是计算不是显存，实测 batch 从 4 加到 8 的吞吐提升在 10% 以内。

---

## CPU 配置

`training.workers` 是数据加载进程数，默认 4。

| 物理核心 | workers |
|---|---|
| 4 核 | 2 |
| 6-8 核 | 4（默认） |
| 12 核以上 | 6-8 |

每个 worker 约占 1.5-2 GB 内存，内存紧张就调小。

`cpu_threads`、`interop_threads`、`opencv_threads` 不要调大。训练是 GPU 密集型，这三个调大只会让 CPU 抢占 GPU 的调度时间。

判断要不要加 worker：跑起来后看终端的 batch/s。如果 GPU 利用率（`nvidia-smi`）长期低于 80%，加 worker 有用；已经 90% 以上就是计算受限，加了没用。

---

## 改配置

配置在项目根目录的 `config.json`，四个启动脚本传的都是它。保存时必须存 UTF-8。

各阶段都有 `name` 字段，照名字找。例如 8 GB 显卡要改 `fusion` → `diffusion` → `phases` 里 `"name": "latent512"` 那一段：

```json
{
  "name": "latent512",
  "batch_size": 2,
  "gradient_accumulation": 2
}
```

乘积仍是 4，没变。

### 哪些改动会作废已有进度

程序用指纹判断存档能否续用，对不上就从该阶段第 1 轮重来。

| 改动 | 后果 |
|---|---|
| `fusion.vq.phases[]` 任何一项 | 该 VQ 阶段重训 |
| `fusion.diffusion.phases[]` 任何一项 | 该扩散阶段重训 |
| `fusion.style_encoder` 的 `size`、`epochs`、`batch_size`、`learning_rate`、`virtual_length`、`references_per_set`、`cell_grid`、`query_gain` | 风格阶段重训 |
| `fusion.refiner` 的 `size`、`epochs`、`batch_size`、`gradient_accumulation`、`learning_rate` | 精修阶段重训 |
| `fusion.style_encoder.early_stopping` | 安全 |
| `fusion.refiner.minimum_epochs`、`minimum_relative_improvement` | 安全 |
| `training.workers`、`training.amp`、`checkpoint_every_steps`、`preview_every` | 安全 |
| `build` 任何一项 | 安全，只在最后生成 TTF 时读取 |

VQ 和扩散的 `early_stopping` 写在 phase 里面，改它也会导致该阶段重训，这点和风格、精修阶段不同。

显存相关参数请在开跑前定好，跑到一半改那个阶段要重来。

### 爆显存

看报错发生在哪个阶段（终端会打印阶段名），只改那一段：`batch_size` 减半、`gradient_accumulation` 翻倍，还爆就再来一次。

### 字体太大

体积几乎全在字形轮廓上。控制它的是 `build.curve_simplify`，值越大保留的点越少。在参考字体上实测 17 个不同复杂度的字：

| `curve_simplify` | 点数/字 | Dice |
|---|---|---|
| 0.55 | 611 | 0.9848 |
| 1.2（默认） | 167 | 0.9843 |
| 1.8 | 130 | 0.9812 |
| 2.5 | 117 | 0.9786 |

手工设计的字体约 156 点/字，默认值就在这个水平，两万多字约 14 MB。超过 1.8 吻合度明显下降；调到 0.55 也没好处，多出的点是在描栅格锯齿而非字形。

其余旋钮效果很弱，`maximum_points_per_contour` 从 480 压到 96 也只减到 69%，不用动。

看成品大在哪：

```bash
python tools/font_size_report.py build/target-HanziStyleForge.ttf
```

### 检查改动

```bash
verify.bat        # Linux/macOS: ./verify.sh
```

检查 JSON 语法和取值范围。JSON 最常见的错误是多了或少了逗号，最后一项后面不能有逗号。

---

## 中断

每个阶段和每个生成的字都有存档，再跑一次同样的命令就从断点继续。断电、蓝屏、Ctrl+C 都一样。启动脚本还会在出错后自动重试，连续失败 20 次才停。

不用启动脚本、自己敲命令恢复时注意：主动请求停止会在根目录留下 `STOP_AFTER_CHECKPOINT` 标记，启动脚本每次启动会自动删，手敲不会，那样新运行会在第一个检查点就停下。手动恢复前先删掉它。

---

## 其他工具

```bash
python png_to_svg.py <图片目录> [输出目录]
```

把一批字形图片矢量化成 SVG，用的是和 `build` 相同的轮廓提取。简化容差按输入图尺寸自动缩放。

---

## 常见问题

**要跑多久？** 几天到几周，取决于字数和显卡。默认配置在 12 GB 笔记本显卡上按周计。

**能只跑一部分字吗？** 可以。`scope.mode` 改成 `chars_file`，`scope.extra_chars_file` 指向字表（每行一个字或 `U+4E00` 形式的码位）。先拿几百字试跑是个好习惯。

**生成的字会不会不符合规范？** 有一道结构闸门：连通分量数、孔洞数和参考字对不上就拒绝，改用参考字形。这保证不出现多一笔少一笔的错字，代价是一部分字保持参考字体的样子。

**非汉字内容会被改吗？** 不会。拉丁字母、数字、标点、假名、谚文，以及 OpenType 排版和 hinting 数据都从 `target.ttf` 原样保留，构建时逐字节校验。

**报错 `requires CUDA, but torch.cuda.is_available() is False`** 没检测到可用的 NVIDIA 显卡。更新驱动，或确认装的是 CUDA 版 PyTorch。

**报错 `does not support training.device='mps'`** Apple GPU 不受支持。

---

## 须知

- 完整流程可能持续数天到数周
- 本仓库不含任何字体文件、预训练权重或第三方数据集
- 生成的字体可能同时受 `target.ttf` 和 `ref` 的许可证约束。请只使用你有权训练、修改和发布的字体
- 实验性工具。发布前请检查质检报告并人工测试

本软件采用 Apache License 2.0，见 [LICENSE](LICENSE) 和 [NOTICE](NOTICE)。许可证只覆盖软件本身，不授予你对任何字体的训练、修改或发布权利。第三方引用见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

---

## 工作原理

```text
target.ttf（风格）  ref.otf（字形结构）
        └──────┬──────┘
               ↓
   风格编码 → VQ 笔画码本 → 潜空间扩散 → 精修
               ↓
   多候选生成 → 结构闸门筛选 → 质检 → 轮廓矢量化 → TTF
```

训练样本有两种：目标字体自重建，以及同形字表覆盖的「参考结构 → 目标字形」配对。两者的真值都只来自 `target.ttf`，结构输入只来自 `target` 自身或 `ref`，数据流在运行时校验。

## 贡献

欢迎 Issue 和 Pull Request。提交第三方代码、数据或模型时请注明来源与许可证。
