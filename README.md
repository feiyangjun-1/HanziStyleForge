[简体中文](README.md) | [English](README.en.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

# HanziStyleForge Fusion

**用一个字体的风格，按另一个字体的字形标准，重新生成全部汉字。**

工具会学习目标字体的笔画风格，然后照着参考字体的字形结构，把参考字体覆盖的**每一个**汉字重新画一遍，最后打包成可以直接安装的 `.ttf`。

注意是"重新生成"而不是"补缺"：即使目标字体里已经有这个字，也会按参考字形重画。这样整套字才是统一的。

两种典型用法：

- **扩充字数。** 目标字体只做了几千字，想扩到两万多字
- **改字形标准。** 目标字体是台标（或日标、韩标）字形，想改成大陆规范——把 `ref.otf` 换成大陆版参考字体即可，风格保留，字形按新标准重画

> 实验性项目。完整跑一次要几天到几周，中途可以随时中断，下次接着跑。

---

## 你需要准备两个字体

| 文件 | 作用 | 说明 |
|---|---|---|
| `fonts/target.ttf` | **风格来源** | 你喜欢的那个字体。程序只从它学"笔画长什么样" |
| `refs/ref.otf` | **字形来源** | 一个字数齐全的字体。程序按它的字形结构来画 |

举例：`target.ttf` 放思源黑体改的手写风字体，`refs/ref.otf` 放思源黑体本体，产出就是"手写风格 + 思源黑体字形标准"的完整字体。

**字形标准由 `ref.otf` 决定。** 想要大陆规范就放大陆版参考字体，想要台湾、香港、日本、韩国标准就换对应的。程序不会自己判断哪种"更对"。

### 对字体文件的要求

- 静态字体。**可变字体、TTC、OTC 不支持**
- `target.ttf` 要是 TrueType（含 `glyf` 表）
- `ref.otf` 可以是 TrueType 或 CFF/OTF

---

## 硬件要求

**必须有 NVIDIA 显卡。** 训练是这个项目的主体，没有 CUDA 就跑不动。

| 平台 | 能否训练 |
|---|---|
| Windows + NVIDIA | 可以 |
| Linux + NVIDIA | 可以 |
| macOS | **不行**。Mac 没有 NVIDIA 显卡，Apple 的 MPS 也未支持，只能用 CPU，慢到不现实 |
| 无独显的 Linux / Windows | 同上 |

其他要求：Python 3.10-3.14，建议至少 150 GB 空闲磁盘。显存 12 GB 够用（默认配置就是按 12 GB 调的）。

在 Mac 上你仍然可以安装并运行自检、检查字体、或者把已经生成好的字形图片打包成字体，但训练请找一台有 N 卡的机器。

---

## 不同显存怎么设置

默认配置 `config_months_12gb.json` 是按 **12 GB 显存**调好的，12 GB 的卡直接用，不用改任何东西。

下面是在 RTX 5070 Ti Laptop（11.9 GB）上实测的单步峰值显存，推荐值都是从这里算出来的：

| 阶段 | 分辨率 | 每多一个样本 | 固定开销 |
|---|---|---|---|
| VQ 自编码器 | 256 / 384 / 512 | 0.37 / 0.83 / 1.47 GB | 约 0.13 GB |
| 扩散（含 VQ、风格编码器、EMA） | 256 / 384 / 512 | 0.34 / 0.76 / 1.35 GB | 约 0.44 GB |
| 精修 | 384 | 0.62 GB | 约 0.40 GB |

### 推荐值

只列需要改的地方，没列出来的保持默认。

| 显存 | 要改什么 |
|---|---|
| **8 GB** | `fusion.diffusion.phases[2]`（latent512）：`batch_size` 4 → **2**，`gradient_accumulation` 1 → **2** |
| **12 GB** | **不用改**，默认就是 |
| **16 GB 及以上** | 把累积折掉，换成更大的实际批：<br>`fusion.vq.phases[1]`（vq384）：批 3 → **6**，累积 2 → **1**<br>`fusion.vq.phases[2]`（vq512）：批 1 → **4**，累积 4 → **1**<br>`fusion.direct_baseline.phases[1]`：批 3 → **6**，累积 2 → **1**<br>`fusion.refiner`：批 2 → **4**，累积 2 → **1**<br>`fusion.purification`：批 2 → **4**，累积 2 → **1** |

### 一条必须遵守的规则

**`batch_size × gradient_accumulation` 这个乘积不能变。**

这个乘积叫**有效批大小**，它决定训练结果。显存小就把 `batch_size` 减半、`gradient_accumulation` 翻倍，算出来的梯度是一样的，只是分几次攒。乘积变了，训练行为就变了，不只是快慢的差别——学习率也得跟着重调。

### 显存更大不会快很多

这条流水线是**算力受限**的，不是显存受限。实测把梯度累积折掉后，扩散阶段只快了 1.06 倍，VQ 阶段 1.12 倍。24 GB 的卡和 16 GB 的卡跑同一份配置速度基本一样——多出来的显存没有地方可花，除非你提高有效批大小（会改变训练结果）或者提高分辨率。

**别指望换大显存的卡能把几周缩短成几天。** 真正决定时间的是 GPU 算力和字数。

---

## 怎么改配置文件

### 文件在哪

项目根目录的 **`config_months_12gb.json`**。四个启动脚本（`run_months_resilient.bat` / `run.sh` 等）传的都是这一个文件，改它就够了。

用记事本、VS Code 或任何文本编辑器都能打开。**保存时必须存成 UTF-8。**

### 怎么找到要改的地方

配置是一层套一层的。上面写的 `fusion.vq.phases[2].batch_size` 就是这样往下找：

```json
{
  "fusion": {                    ← 找到 "fusion"
    "vq": {                      ← 里面找 "vq"
      "phases": [                ← 里面找 "phases"，这是个列表
        { "name": "vq256", ... },     ← [0] 第一个
        { "name": "vq384", ... },     ← [1] 第二个
        { "name": "vq512",            ← [2] 第三个（从 0 数起）
          "size": 512,
          "batch_size": 1,       ← 改这里
          "gradient_accumulation": 4
        }
      ]
    }
  }
}
```

每个阶段都有 `name`，照着名字找最保险。

### 完整例子：8 GB 显卡

找到 `fusion` → `diffusion` → `phases`，里面 `"name": "latent512"` 那一段：

```json
{
  "name": "latent512",
  "size": 512,
  "batch_size": 2,             ← 原来是 4
  "gradient_accumulation": 2,  ← 原来是 1
  ...其余不动
}
```

乘积从 `4 × 1` 变成 `2 × 2`，还是 4，没变。

### 改之前一定要知道：哪些改动会让已有进度作废

程序用「指纹」判断存档能不能接着用。指纹对不上就**从这个阶段的第 1 轮重新开始**，之前跑的全部作废。

| 你改的 | 后果 |
|---|---|
| `fusion.vq.phases[]` 里的**任何一项** | 该 VQ 阶段从头重训 |
| `fusion.diffusion.phases[]` 里的**任何一项** | 该扩散阶段从头重训 |
| `fusion.style_encoder` 的 `size`、`epochs`、`batch_size`、`learning_rate`、`virtual_length`、`references_per_set`、`cell_grid`、`query_gain` | 风格阶段从头重训 |
| `fusion.refiner` 的 `size`、`epochs`、`batch_size`、`gradient_accumulation`、`learning_rate` | 精修阶段从头重训 |
| `fusion.style_encoder.early_stopping` 里的任何一项 | **安全**，下一轮就生效 |
| `fusion.refiner.minimum_epochs`、`minimum_relative_improvement` | **安全** |
| `training.workers`、`training.amp`、各阶段的 `checkpoint_every_steps`、`preview_every` | **安全** |

**有一个容易踩的坑**：VQ 和扩散的 `early_stopping` 是写在 phase 里面的，所以改它**也会**导致该阶段重训——这一点和风格阶段、精修阶段不一样。

所以：**显存相关的参数请在开跑之前定好。** 跑到一半发现爆显存再改，那个阶段就得重来。

### 爆显存了怎么办

看到 `CUDA out of memory` 时，先看报错发生在哪个阶段（终端会打印阶段名，例如 `vq512`、`latent384`），然后：

1. 找到那个阶段，`batch_size` 减半
2. 同一段里 `gradient_accumulation` 翻倍
3. 还爆就再重复一次

只改出问题的那个阶段，别一次全改。

### 改完怎么确认没写坏

```bash
verify_project.bat
```

Linux / macOS 用 `./verify.sh`。它会检查 JSON 语法和取值范围。JSON 最常见的错误是**多了或少了逗号**——最后一项后面不能有逗号。

---

## 开始使用

### Windows

双击这四个文件，按顺序：

| 步骤 | 双击 | 做什么 |
|---|---|---|
| 1 | `install_cuda130.bat` | 装环境。只需一次 |
| 2 | — | 把两个字体放进 `fonts\` 和 `refs\` |
| 3 | `verify_project.bat` | 检查一切正常 |
| 4 | `run_months_resilient.bat` | 开跑 |

想中途暂停就双击 `request_safe_stop.bat`，程序会在下一个存档点安全退出。再次双击 `run_months_resilient.bat` 就接着跑。

### Linux 和 macOS

打开终端，进入项目目录：

```bash
./install.sh
```

把两个字体放进 `fonts/` 和 `refs/`，然后：

```bash
./verify.sh
./run.sh
```

想中途暂停：

```bash
./stop.sh
```

再运行一次 `./run.sh` 就从存档点接着跑。

> 如果提示 `Permission denied`，先执行 `chmod +x *.sh`。

---

## 产出在哪

```text
build/target-HanziStyleForge-Fusion.ttf        ← 成品字体
build/target-HanziStyleForge-Fusion.ttf.report.json   ← 构建报告
work_hanzistyleforge_fusion_months/qa/index.html      ← 质检报告，用浏览器打开
```

**装字体之前先看质检报告。** 里面有逐字对照图，能看出哪些字生成得好、哪些回退成了参考字形。

训练数据、模型存档和生成进度都在 `work_hanzistyleforge_fusion_months/`，几十 GB，**跑的过程中别删**。

---

## 中断了会怎样

不会怎样。每个阶段和每个生成的字都有存档，再跑一次同样的命令就从断点继续。

断电、蓝屏、Ctrl+C 都一样。Windows 的 `run_months_resilient.bat` 和 Linux/macOS 的 `run.sh` 还会在出错后自动重试，连续失败 20 次才会停下来——那说明是真的有问题，不是一时的波动。

> **如果你不用启动脚本、而是自己敲命令恢复**：主动请求停止会在项目根目录留下一个 `STOP_AFTER_CHECKPOINT` 标记文件。启动脚本每次启动都会自动删掉它，手动敲命令则不会，那样新的运行会在第一个检查点就停下。手动恢复前先删掉这个文件（Windows `del STOP_AFTER_CHECKPOINT`，Linux/macOS `rm -f STOP_AFTER_CHECKPOINT`）。

---

## 常见问题

**要跑多久？**
几天到几周，取决于字数和显卡。默认配置在 12 GB 的笔记本显卡上是按周计的。

**能不能只跑一部分字？**
可以。在配置文件里把 `scope.mode` 改成 `chars_file`，用 `scope.extra_chars_file` 指向一个字表文件（每行一个字或一个 `U+4E00` 形式的码位）。先拿几百字试跑一遍是个好习惯。

**生成的字会不会不符合规范？**
程序有一道结构闸门：生成的字如果连通分量数、孔洞数和参考字对不上，就会被拒绝并改用参考字形。这保证不会出现"多一笔少一笔"的错字，代价是有一部分字保持参考字体的样子而不是目标风格。

**非汉字内容会被改吗？**
不会。拉丁字母、数字、标点、假名、谚文,以及 OpenType 的排版和 hinting 数据都从 `target.ttf` 原样保留，构建时会逐字节校验。

**报错 `requires CUDA, but torch.cuda.is_available() is False`？**
没检测到可用的 NVIDIA 显卡。更新显卡驱动，或确认装的是 CUDA 版 PyTorch。

**报错 `does not support training.device='mps'`？**
Apple GPU 不受支持。见上面的硬件要求。

---

## 使用前须知

- 完整流程可能持续数天、数周或更久
- 本仓库不含任何字体文件、预训练权重或第三方数据集
- **生成的字体可能同时受 `target.ttf` 和 `ref.otf` 的许可证约束**。请只使用你有权训练、修改和发布的字体
- 这是实验性工具。正式发布前请检查质检报告并人工测试

第三方引用和许可信息见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

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

程序只从 `target.ttf` 学风格，只从 `ref.otf` 取结构，两者的数据流是分开的并在运行时校验。

## 贡献

欢迎 Issue 和 Pull Request。提交第三方代码、数据或模型时请注明来源与许可证。
