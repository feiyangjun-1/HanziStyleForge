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
