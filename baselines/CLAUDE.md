# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本目录下工作时提供指导规范。

`baselines/` 是 TinyBrains Ants 参赛模型训练的核心实现：包含 `tb_baselines` Python 包（特征编码、脚本启发式教师模型、学习算法、模型导出），由 `ants-starter` 通过 `#subdirectory=baselines` 使用 pip 安装，因此**该目录名与包名是面向所有 starter 克隆项目的刚性契约**。这里训练的参赛模型均为普通选手模型，无任何特殊特权。它拥有独立的工具链且不打包进任何发布中：`../build.sh` 从不运行它，`../tools/pack.py` 仅打包 `../dist`，因此此处的修改绝不会变动引擎摘要。`README.md` 是面向人类的指南（包含命令、设计规则、构件与不变量）；`../CLAUDE.md` 是卡带核心指南，`../../CLAUDE.md` 是平台指南。

`models/<class>-<method>/` 是导出脚本写入每个构件的位置（`model.onnx`、生成的 `manifest.json`、`metrics.json`、`card.md`）。**该目录已被 git 忽略**：平台绝不在本仓库提交模型权重。值得保留的模型会提交至 `ants-starter/models/` 供参赛选手作为对手测试，或在赛季管理后台作为基准线上传。

## 检查与门禁（Checks）

依赖项：需要先构建游戏卡带（`games.toml` 指向 `../dist`）并安装最新的 `tinybrains` CLI：

```sh
(cd .. && ./build.sh)                              # 生成 ../dist：Wasm 组件、cartridge.json 与参考观测数据
pip install -e '.[dev]'                            # 安装依赖：torch, numpy, onnx, pytest
pytest tests/ -q                                   # 核心门禁测试；需要 PATH 中存在 `tinybrains` 或设置了 TINYBRAINS 环境变量
python -m tb_baselines.adapters > /tmp/a.json      # 生成的基础适配器清单
python -m tb_baselines.adapters --memory > /tmp/m.json   # 带全局记忆端口的适配器清单
python -m tb_baselines.adapters --memory --ants > /tmp/a.json   # 带单蚁记忆三输入端口的适配器清单
```

一致性测试会加载已训练的计算图：优先读取 `$TB_CONFORMANCE_ONNX`，否则读取 `../../ants-starter/models/micro-bc/model.onnx`。CI 会克隆 starter 并显式指定该环境变量，因此文件缺失会直接判定为失败而非跳过。记忆适配器测试使用 `onnx` 构建专属的 4 端口计算图，需要支持记忆的 `tinybrains` CLI（0.3.0 之后的版本）；已发布的 0.3.0 CLI 会因此报错并提示原因。

## 规则与约束（Rules）

**三大运行关卡，且其中只有两个是真实的平台裁决：**

```
tinybrains env       训练环境推演    速度极快；无超时惩罚，无犯规判定，无适配器介入
tinybrains <match>   eval.py         真实的对弈运行路径，仅缺少准入检查
tinybrains check     export.py       平台准入时真正执行的裁决
```

在 env 环境中训练表现良好的策略，依然可能因超时被判犯规（strike），或因适配器超出预算而被平台准入拒绝。切勿将来自 env 的对战结果直接当作正式评估结果汇报。

- **`planes.py` 是特征编码的唯一定义，且被实现渲染了两次**。NumPy 编码器用于神经网络训练；`adapters.py` 生成天梯运行所需的 `manifest.json`。`tests/test_adapter_conformance.py` 在卡带的参考观测数据上运行真正的评估器（`tinybrains adapt`，即 Orion 节点在准入时评估 manifest 所使用的 **datalogic**），并断言两者逐元素严格完全一致。**这是本目录下最重要的核心测试**。
- **`planes.MEMORY` 实现了三种形式，由同一个测试约束其一致性**：直通适配器（`memory_adapter`）、计算图内部的 `Max` 算子（`nets.WithMemory`）以及训练器的状态更新并集（`remember`, `encode_memory`）。记忆与棋盘一样采用 `i8`，每个单元格 2 字节。数据集的每一行在 `m` 键下携带对应座位的记忆状态（`collect.py`），因此修改 `MEMORY` 意味着需要重新生成数据集，且 `train/bc.py --memory` 会拒绝数据头中声明了其他特征平面的旧数据集。
- **单蚁记忆是一个多表连接操作（join），实现了两次，由同一个测试约束其一致性**：`planes.py` 的 `_ant_rows_logic`（适配器端：将上回合行按 ID 存入表，按本回合 `ids` 顺序读回，并散布到各蚂蚁所在单元格）与 `ant_planes_in_view`（NumPy 端）通过 `tinybrains adapt --obs` 进行比对；`train/seq.py` 的 `AntCarry` 是梯度流经的 PyTorch 副本。蚂蚁 ID 模运算对齐到 `ANT_TABLE`，计算图会在每只蚂蚁的数据行前写入其 ID。
- **计算图内部计算的记忆必须按回合时序训练**：`train/seq.py` 按照顺序按 `k` 步回放每个座位的对局数据（`collect.py`），并采用截断沿时间反向传播（TBPTT）；跨回合传递的记忆是运行器所传递的舍入整数，通过 `nets.LearnedMemory` 与 `nets.PerAnt` 中的直通估计器（Straight-Through Estimator, STE）传递。`train/bc.py` 则用于对历史行执行固定函数记忆（`planes.MEMORY`），可由 NumPy 独立传递。
- **计算图按整体大小进行衡量，因此 `export.py` 必须对其进行精简（shorten）**：`shorten` 会剥离节点名称，将所有中间张量重命名为短 ID，并合并共享重复的 `Constant` 常量节点；输入输出端口和初始化器保持原名。各规格等级的填充大小在此精简后进行测量，对精简规则的任何修改都会改变每个构件的字节数。
- **教师模型仅在明确要求时才读取记忆**：`Teacher.orders(..., memory)` 会读取座位的记忆状态（见过的敌方蚁穴在确认消失前保持为攻击目标；视野外的可见食物作为弱目标），`collect.py --remember` 会向其提供记忆。各等级天梯的数据集采集时默认不带记忆，因此其教师模型步步动作完全相同（`tests/test_teacher.py`）。
- **`manifest.json` 必须自动生成，严禁手工编辑**：只修改特征编码的一处渲染而不同步另一处，正是该一致性测试专门要拦截的 Bug。
- **观测字段变动必须在同一次提交中同步更新此处**：`planes.py` 负责编码 `../engine/src/observe.rs` 输出的内容；当特征平面集合发生变动时，必须重新生成清单并更新 `classes.toml` 中的 `manifest_bytes`。
- **`classes.toml` 中的数字全部来自真实测量，且每个数字旁都记录了测量数据**：这里没有任何凭空猜测的数值。重新测量时，必须同时替换数字及其备注。
- **变体模型通过命令行指定，而非建立第二张表格**：`--arch/--channels/--blocks` 用于覆盖等级默认配置，运行生成的 `history.json` 和模型卡片会记录实际构建的结构。`classes.toml` 保持 5 个固定等级不变。
- **教师模型是标注源，而不是基线模型**：它绝不对外交付。天梯等级唯有在每个等级均从*同一个*教师模型蒸馏时才有意义，因此修改教师模型意味着需要重构整个数据集并重新训练全部模型。
- **价值头（Value Head / Critic）绝不导出**：`export.py` 仅提取主干（Trunk）和策略头（Policy Head）。这是实质性的预算节约（在 nano 等级中，价值网络会占去三分之一的预算），同时也确保了特权信息的安全性：价值网络在训练时即使能够看到真实分数也是安全的，因为它在任何对局运行前就被剥离丢弃了。
- **引擎摘要属于每个构件的必备元数据**：数据集头部、`metrics.json` 以及 `card.md` 的 *Measured on*（测量运行基准）。针对一个引擎训练的模型无法在另一个引擎上无损复现。模型卡片中标注的是 *Measured on* 而非 *Engine*，因为它是测量记录，而非要求读者必须与 `games.toml` 匹配的锁定项；`ants-starter` 中的卡片特意保留了较旧引擎的记录，不会随着每个 ants 版本迭代盲目追逐刷新。
- **`data/`、`runs/`、`replays/` 和 `models/` 属于 git 忽略的输出文件**：数据集约为 90 MB，可通过随机种子重新生成，且检查点（checkpoint）不属于最终交付构件。
- **模型卡片由 `export.py` 中的 `CARD` 模板生成**，且 `ants-starter` 会提交其模型的卡片。修改模板中的固定文案需要手动同步修改这些已提交的卡片。

## 关键陷阱与易错点（Gotchas / What breaks）

- **旧版 `tinybrains` 会导致环境测试失败**：`tests/test_env.py` 需要支持 `--maps` 的二进制文件；旧版本会异常退出，导致每个环境测试都报 "the environment exited"。请将 `TINYBRAINS` 指向当前构建产物（`../../cli/target/release/tinybrains`）或最新发布版本。
- **旧版 `tinybrains` 同样会导致记忆适配器测试失败**：在携带 `memory` 或 `ant_memory` 的视野上执行 `adapt --obs` 需要携带 carry 支持；0.3.0 会拒绝嵌套数组并报错说明原因。Ants 的 CI 从 cli 的 `main` 分支构建 CLI，因此这些测试只有在 main 包含 carry 支持后才能通过。
- **`tinybrains check` 通过并不等于白名单检查通过（在 CLI `b7e1625` 之前）**：包含 `GreaterOrEqual` 的计算图能通过 `check`，但在准入时会被报 `OP_NOT_ALLOWED`；本地运行时支持的算子多于准入部署允许的范围。应使用带半整数阈值的 `Greater` 表达比较逻辑，并在上传前对照文档的 *Format* 页面核对 `metrics.json` 中的 `operators`。
- **单蚁记忆装不下 nano 等级**：其计算图（gather、id 列、两个门控）加上 2,674 字节的清单在未计入权重时就已经超出了预算上限；`micro` 是能容纳它的最小等级，且赛季等级必须允许其额外开销（单单此项就占用每单元格 3 字节，加上棋盘记忆共 5 字节）。
- **在同一台 MPS（Apple Silicon GPU）上同时跑 4 个训练会超载**：时序训练器需要缓存 `bptt` 回合的激活值；在一台 16 GB 内存的机器上同时跑两个时序训练和两个 `bc.py` 进程会导致 Metal 内存耗尽（OOM），进程虽然存活但会产生垃圾数据。时序训练器应一次只跑一个，且后台作业应使用 `setopt nobgnice` 启动：zsh 默认会对 `&` 后台任务降低优先级，导致训练器被调度到能效核心上。
- **CI 安装的是 `numpy`、`onnx` 和 `pytest`，不安装 PyTorch**：图侧记忆测试（`test_the_graphs_memory_update_matches_the_encoder`）在没有 PyTorch 时会跳过，仅在开发环境中运行；CI 运行的是其对应的 NumPy 测试。
- **感受野是刚性约束，而不是单纯的模型容量**：感受野窗口过小的模型能够顺利训练、导出、通过准入，但会在对局中无声无息地惨败。README §设计规则中详细记录了感受野法则。
- **fp16 初始化器是免费的容量加成**：在每次使用时通过 `Cast` 转回 float32 会被运行时执行常量折叠，因此*文件体积*减半而推理结果完全不变：在同一等级下参数量提升至 2.03 倍，对战表现完全一致。大小指标统计的是构件的原始字节数，因此没有任何理由交付 fp32 模型。
- **超过 `mini` 等级后，回合时间限制在达到字节上限前就先生效**：一个座位独占整个回合时限（每个座位执行一次 `model_infer`，各自拥有独立的 `timeout_ms`），而 `export.py` 仍会拒绝超过该份额 70% 的构件：在此处消耗 95% 时钟的模型在更慢的宿主机器上将无容身之地。
- **`ConvTranspose` 不在允许的算子白名单中**：使用 `Resize` 作为上采样算子。
- **棋盘具有环面环绕性，且 Padding 填充开销巨大**：在每次卷积前执行环绕填充的计算量是原模型的 2.3 倍；在每个分辨率阶段只执行一次环绕填充，计算量完全相同，但复制开销降至三分之一。
- **批次按棋盘尺寸划分**：环境在一个 wave 中只运行单张棋盘，基础棋盘有 5 种尺寸且赛季棋盘可为其中任意尺寸，单个张量无法容纳不同尺寸的数据。`Step.groups` 负责按尺寸分组，而 `Step.boards` 在尺寸不一时会直接抛出异常，而不是静默返回残缺批次。
- **`dilate` 从蚂蚁位置向外扩散，而不是滚动（roll）整个平面**：将棋盘按所有 241 个圆盘偏移量进行 roll 操作大约占了整个训练循环的一半耗时；从每个已标记单元格向外遍历圆盘，仅需极少量的写入即可得到完全相同的结果。它不属于特征编码本身（卡带直接发送 `vis`，`planes.py` 使用 `rle_expand` 读取）；保留它是为了独立验证引擎输出的掩码与训练器计算出的圆盘完全一致。
