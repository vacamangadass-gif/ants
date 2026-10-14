# ants/baselines

**TinyBrains Ants 参赛模型训练方法**：包括观测特征编码、脚本启发式教师模型、学习算法，以及使用与评估选手模型相同的代码对模型进行规格等级测量的导出工具。它构成了 `tb_baselines` Python 包，供 ants-starter 的 `train.py` 安装使用。本仓库自身不提交任何已训练的模型权重：供参赛选手作为基准对照的模型统一保存在 [ants-starter](https://github.com/Tiny-Brains/ants-starter) 的 `models/` 目录下。

它是 [ants](../README.md) 的一个子目录，拥有独立的工具链。此处的任何内容都不属于游戏卡带的构建组成部分：`../build.sh` 不会执行它，`../tools/pack.py` 仅打包 `../dist`，因此此处的修改不会改变引擎摘要，也不会进入官方发布包。CI 的 `build` 工作流会在每次构建时针对当前构建产物运行此处的自动化测试。

## 范围与定位（Scope）

包含两个并非完全正交的研究维度：

- **等级天梯（The class ladder）**：将同一个固定的教师模型蒸馏到 5 个规格等级中。完全相同的数据、标签和损失函数；仅仅改变计算预算。这是各规格等级旨在衡量的模型体积与保真度曲线。
- **方法列（The method column）**：固定一个等级和一个数据集，只改变学习算法，从而使得对比发生在算法之间，而非将算法与资源预算混杂在一起。其对照组 `percell` 仅使用 1x1 卷积，参数量与卷积主干网络对齐，因此具有相同容量但完全没有空间感受野：两者之间的表现差距量化了“环顾四周”的价值。

这两个维度无法完全正交，因为计算消耗并不会随着字节大小线性扩展。超过 `mini` 等级之后，回合时间限制在达到字节上限之前就早已成为刚性瓶颈，没有任何密集型架构能够触及 `large` 等级的上限。[`classes.toml`](classes.toml) 记录了所有具体数值及其背后的实测数据。

**记忆列（A memory column）**独立于天梯之外，因为它拥有专属的教师模型：`collect.py --remember` 让教师模型读取每个座位的记忆状态（参见 `teacher.py` 中的*会记忆的教师*：见过一次的敌方蚁穴在确认消失前保持为攻击目标，视野外曾经见过的食物作为较弱的目标），因此其标签取决于座位记住了什么，带记忆的模型也因此有了可以模仿学习的信号。在该数据集上，以相同方式训练了三种模型，仅在所携带的状态上有所不同：

- 仅基于棋盘的 `train/bc.py`：遗忘的代价（对照组）；
- 携带 `planes.MEMORY` 两个平面的 `train/bc.py --memory`：见过的食物与见过的敌巢，在计算图中通过 `Max` 算子维护并在每回合由运行器传回（参见文档中的 *Memory* 章节）：属于固定函数记忆，其内部不含任何可学习参数；
- 采用 `nets.LearnedMemory` 的 `train/seq.py`：同样的两个 `i8` 平面，但是由优化器塑造的门控更新写入，通过截断沿时间反向传播（TBPTT）按时序回放每个座位的对局数据进行训练。`--ants` 会增加单蚁记忆（`planes.ANT_MEMORY`），每只蚂蚁对应一行并按 ID 追踪，能容纳它的最小等级为 `micro`。

等级天梯本身的数据集采集时依然不带记忆，因此其教师模型保持不变。

**该对比列的实测结果**（均在 nano 等级下训练 5 个 epoch，随后通过 `tinybrains <match>` 在赛季的 9 张双人棋盘上进行单循环赛，覆盖每对对决的两个座位，每场 1000 回合；由 `eval.py` 评测，各模型卡片记录了各自构件的具体数值）：

| 携带状态 | 标签来源 | 留出集吻合度 | 循环赛胜率 | 场均得分（己方 / 对手） |
|---|---|---|---|---|
| 固定记忆（starter 的 `nano-bc-max-r`） | 会记忆的教师 | 81.8% | 72% | 4.44 / 2.32 |
| 仅限棋盘 | 会记忆的教师 | 81.0% | 49% | 3.46 / 3.38 |
| 学习型记忆 | 会记忆的教师 | 77.0% | 47% | 3.15 / 3.53 |
| 固定记忆 | 遗忘型教师 | 82.8% | 45% | 3.24 / 3.69 |
| 仅限棋盘（starter 的 `nano-bc`） | 遗忘型教师 | 81.9% | 37% | 2.75 / 4.12 |

在 micro 等级下，增加单蚁记忆后的相同单循环赛测试：

| 携带状态 | 标签来源 | 留出集吻合度 | 循环赛胜率 | 场均得分（己方 / 对手） |
|---|---|---|---|---|
| 仅限棋盘（starter 的 `micro-bc`） | 遗忘型教师 | 82% | 81% | 5.15 / 1.70 |
| 固定记忆（nano 等级） | 会记忆的教师 | 81.8% | 56% | 3.81 / 2.98 |
| 学习型记忆 + 单蚁记忆 | 会记忆的教师 | 88.6% | 46% | 3.37 / 3.37 |
| 无感受野（starter 的 `micro-percell`） | 遗忘型教师 | | 17% | 1.07 / 5.35 |

从中可以得出四项关键结论：
1. **一旦标签利用了记忆，记忆的价值就极其巨大**：相同等级、数据与方法下，带与不带这两个平面胜率差距为 72% 对 49%，正面交锋战绩为 11 胜 2 负，分差正来自于记住了敌方蚁穴并成功将其摧毁。
2. **吻合度（Agreement）并不代表实际战斗力**：两个模型在留出测试集上的吻合度仅差 0.8 个百分点，而固定记忆在遗忘型教师标签上的吻合度最高（82.8%），实战胜率却倒数第二。
3. **学习型记忆在 5 个 epoch 内无法赶上固定记忆**：其实战表现与仅看棋盘的模型类似，吻合度低 4 个百分点，受限于训练器的步数积累（`train/seq.py`）；在更长时间的训练得出相反结论前，固定记忆依然是最佳基线。
4. **模型容量在跨等级时彻底碾压记忆**：starter 的普通 micro 模型（参数量是 nano 的 8 倍，以平铺方式训练），以 12-2-4 的战绩击败了带固定记忆的 nano，并以 13-4-1 击败了带单蚁记忆的 micro；后者与教师的吻合度高达 88.6%（此处最高），但因优化器更新步数仅为前者的十分之一而实战偏弱。单蚁记忆证明了其通路可行，其真正价值有待步数与 bc.py 相当的训练器来充分释放。

## 运行命令（Run it）

```sh
(cd .. && ./build.sh)                    # 生成 ../dist（games.toml 解析路径）：组件、清单、参考测试集
brew tap tiny-brains/cli https://github.com/Tiny-Brains/cli && brew install tiny-brains/cli/tinybrains
                                         # 或指定本地构建：export TINYBRAINS=../../cli/target/release/tinybrains
pip install -e '.[dev]'                  # 安装环境：torch, numpy, onnx, pytest

pytest tests/ -q                                    # 准入一致性门禁；详见下文
python -m tb_baselines.collect --seat-turns 250000  # 采集教师数据集，约 9 分钟，产生 90 MB 数据
python -m tb_baselines.train.bc --class micro --epochs 6
python -m tb_baselines.train.bc --class micro --arch percell --channels 108 --blocks 2 --epochs 5 --out runs/micro-percell   # 对照组
python -m tb_baselines.collect --remember --seat-turns 250000 --out data/teacher-memory.jsonl.gz  # 记忆列的教师数据集
python -m tb_baselines.train.bc --class nano --memory --data data/teacher-memory.jsonl.gz --epochs 5 --out runs/nano-bc-max-r   # 固定记忆
python -m tb_baselines.train.seq --class nano --memory learned --data data/teacher-memory.jsonl.gz --out runs/nano-bc-learned-r   # 学习型记忆
python -m tb_baselines.train.seq --class micro --memory learned --ants --data data/teacher-memory.jsonl.gz --out runs/micro-bc-learned-ants-r   # 加上单蚁记忆
python -m tb_baselines.export --class nano --weights runs/nano-bc-max-r/best.pt --out models/nano-bc-max-r
python -m tb_baselines.train.ppo --class micro --iters 200                              # 强化学习自对弈
python -m tb_baselines.export --class micro --weights runs/micro-bc/best.pt --out models/micro-bc
python -m tb_baselines.eval models/micro-bc ../../ants-starter/models/nano-bc --boards 3
```

`export` 会将 `model.onnx`、生成的 `manifest.json`、由 `tinybrains check` 测量得到的 `metrics.json` 以及人工可读的 `card.md` 写入到 `models/<class>-<method>/` 目录（已被 git 忽略）。模型卡片中的 *Reproduce* 模块会根据数据集的 `.stats.json` 和训练运行的 `history.json` 回放还原生成该模型的完整命令；在支持该功能前的旧版本则仅记录导出命令。记忆列的运行产物按照所携带的状态及教师类型命名，因此名称即标明了其在测试表格中的对应行：`-max` 表示通过 `Max` 维护的 `planes.MEMORY`，`-learned` 表示计算图自学习的记忆，`-ants` 表示单蚁记忆，`-r` 表示由会记忆的教师生成的标签。未指定 `--out` 的 `train.bc --memory` 会写入 `runs/<class>-bc-memory`，即在*遗忘型*标签上的固定记忆模型名称（对应 45% 胜率那一行）。如果导出的模型超出其声明等级的规格上限，导出脚本将直接拒绝。值得保留的模型会提交到 ants-starter 的 `models/` 供选手作为对手测试，或作为基线模型上传到赛季中。

### 核心门禁测试（The one test that matters）

使用 Python 进行训练的参赛者需要将每个观测编码两次：一次编码在 `manifest.json` 中（供天梯平台运行），一次编码在 NumPy 中（供训练优化器读取）。一旦这两处实现产生分歧，两部分代码各自都能正常运行，但模型在竞技场中的实际得分将远低于训练曲线的预期，且系统不会输出任何明确告警。

因此，特征编码规则在 [`planes.py`](src/tb_baselines/planes.py) 中统一定义，并排包含了两套渲染实现，而 `tests/test_adapter_conformance.py` 会在卡带的官方参考观测数据上运行 **datalogic（Orion 节点执行适配器所使用的真实评估器）**（通过 `tinybrains adapt` 调用），并断言两者逐元素严格完全一致。**如果本目录下只记住一个设计思想，那就是这一条。**

该测试会加载已训练的计算图来校验清单：优先读取 `$TB_CONFORMANCE_ONNX`，否则读取本仓库同级检出的 `ants-starter/models/micro-bc/model.onnx`。若两者均不存在则测试跳过；若设置了环境变量但文件不存在则测试报错，CI 即以此方式严格执行检查。

记忆机制同样遵守该标准。记忆适配器负责传递上回合的记忆并在第 0 回合生成全 0，测试通过 `tinybrains adapt --obs` 在包含和不包含记忆的视野上进行验证，对比使用 `onnx` 构建的 4 端口图。训练器的状态更新函数 `remember` 会对照计算图在参考观测上执行的 `Max` 算子进行校验，且在安装了 PyTorch 的环境下直接对照计算图本身进行比对。单蚁记忆同样受此约束：适配器通过 ID 进行的连接操作（`to_list`、`scatter`、`concat`、`gather`，随后散布到蚂蚁所在的网格格点）通过 `tinybrains adapt --obs` 在携带前序视野蚂蚁数据行的视图上运行，其 3 个输入与训练器的 NumPy 连接操作进行逐元素比对。不支持记忆传递的旧版 CLI 会使这些测试失败并输出明确的错误提示。

## 设计规则（Design rules）

- **在小规格等级中，将预算倾斜给感受野范围而非通道宽度**。目标是每个方向至少约 15 格的感受野（视野半径约为 8.8 格），使用 3x3 卷积并在 4 层中分别采用 1、2、4、8 的空洞率（dilation）即可达成。空洞率属于 `Conv` 算子的标准属性，无需算子白名单外的特殊算子。
- **感受野呈现阈值效应而非渐进提升**。2 格上下文的对弈表现与没有上下文几乎毫无二致；因此感受野必须一次性调优达到阈值之上，一层一层递增往往会让人误以为走进死胡同。
- **交付 fp16 权重初始化器**。在每次使用时通过 `Cast` 转回 float32 会被平台运行时常量折叠（constant-fold），从而使构件文件体积减半：在相同的规格等级限制下参数量可达 2.03 倍，且对弈表现完全相同。等级大小指标衡量的是构件的原始字节数，因此没有任何理由交付 fp32 权重。
- **使用 `Resize` 进行上采样**。`ConvTranspose`（转置卷积）不在平台的算子允许白名单中。
- **记忆在学习器证明需要之前保持为固定函数**。`planes.MEMORY` 的两个平面是两个棋盘平面的 1 值集合，通过 `Max` 算子维护：无需训练，不存在 NumPy 与计算图产生分歧的风险，且作为 `i8` 每个单元格仅占 2 字节。学习型记忆（`nets.LearnedMemory`）是同样的两个平面，但由门控更新写入，需要采用按时序回放座位的训练器（`train/seq.py`），因为当前记忆是计算图在前一行的自身输出。
- **网络传输的数据类型就是训练传递的数据类型**。学习型记忆在计算图内部通过 `Floor` 和 `Cast` 舍入为 `i8`（单蚁记忆舍入为 `u8`），训练器同样通过直通估计器（STE）传递舍入后的值：模型绝不会去学习读取运行器根本无法传回的高精度浮点数。`Round` 不在算子白名单中，而 `Floor` 在。
- **数据行通过 ID 严格追踪蚂蚁**。`mine` 列表每回合都会重新排序，因此单蚁记忆是一张以 ID 为索引的表，适配器在每回合按照当前 `ids` 顺序读回（即文档中所述的 join 连接操作），且 ID 对 `planes.ANT_TABLE` 取模以确保单行能容纳进单字节中。计算图会在每只蚂蚁的数据行前写入其 ID；除此机制外没有任何手段能将数据行与具体的蚂蚁终身绑定。

## 构件清单（Artifacts）

| 构件文件 | 读取方 |
|---|---|
| `models/<class>-<method>/model.onnx` + `manifest.json` | 预签名上传：选手的参赛提交，或管理员向赛季上传的基线模型；准入系统直接从对象存储读取两者 |
| `models/<class>-<method>/metrics.json` | 人工查看以及 `eval`。平台运行时不读取此文件；准入系统会自行测量验证 |
| `models/<class>-<method>/card.md` | 人工查看 |
| `tb_baselines` Python 包 | ants-starter 的 `train.py`，通过 `git+https://github.com/Tiny-Brains/ants#subdirectory=baselines` 安装 |

**平台部署对此无任何特殊义务**：这些构件均属于普通参赛提交，无任何特权接口。一个赛季的基线模型直接在赛季管理页面上传，经过与普通提交完全相同的准入测试，初始处于禁用状态，直到管理员手动启用，在本地测试栈和生产环境中均完全一致。

## 目录结构（Layout）

```text
classes.toml                    等级规格表：每个数字及其背后的实测记录
games.toml                      解析 ../dist 目录下的卡带
src/tb_baselines/
  planes.py                     特征编码与记忆定义的核心，分别为 NumPy 和 manifest 提供实现
  adapters.py                   从 planes.py 生成 manifest.json，按需包含记忆端口
  env.py                        对接 `tinybrains env` 的环境客户端
  teacher.py                    等级天梯进行蒸馏的脚本启发式机器人；包含带记忆版本
  collect.py                    将教师对战推演采样为数据集，每行携带座位的记忆状态及其局内位点
  merge.py                      将并行采集的数据集合并为一个，隔离不同 episode
  nets.py                       各等级对应的网络架构，以及计算图可携带的记忆模块
  train/bc.py                   行为克隆（监督模仿学习）
  train/seq.py                  按回合时序执行行为克隆，用于支持计算图内部计算的学习型记忆
  train/ppo.py                  PPO 强化学习自对弈
  export.py                     torch -> ONNX -> fp16 -> 平台裁决、指标度量与模型卡片生成
  eval.py                       通过 `tinybrains <match>` 运行循环赛评估
tests/                          准入一致性门禁，以及 env 客户端测试套件
data/ runs/ models/ replays/    已被 git 忽略的运行时输出目录
```

## 核心不变量（Invariants）

- **`manifest.json` 必须由程序生成，严禁手工编辑**。它是特征编码两套渲染实现中的一种；单独手动修改它会重新引入一致性测试本意要拦截的实现偏差。
- **教师模型绝不随发布交付**。它仅仅是训练标签源。修改它会导致整个等级天梯失效，因为天梯之所以具备可比性，前提是所有等级均蒸馏自同一个教师模型。
- **价值头（Value Head）绝不导出**。评判网络（Critic）之所以可以接触到特权信息，正是因为它在任何正式对局之前就已经被彻底剔除。
- **`tinybrains env` 不是正式裁判**。无超时限制、无犯规累计、无适配器介入。来自 env 的对战成绩不代表平台正式成绩。
- **每个构件都必须注明其测量所依据的引擎摘要**：数据集头部、`metrics.json` 以及 `card.md` 的 *Measured on*。引擎变更属于规则变更，未记录对应引擎摘要的模型将无法被无损复现。这是实测记录，而非锁定项：starter 运行时采用其 `games.toml` 中固定的 `engine`，且模型卡片不会随 ants 的每次发版而盲目重新测量，因为引擎变化并不改变模型本身。
- **目录名和 `tb_baselines` 包名是面向所有 ants-starter 克隆项目的刚性外部契约**。
- **记忆是唯一的定义 `planes.MEMORY`，被渲染实现了三次**：manifest 的直通适配器、计算图内部的 `Max` 算子，以及训练器的 `remember`。数据集的一行数据在 `m` 键下携带对应座位的记忆状态，因此修改 `MEMORY` 意味着必须生成全新数据集，且 `train/bc.py --memory` 会直接拒绝在其他定义下采集的数据集。
- **单蚁记忆是唯一的连接操作 `planes.py`，被渲染实现了两次**：适配器端（通过 datalogic）与训练器端（通过 NumPy 和 PyTorch）。一致性测试严格确保前两者等价，而 `train/seq.py` 负责在训练中传递第三者。

## 已知局限与缺口（Known gaps）

- PPO 自对弈代码已实现并经过冒烟测试，但尚未经过足够长时间的完整训练以习得成熟策略。
- `small` 等级尚未产生经过充分训练的构件，且 `large` 等级目前没有网络架构：在单座位的回合时间份额内，目前没有密集型网络能在不超时的情况下触及该等级的参数上限。
- 教师模型偏向于采食而极少主动摧毁敌穴，因此等级天梯当前蒸馏出的是采食型策略。
- PPO 自对弈目前不携带记忆支持。`ActorCritic` 仅读取棋盘状态；记忆模块当前仅在行为克隆（固定或学习型）中受支持。
- 此处的等级表未声明记忆数值。赛季通过 `memory_flat_bytes` 和 `memory_cell_bytes` 设定配额，默认均为 0；记忆基线模型需要每单元格 2 字节，`export` 会汇报该开销但不会对此作出裁决。
- 携带记忆的 nano 模型占到了其预算上限的 84%，高于该等级所设定的 70% 目标：增加了两个输入通道的权重，且 manifest 增大了 263 字节。

## 许可证（License）

Apache-2.0，与 ants 其余部分一致：详见 [LICENSE](../LICENSE)。
