# ants

Ants 是 TinyBrains 的基准游戏卡带（game cartridge）：一个基于环面网格的确定性蚁群对抗游戏，源自 2011 年 Google AI Challenge。本仓库负责构建平台所运行的 WebAssembly 组件、元数据清单、五个基础棋盘以及回放查看器，并将训练 Ants 参赛模型的流水线（[`baselines/`](baselines/README.md)）置于其编码的规则旁共同维护。GitHub Releases 中发布的单个 `dist/` 归档包供所有下游消费端使用：Soma、Kalam、web 及其手册、`tinybrains` CLI 以及 ants-starter。

该组件对底层运行平台一无所知：没有积分天梯、没有准入验证、没有具体模型、没有调度逻辑。模型所观察到的数据、模型所提交的响应，以及卡带所必须遵守的契约，均发布在参赛者指南中（[`web/docs`](https://github.com/Tiny-Brains/web/tree/main/docs)：*What your model sees*、*What your model answers*、*Adding a game* 以及 *Ants* 规则章节）。

## 构建与测试

无需配置任何服务器、数据库、容器或常驻运行的平台环境。

| 命令 | 作用 | 依赖环境 |
|---|---|---|
| `./build.sh` | 确定性检查、引擎宿主测试与 `mapgen` 测试，随后将全部构件打包到 `dist/`；输出引擎摘要 | rustup（`rust-toolchain.toml` 固定了 rustc 与 `wasm32-unknown-unknown` 目标）、`build.sh` 中指定 `WASM_TOOLS_VERSION` 的 `wasm-tools`、Python 3.11+ |
| `viz/build.sh` | 使用 `jco` 将组件转译到 `dist/viz/`，复制查看器代码，运行 `viz/check.mjs` | Node；需先运行 `./build.sh` |
| `tools/pack.py DIR` | 将 `dist/` 打包为 `DIR/ants-artifacts.tar.gz`；输出引擎、归档与内容的摘要及发布所用的 tag | 完整的 `dist/`（包含查看器） |
| `tools/deny.sh` | 单独运行确定性检查：确保 `engine/src/` 中不包含浮点运算 | |
| `tools/equivalence.py OLD NEW` | 在随机策略和贪心策略下，通过两个 dists（已解压的发布版与当前 `dist/`）执行带种子的相同对局，逐回合比对每个视角、得分和结局；`--allow` 指定允许其中一方携带的额外视野键 | `tinybrains`, numpy |
| `(cd engine && cargo test)` | 单独运行宿主测试套件 | |
| `(cd engine && cargo fmt --check && cargo clippy --all-targets -- -D warnings)` | 代码风格与静态分析检查；CI 会在 `mapgen/` 中运行相同检查 | |
| `(cd viz && node check.mjs)` | 查看器检查：几何、步数计数、领地、标签、CSS 作用域隔离 | Node, `dist/` |
| `(cd baselines && pytest tests/ -q)` | 编码一致性门禁（[baselines](baselines/README.md#the-one-test-that-matters)） | `dist/`, `tinybrains`, 本仓库同级的 ants-starter 检出或 `TB_CONFORMANCE_ONNX` |

`engine/` 和 `mapgen/` 是各自独立的 Cargo 工作区；根目录下没有 `Cargo.toml`。

**引擎摘要（engine digest）是 `tb-ants.wasm` 的 sha256 散列值**，它由四个要素决定：源码、精确的 rustc 版本、精确的 `wasm-tools`（它会将自身版本印记写入组件中）以及运行 rustc 的宿主环境。`rust-toolchain.toml` 与 `build.sh` 固定了前三者并重映射了所有构建路径，因此代码检出的绝对路径无关紧要。宿主环境则无法被锁定：在 Apple-silicon Mac 上构建的代码与在 arm64 Linux 上构建的代码，其函数排布顺序不同。官方发布构建统一在 `aarch64-unknown-linux-gnu` 上进行，因此只有在该宿主上构建的哈希值才能与官方发布摘要一致。请始终以 CI 工作流生成的摘要为准，切勿引用本地笔记本电脑上的构建摘要。

## 构件清单（Artifacts）

`./build.sh` 与 `viz/build.sh` 生成输出到 `dist/`（已被 git 忽略），官方发布包中包含的归档与该目录结构完全相同。

| 路径 | 内容说明 | 生成方式 |
|---|---|---|
| `tb-ants.wasm` | Wasm 组件；其 sha256 即为引擎摘要 | `cargo build` + `wasm-tools component new` |
| `plugin.toml`, `plugin.json` | Orion 插件 ABI，手写源文件及派生生成的 JSON | `engine/plugin.toml`；`tools/package.py` |
| `cartridge.json` | 规格限制（`limits.boards` 是赛季棋盘必须满足的包络）、适配器预算、基础棋盘目录、`about` 信息 | `engine/src/bin/manifest.rs`；`tools/package.py` 追加 `maps`、`limits.boards` 与 `engine/about.json` |
| `maps/` | 提交到仓库中的五个基础棋盘 | 从 `maps/` 复制 |
| `reference/observations.json` | 准入系统用于验证适配器的参考观测数据，基于基础棋盘绘制 | `engine/src/bin/reference.rs` |
| `viz/` | 查看器捆绑包，以及 `engine.json`（记录转译自哪个引擎摘要） | `viz/build.sh` ([viz/README.md](viz/README.md)) |

该组件属于插件 `tb.ants`，ABI 版本为 `orion:plugin@1.0.0`。`engine/plugin.toml` 声明了各个导出函数的输入规范。

| 导出函数 | 输入参数 | 返回结果 |
|---|---|---|
| `tb.ants.worldgen` | 随机种子与棋盘数据（`map`，或每个种子对应的 `maps`）；可选 `players` 和 `max_turns` | 初始打包的 `wave_state` |
| `tb.ants.observe` | `wave_state` 以及每个座位的不透明 `refs` | 每个存活座位的视野视角，原样回显对应的 ref |
| `tb.ants.step` | `wave_state` 与行动指令 actions | 下一轮 `wave_state`、`done` 完成列表、`ended` 结束列表与回放增量 delta |
| `tb.ants.finish` | `wave_state` | 比赛排名、得分、结束原因以及各场对局使用的棋盘 |
| `tb.ants.replay-decode` | 回放包络，以及指定的 `turn` 或 `from`/`to` 回合范围 | 单帧数据，或单次遍历获取的一批连续帧 |

`wave_state` 对所有外部调用者均保持不透明。行动指令必须遵循最后一次观测的顺序，或者显式声明 `{m, seat, action}`。`replay-decode` 中的连字符是函数标识符的一部分。

## 使用本地构建或发布版本

可将游戏注册表指向检出目录中的 `dist/` 进行对战或校验：

```toml
[games.ants]
name = "Ants"
path = "../ants/dist"
```

解压后的发布归档具有完全相同的目录树结构，并且是官方权威版本：

```sh
gh release download --repo Tiny-Brains/ants --pattern ants-artifacts.tar.gz && tar -xzf ants-artifacts.tar.gz -C dist
```

注册表还可以锁定特定发布版本，`tinybrains` 会一次性下载并校验归档文件和内部组件的摘要哈希：

```toml
[games.ants]
name = "Ants"
repo = "Tiny-Brains/ants"
release = "engine-<12 hex>"
artifacts = { file = "ants-artifacts.tar.gz", sha256 = "sha256:..." }
engine = "sha256:..."
```

若需在未发布的引擎版本上构建 Soma、Kalam、web 或手册，可在 Docker 构建命令中传入 `--build-context ants=<此 dist/ 路径>`。

## 版本发布（Releasing）

`build` 工作流（`.github/workflows/build.yml`）是唯一的发布构建源。推送到 `main` 分支的每次提交都会在 `ubuntu-24.04-arm` 运行器上执行门禁检查、构建所有构件、打包 `dist/`、运行基线一致性测试、使用构建产物与 ants-starter 对战，并汇报此构建与哪次发布完全一致（字节级复现）或是发布时将切出的 tag。这两项均使用从 `cli` 仓库的 `main` 分支构建的 CLI，而非最新发布版，因此对局文件格式的变更优先在此发布：随后 ants-starter 的 `games.toml` 同步更新并与 CLI 一同发布，进而 starter 的 CI 负责验证参赛选手安装的 CLI。

```sh
gh workflow run build.yml                   # 试运行：执行上述所有检查，不发布
gh workflow run build.yml -f publish=true   # 从 main 分支正式切出发布（若尚无包含该构建的发布）
```

发布包文件为 `ants-artifacts.tar.gz`，标签命名为 `engine-<摘要前12位十六进制>`。若同一引擎版本的归档发生变动（例如修复了查看器或更新了参考观测），则标签为 `engine-<12位十六进制>-2`。工作流在 gzip 压缩前比对 tar 内容以识别构建重复性。Tag 一旦生成绝不重新打标：各注册表均锁定了归档文件的字节哈希。

**发布即部署**。Soma、Kalam、web 和手册的镜像在构建时会自动拉取最新发布（除非通过 `ANTS_RELEASE` 显式锁定了 tag），因此各服务的下一次镜像构建无需本仓库提交即可直接接入新引擎。正在进行在线赛季的部署环境必须锁定 `ANTS_RELEASE`。当发布了包含新引擎摘要的版本后：

1. 基于同一发布包重新构建 Soma、Kalam 和 web（含手册）。基于不同摘要的运行器无法互相认同，而基于不同摘要的查看器会渲染出从未发生过的对局。
2. 重新签署插件签名（运行 web 的 `scripts/setup/sign-plugins.sh`）。若组件的 Ed25519 签名不匹配，节点启动时将进入 `degraded` 降级状态。
3. 声明新版本。在规则未发生变动时，Soma 的 `bootstrap` 会将其镜像携带的摘要声明为 patch 补丁（在线赛季可平滑接受）；若设置了 `ENGINE_RELEASE=1` 则视为规则变更，在线赛季期间将拒绝接收。
4. 在 ants-starter 的 `games.toml` 中升级发布配置块；release notes 中已提供对应内容。
5. 重新捕获 web 的 `docs/tutorials/replays/real-match.json`（来自在新引擎上运行的真实对局）。该文件是从运行中的服务栈捕获的真实比赛记录，若其过时，手册的 `tutorials/build.sh` 将报错拒绝。

## 棋盘（Boards）

本仓库在 `maps/` 下维护**五个基础棋盘**，每个尺寸级别一个，绝无其他棋盘：

| 棋盘名称 | 尺寸 | 座位数 |
|---|---|---|
| `basic-tiny-2p` | 24 × 24 | 2 |
| `basic-small-3p` | 36 × 36 | 3 |
| `basic-medium-4p` | 48 × 64 | 4 |
| `basic-large-6p` | 80 × 96 | 6 |
| `basic-xlarge-8p` | 120 × 124 | 8 |

**它们构成了规格包络（envelope）**。`tools/package.py` 根据它们覆盖的范围导出 `limits.boards`（2 到 8 个座位，边长 24 到 124，单元格上限 14,880 个），参考测试集也绘制在它们之上，因此赛季棋盘必须容纳在该规格范围内，Soma 会直接拒绝超出包络的棋盘。Wasm 组件本身不包含任何棋盘：`worldgen` 从调用方接收完整棋盘并进行校验。

**棋盘属于设计产物；切勿手工编辑棋盘**。`mapgen/recipes/<name>.toml` 是记录为数据的单张棋盘设计：包括尺寸、座位平移量、绕各座位中心的装饰点群、绘制在上面的所有形状与图案及其随机种子，以及食物分布方案。`mapgen/src/design.rs` 负责具体渲染。

```sh
cd mapgen
cargo run -- generate                    # 将 recipes/ 下的每个配方生成到 ../maps/，清理无配方对应的棋盘
cargo run -- generate recipes/basic-tiny-2p.toml
cargo run -- check                       # 检查每个已提交的棋盘是否与对应配方生成的产物逐字节完全一致
cargo run --release -- check --play 6    # ...并在每个棋盘上运行对弈测试，所有座位运行相同的行动策略
cargo run -- show ../maps/basic-tiny-2p.json
cargo run --release -- explore --slots slots.toml --out run --n 500   # 批量提议生成数百个候选设计
cargo run --release -- playtest run picks.txt                         # 对选中的候选设计进行试玩测试
cargo run -- adopt run picks.txt                                      # 将挑选的设计固化并导出为配方
cargo run --release -- sweep             # 对称公平性扫描：覆盖多种风格、2-8个座位、进行同构对弈
```

`explore` 负责提议候选，由人工目测挑选；`adopt` 将挑选结果固化，从而无论采样算法后续如何调整，已确定的棋盘都能保持不变。卡槽配置中的 `size` 可以是一个分类（`tiny`、`small`、`medium`、`large`、`xlarge`，会在该类别的边长范围内生成），也可以是写为 `ROWSxCOLS` 的精确尺寸（如 `size = "36x44"`，用于尺寸已确定、只需挑选该尺寸下最佳棋盘的场景）。精确尺寸会被视为固定前提：采样器根据其地形偏好与长宽比例来提议生成，而非强制使用预设形状。`build.sh` 会运行 `mapgen` 的测试套件，重新渲染所有已提交棋盘并严格比对字节一致性。

**赛季棋盘不存放在本仓库**。它们使用相同的工具并指定外部目录生成（`--recipes DIR --maps DIR`），由管理员上传到赛季平台，且仅在赛季结束关闭后才会备份到归档仓库。本仓库从不提交任何赛季棋盘，发布包中也绝不携带。

## 目录布局（Layout）

```text
engine/                    游戏卡带：编译为 Wasm 组件的 Rust crate
  src/lib.rs               插件分发：五个导出函数及其 JSON 格式，不含游戏规则
  src/grid.rs              环面网格、距离计算、对称性、移动方向、位图、种子随机数生成器
  src/maps.rs              棋盘文件：解析、验证以及调用方传入数据的处理
  src/state.rs             单场对局运行中的状态
  src/turn.rs              固定顺序的回合结算、步数截止计数器、结束条件判定、排名
  src/food.rs              隐式速率下的食物逻辑、打乱的对称集合
  src/observe.rs           座位视野视角、已揭示水域与迷雾
  src/codec.rs             压缩打包并 base64 编码的 wave_state
  src/replay.rs            动作流录制与重模拟为动画帧
  src/authoring.rs         仅宿主运行：参考观测数据
  src/bin/                 manifest 与 reference 生成器（args/ 为共享参数解析）
  src/tests/               宿主测试套件，按领域划分文件；fixtures/ 存放平台生成的真实回放
  plugin.toml              手写定义的 Orion ABI
  about.json               游戏介绍说明，合并入 cartridge.json
mapgen/                    棋盘工厂：独立的 crate，调整它不会变动引擎摘要
  recipes/                 五个基础棋盘各自的设计配方
  src/main.rs              命令行子命令：generate、check、show、explore、playtest、adopt、sweep
  src/design.rs            渲染器：在全群变换下执行设计步骤、蚁穴、修整、食物
  src/sym.rs               座位平移、各中心点群、轨道、点阵结构
  src/sample.rs            采样器：输入槽位与种子，输出设计草案
  src/explore.rs           候选探索与试玩：为每个槽位生成多组设计并渲染、校验、打分
  src/grid.rs              生成器共享的环面、轨道与随机数生成器
  src/measure.rs           棋盘各项量化指标，以及每张棋盘必须遵守的规则约束
  src/recipe.rs            `sweep` 测试所使用的区域生成配方
  src/make.rs              区域生成器：区域、墙壁、门、蚁穴、填充
  src/set.rs               区域生成器集合、文件格式与引擎验证逻辑
  src/play.rs              棋盘对战测试：同构一致性检查与贪心探索冒烟测试
  src/sweep.rs             覆盖多种风格与座位数的公平性全量扫描
  src/tests.rs             mapgen 的自动化测试，由 build.sh 执行
viz/                       回放查看器：为 Web 前端、教程手册和 CLI 提供的统一捆绑包
baselines/                 Ants 参赛选手的训练流水线（Python）；无模型权重，不包含在发布包中
maps/                      五个基础棋盘，每张棋盘一个 JSON 文件
tools/deny.sh              确定性检查脚本
tools/package.py           完善 dist/：生成 plugin.json、棋盘目录与 limits.boards、拷贝棋盘、生成报告
tools/pack.py              将 dist/ 确定性打包为发布的单一归档文件并计算摘要
tools/equivalence.py       在两个 dists 间逐回合运行对弈比对：证明重新构建未改变游戏规则
build.sh                   门禁检查并将所有构件输出到 dist/
rust-toolchain.toml        精确固定的 rustc 版本，属于引擎摘要的计算输入
.github/workflows/build.yml 每次 push 时的门禁与构建；按需执行正式发布
dist/                      构建产物输出目录，已被 git 忽略；与发布归档包含的文件完全一致
```

## 核心不变量（Invariants）

- **游戏逻辑严格使用整数运算**。`tools/deny.sh` 会禁止 `engine/src/`（除 `tests/` 和 `bin/` 外）中出现任何浮点类型或操作。这是确保平台运行环境与浏览器端二进制级完全一致、且回放能作为纯动作流存储的核心保证。
- **种子、动作与棋盘决定比赛结果**。不存在外界随机性，不存在系统时钟。天梯对战由 Soma 的配对时钟决定棋盘与随机种子，选手无法指定棋盘进行针对性训练。
- **在 `engine/src/`（除 `tests/` 和 `bin/` 外）或 `engine/plugin.toml` 下的任何修改，都会导致全新的引擎摘要**，即使仅修改注释也是如此（因为代码中 panic 错误信息携带了精确行号）。新摘要属于全平台发布和天梯重置事件。
- **探索历史会被保留**。模型接收单次观测，并在其类别允许记忆时接收其上回合写入的内容；引擎会将每个座位的视野合并至已探索状态中，观测数据中会始终携带已知水域。
- **蚂蚁在其生命周期内保持固定 ID**。视野中的 `ids` 为 `mine` 中的每只蚂蚁赋予每个座位独立自 0 起计数的终生编号，永不重复分配。`mine` 本身保持行优先排序，不代表蚂蚁身份。按座位独立编号保证了座位的对称性，`sweep` 会对此进行验证。视野本身绝不包含 `memory` 或 `ant_memory`（这两个键归运行器专有）。
- **观测视角相对于观察者对称（observer-relative）**。重新标记每个座位号并重新询问同一选手，其接收到的字节数据完全一致（`a_view_is_observer_relative`）。
- **无论由谁提供，每个棋盘都必须绝对公平且连通完整**。`engine/src/maps.rs` 会对传入 `worldgen` 的每一个棋盘（包括赛季上传的棋盘）进行严格校验：确保其在自身平移量下同构对称、陆地构成单连通可行走整体、且每个蚁穴都有通畅的出口。
- **回放包络携带其对局所在的完整棋盘及种子**，因此无论该赛季后续如何变迁，回放都能被无损重新模拟。解码测试运行在平台实际生成的包络数据上。
- **查看器始终通过组件本身进行重新模拟，绝不复制代码逻辑**。用 JavaScript 重新实现任何游戏规则都会引入第二套引擎。
- **生成产物绝不提交，且所有生成产物均输出在 `dist/` 中**。下游消费端通过相对路径读取 `dist/`，因此重命名其中的文件需要同步修改所有消费者与 CLI。
- **通信协议公开透明，并与引擎同步演进**。视野字段的任何变动都必须在同一批次中同步更新参赛者指南与 `baselines/src/tb_baselines/planes.py`。
- **游戏规则以 2011 年竞赛引擎为准**。若公开规范与竞赛原始引擎（`aichallenge/ants/ants.py`）产生分歧，以竞赛引擎为准。Focus Battle 页面的推导案例作为 `spec_scenario_*` 测试包含在代码中。
- **`baselines/` 与 `tb_baselines` 包是公共依赖路径**。ants-starter 通过 `git+https://github.com/Tiny-Brains/ants#subdirectory=baselines` 使用 pip 进行依赖安装。

## 已知局限与缺口（Known gaps）

- 尚待进行 10,000 场跨宿主环境的一致性压力测试：目前平台运行时与浏览器端之间的确定性比对仅在小规模用例下进行了验证。
- web 的 `scripts/check/configs.sh` 会比对 Kalam 引擎与教程手册查看器，但未覆盖 web 自身的查看器或 Soma 镜像，且目前没有比对 ants-starter 锁定的 `engine`。
- 观测视野中目前不携带当前回合数（turn number）。残局决策是否需要回合数尚未经过实战检验。
- **`ant_memory` 的 ID 表在 256 处回绕，导致新蚂蚁可能继承已阵亡蚂蚁的历史字节**。蚂蚁 ID 按座位累计递增，很快就会超过 `ANT_TABLE` 阈值——在 `basic-xlarge-8p` 上所有存活蚂蚁在 1000 回合前就已超过该范围——且槽位在阵亡后不会清空，因此“新蚂蚁读取全 0”并不成立。由于适配器、numpy 渲染和神经网络图的回绕行为完全一致，天梯运行器与训练器保持了一致，`test_adapter_conformance` 依然成立；所损失的是单只蚂蚁的长效身份标识，而非确定性。扩大该键意味着在 `ant_memory` 行中增加第二个字节，需要引入全新的清单定义、新的图结构并重新训练 `nano-bc-max-r` 模型——因此这是权衡后的刻意延期处理，而非疏忽遗漏。

## 许可证（License）

Apache-2.0：详见 [LICENSE](LICENSE)。
