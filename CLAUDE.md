# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指引。

`ants` 是 TinyBrains 的基准游戏卡带（game cartridge）：位于 `engine/` 下的一个 Rust crate，编译为 WebAssembly 组件（`tb-ants.wasm`，插件名称 `tb.ants`，ABI `orion:plugin@1.0.0`），供 Kalam 运行对局并供 Soma 验证棋盘。它不导入任何外部模块：无服务器、数据库、网络或时钟。它拥有 Ants（蚂蚁对战）的全部游戏规则，且对运行平台一无所知。其同级还有三个不属于组件、拥有独立工具链的目录：`mapgen/`（棋盘工厂，仅在宿主环境运行的 Rust crate）、`viz/`（回放查看器，Node）以及 `baselines/`（训练管线，Python，包含独立的 [CLAUDE.md](baselines/CLAUDE.md)）。基线管线深入了解平台细节，但这完全没有问题，因为确定性检查、构建和发布流程从不涉及它们。`README.md` 是面向人类的指南（包含构建、构件、发布、棋盘、结构布局与不变量）；父级 `../CLAUDE.md` 则涵盖所有跨仓库边界的内容。通信协议发布在参赛者指南中（`../web/docs/src/models/observation.md`、`actions.md`、`platform/adding-a-game.md` 以及 `games/ants/` 下的规则），而非保存在本仓库。

## 检查与门禁（Checks）

依赖项：rustup（`rust-toolchain.toml` 固定了 rustc 版本与 wasm32 目标架构）、`build.sh` 中 `WASM_TOOLS_VERSION` 指定版本的 `wasm-tools`、Python 3.11+，以及用于查看器的 Node。不需要 Docker。`engine/` 和 `mapgen/` 是各自独立的 Cargo 工作区，因此每个 `cargo` 命令都必须在其对应的子目录中运行。

```sh
./build.sh                  # 执行 deny.sh、engine 测试、mapgen 测试，随后将所有构件输出到 dist/；打印引擎摘要
viz/build.sh                # 将查看器构建到 dist/viz/，随后运行 check.mjs（需在 ./build.sh 之后运行）
(cd engine && cargo fmt --check && cargo clippy --locked --all-targets -- -D warnings)
(cd mapgen && cargo fmt --check && cargo clippy --locked --all-targets -- -D warnings)
(cd viz && node check.mjs)  # 单独检查查看器
(cd viz && node make-last-frame.mjs)  # 重写保存的最后一帧；在引擎摘要发生变动后必须执行
(cd baselines && pytest tests/ -q)   # 编码一致性门禁；参见 baselines/CLAUDE.md
tools/pack.py DIR           # 将 dist/ 打包为发布归档，输出其摘要及发布的 tag 标签
tools/equivalence.py OLD dist --allow ids   # 将解压后的旧发布版本与 dist/ 在两种策略下逐回合对弈对比
 
cd engine
cargo test a_replay_re_simulates_the_match_it_recorded    # 按名称执行单个测试
cargo test spec_scenario_                                  # 规范说明中演算战斗的测试用例
cargo test measure_what_random_play_produces -- --ignored --nocapture   # 诊断测试，非保证性测试；耗时数分钟
cargo run --bin reference -- --only basic-small-3p:20260919:400          # 生成单个参考观测规范
cargo run --quiet --bin manifest > ../dist/cartridge.json   # 在修改 about.json 或 limits 后单独生成清单
```

`build.sh` 会首先清理 `dist/`，这会特意连带删除 `dist/viz/`：因为由旧组件转译出的查看器对应的只是其他版本的引擎。运行 `./build.sh` 后请重新执行 `viz/build.sh`。

**上述门禁中有三处需要读取本仓库之外的资源**，因此完整的检查套件需要 PATH 中存在 `tinybrains`（或配置了 `$TINYBRAINS`），并在本仓库同级目录下检出 `ants-starter`：
`viz/make-last-frame.mjs` 使用 CLI 与 starter 中的 `models/`（`$STARTER_MODELS`）进行真实对局；`tools/equivalence.py` 需要支持两个 dists 并依赖 numpy 的 CLI；基线一致性测试若缺少 starter 的 `micro-bc` 计算图则会**跳过（skip）**而非报错——这就是 CI 在 `TB_CONFORMANCE_ONNX` 中显式指定它的原因。CI 中的该 CLI 是从 `cli` 仓库的 `main` 分支构建的，而非其最新发布版本，因此 `cli` main 分支损坏会导致本仓库的构建失败。

mapgen 命令详见 README §Boards。一个赛季的棋盘生成命令与此相同，只是指向所有代码仓库之外的路径：

```sh
cd mapgen
cargo run -- generate --recipes ../../maps/recipes --maps ../../maps
cargo run -- check    --recipes ../../maps/recipes --maps ../../maps
cargo run --release -- sweep --seats 8 --styles cave,rooms --boards 10 --turns 1000 --json out.json --export dir
```

## 规则与约束（Rules）

**仅有五个导出函数，全部在 `engine/src/lib.rs` 中按名称分发，绝无其他导出：**

```
tb.ants.worldgen(seeds[], map | maps, players?, max_turns?)  → wave_state
tb.ants.observe(wave_state, refs)      → 每个座位的视野视角，各自回显其不透明 ref
tb.ants.step(wave_state, actions)      → { wave_state, done[], ended[], replay_delta }
tb.ants.finish(wave_state)             → 比赛排名、得分、结束原因以及所使用的棋盘
tb.ants.replay-decode(payload, turn | from,to)  → 单帧数据，或单次遍历获取一个区间
```

`replay-decode` 中的连字符是至关重要的结构组成（Orion 在函数标识符中拒绝 `[^a-z0-9-]`）。
`refs` 是一个**扁平**列表，携带各自的 `m`/`seat`，直接匹配并原样回显，绝不进行解析；
`actions` 可以与上一次 `observe` 的位置顺序保持一致，或者显式指定 `{m, seat, action}`。
*wave* 表示在单次调用中同时推进的多场比赛批次。

**模块职责划分**（位于 `engine/` 下）。必须严格保持此分离：

| 文件 | 职责 |
|---|---|
| `src/lib.rs` | 插件分发与 JSON 结构定义。不包含任何游戏规则 |
| `src/grid.rs` | 环面几何、距离、对称性、方向、视野/攻击/生成半径、`Bits`、`Rng` |
| `src/maps.rs` | 棋盘文件解析与验证（棋盘自身的平移变换、完整的蚁穴轨道、连通可行走的陆地），以及解析调用方传入的内容：仅接收棋盘对象 |
| `src/state.rs` | 进行中的单场比赛状态，包括蚂蚁 ID 及每个座位的 `next_id`；截止步数计数器常量 |
| `src/turn.rs` | 固定顺序的回合结算：移动 → 攻击 → 摧毁蚁穴 → 生成蚂蚁 → 收集食物 → 生成新食物；对局结束判定；排名 |
| `src/food.rs` | 隐式速率下的食物逻辑：速率、对称分布集、打乱旋转、待生成队列 |
| `src/observe.rs` | 单个座位的视角：战争迷雾、已知水域、`vis`、`ids`、相对于观察者的阵营归属 |
| `src/codec.rs` | 打包的 base64 编码 `wave_state`。在此文件外部完全不透明；无版本字段 |
| `src/replay.rs` | 动作流录制与重模拟为动画帧 |
| `src/authoring.rs` | 仅宿主运行：参考观测数据。插件调用无法触及，链接器会将其从 Wasm 组件中剔除 |
| `plugin.toml`, `about.json` | 编写的 ABI 定义，以及合并入 `cartridge.json` 的游戏简介 |

**解释大部分设计的四条核心属性：**

1. **游戏逻辑中仅使用整数运算**，从而确保平台构建与浏览器环境在二进制位级完全一致，这也是回放可以存储为动作流而非全量帧的基础。`tools/deny.sh` 会在 `engine/src/` 中检索浮点构造，排除 `tests/` 和 `bin/`（它们属于宿主工具，绝不编译入组件）。Rust 编译器本身无法强制此约束。
2. **种子、动作与棋盘决定整场比赛**。不存在环境随机性，不存在时钟。由调用方指定棋盘；在天梯上，由 Soma 的配对时钟选择棋盘并分配随机种子，因此选手无法针对自己挑选的棋盘进行定向训练。
3. **探索历史会被保留**。模型接收单次观测结果，并在其类别允许记忆时接收其上回合写入的数据。某些类别可能不允许记忆，因此 `turn.rs` 会在每回合将每个座位的视野合并入 `known`，且观测结果中携带*已知水域（known water）*。每个玩家的已探索掩码占据了 `wave_state` 的绝大部分体积；这种开销正是设计本意。**哪只蚂蚁对应哪个 ID 取决于视野中的 `ids`**，而非 `mine` 的顺序：`mine` 始终保持行优先顺序（所有适配器和策略头部均以此对齐），并在每回合重新排序；而 id 是每个座位独立的，从 0 开始按照蚁穴列表随后是生成顺序计数，在所有移动（包括被阻挡的移动）中保持不变且永不重复分配。每个座位独立编号保证了座位的对称性。只有自己的蚂蚁携带 id：对手的 id 会泄露其历史生成数量。运行器将模型的记忆与视野一同在 `memory` 和 `ant_memory` 键下交回，因此视野本身绝不携带这两个键。
4. **棋盘是数据文件，组件本身不包含任何棋盘**。`maps/` 下存放五个基础棋盘（属于数据内容而非源代码：由 `mapgen/` 渲染，由 `dist/maps/` 发布）。**它们构成了规格包络（envelope）**：`limits.boards` 是它们所跨越的范围，参考测试集基于它们绘制，且上传到 Soma 的赛季棋盘（永不提交到本仓库）必须容纳在其中。无论由谁提供，每个棋盘在对弈前都会经过验证。棋盘携带其自身的**平移量（shift / `symmetry`）**：座位 `k` 的棋盘是座位 0 的棋盘应用该平移量 `k` 次的结果，该属性随 `wave_state` 传递，`food::sets` 和观察者相对所有者均遵循该规则。蚁穴按轨道列出，因此第 `i` 个蚁穴属于座位 `i % players`，且棋盘自身声明了其座位数。回放包络携带其对应的棋盘*以及*种子，因此可以无损重新模拟该赛季棋盘上发生的一切。

**其他规则：**

- **`turn.rs` 中的回合顺序是规则本身**，而非实现细节，且 `wave_state` 对所有调用方保持绝对不透明。
- **单一引擎**。查看器通过 jco 转译的组件进行重新模拟。用 JavaScript 重新实现任何规则，或试图解析 `wave_state` 的工作流，都会引入第二套引擎。
- **规则修改必须引用参考基准**。在公开规范文档与 2011 年竞赛引擎（`aichallenge/ants/ants.py`）存在分歧之处，**以竞赛引擎为准**：因为所有参赛机器人的实际评分均以它为准。代码注释中保留了 `ants.py:NNN` 行号引用；请保持这一习惯。Focus Battle 页面演算示例作为 `spec_scenario_*` 测试包含在内。
- **协议发布在参赛者指南中，而非本仓库**。对视野字段、动作字母表或 ABI 输入的任何修改，都必须在同一批次中同步修改 `web/docs`。测试 `a_view_carries_exactly_the_fields_a_model_is_promised` 会针对引擎实际输出校验视野格式，确保意外新增的字段会触发测试失败而非破坏线上适配器；`a_view_never_carries_the_keys_the_runner_writes_memory_into` 确保 `memory` 和 `ant_memory` 始终归运行器专有。
- **观测格式的变动属于基线变动**。`planes.py` 负责对 `engine/src/observe.rs` 输出的内容进行编码，且一致性测试运行在 `dist/reference/observations.json` 上，因此对两者的修改必须与编码更新及其重新生成的清单包含在同一个提交中。
- **棋盘属于设计产物；切勿手工编辑棋盘**。`mapgen/recipes/<name>.toml` 是写出的设计数据；`design.rs` 负责渲染它且是契约；`sample.rs` 仅负责采样提议，因此改进采样器不会重新绘制已被采纳的棋盘，而修改渲染器则会重绘所有棋盘。`check` 命令会重新渲染所有配方并按字节对比，因此任何手工编辑或未重新生成的渲染器变更都会导致 `build.sh` 失败。`recipe.rs`、`make.rs` 和 `set.rs`（区域生成器）仅保留给 `sweep` 以及它们自身的测试。`mapgen` 会通过 `tb.ants.worldgen` 验证其写出的每个棋盘，这与赛季上传在 Soma 节点上运行的检查完全一致。**配方同样不是手写的**：棋盘的制作流程是 `explore` → `playtest` → `adopt`（见 README §Boards），它会提出数百个设计，人工用肉眼挑选并试玩，最后将选中的方案写入为配方文件。
- **`sweep` 是对称公平性证明，而非性能基准**。每个座位运行同一个策略（仅读取其自身视野转换到座位 0 坐标系后的数据），因此在公平规则与公平棋盘下，每个座位的视野在每个回合都应当是座位 0 视野经平移后的结果，且所有对局必须平局结束。出现偏差即为 Bug，会报告裁判棋盘首次破坏对称性的回合（`MAPGEN_TRACE=1` 会绘制对应方格）。在修改任何可能对不同座位区别对待的规则后，必须运行该测试。
- **提交直接合入 `main`**。提交信息应采用平台语气的完整句子（例如 "Separate the engine from the tools that generate its artifacts"），而非 conventional-commit 前缀。
- **文档注释必须阐明原因（Why）**：被拒绝的替代方案、防止的潜在故障、所实现的参考行。测试命名应为所强制执行规则的完整句子（例如 `a_map_that_is_not_symmetric_is_refused_rather_than_played`），在 `engine/src/tests/` 下每个领域一个文件。

## 关键陷阱与易错点（Gotchas / What breaks）

- **在 `engine/src/`（除 `tests/` 和 `bin/` 外）或 `engine/plugin.toml` 下的任何修改，都会产生全新的引擎摘要**，即使仅修改注释也是如此（panic 位置包含代码行号）。`games.active_engine_digest`、每个副本节点的 `engine_digest`、插件签名、`viz/engine.json` 以及保存的最后一帧（`viz/last-frame-basic-xlarge-8p.json`）都会随之变动，这属于发布事件和天梯重置事件。必须在提交中予以说明。
  要证明重构未改变任何规则，请将解压后的发布归档中的 `cartridge.json`、`plugin.json`、`reference/observations.json` 和 `maps/` 与当前输出进行 diff 对比，并使用 `tools/equivalence.py` 在随机和贪心策略下逐回合对弈对比两者的全部视角、得分和结局，使用 `--allow` 列出修改新增的字段键。可能影响座位对称性的修改还必须运行 `mapgen sweep --seats 8`。
- **`build` 工作流生成的摘要才是平台官方认可的**。组件的二进制字节取决于四点：源码、rustc（精确版本；补丁升级会导致字节变动）、`wasm-tools`（精确版本；其自身版本会打入组件中）以及 **运行 rustc 的宿主环境**。`build.sh` 重映射了 crates.io、标准库和检出目录路径，因此代码检出的绝对路径无关紧要；但宿主环境有决定性影响，没有任何编译器参数可以抹除这一差异。官方发布在 `aarch64-unknown-linux-gnu`（`ubuntu-24.04-arm`）上构建，因此将运行器切换到 x86-64 会在没有规则变更的情况下导致引擎摘要改变。本地笔记本电脑上构建的摘要绝不会与天梯官方摘要一致。
- **发布即部署**。Soma、Kalam、web 以及 web/docs 的 Dockerfile 在构建时除非通过 `ANTS_RELEASE` 锁定特定 tag，否则均会自动拉取最新发布版本，因此一次发布就是它们下一批镜像所声明、运行和展示的引擎版本，甚至会直接影响正在进行的赛季。
- **新的引擎摘要即意味着新版本发布，否则参赛者将继续在旧引擎上对弈**。ants-starter 的 `games.toml` 通过归档摘要与 `engine` 摘要锁定特定版本，且没有本地检出可回退。运行 `gh workflow run build.yml -f publish=true` 会触发发布并在 release notes 中生成供粘贴的配置块。切勿重复发布同一个 tag。Tag 格式为 `engine-<12位十六进制>`，若同一引擎发布不同归档（如修复查看器）则为 `engine-<12位十六进制>-2`；工作流会在打包 tar 之前比对内容，若当前构建与已有发布完全相同则会明确提示而非切出新 tag。
- **没有任何自动化机制检查下游消费端是否版本一致**。web 的 `scripts/check/configs.sh` 仅比对 Kalam 的引擎与教程手册的查看器，但不比对 web 自身的查看器、Soma 镜像或 starter 锁定的 `engine`。在一次发布前后构建的镜像可能会运行两套不同的引擎。
- **所有生成产物均位于 `dist/`，该目录下的任何文件都不应提交到 git，且该目录之外没有任何生成文件**。`dist/` 的目录结构与发布的 `ants-artifacts.tar.gz` 完全一致，这允许游戏注册表中的 `path` 直接指向本地检出目录，或将 `release` 指向归档文件。下游通过路径直接读取文件（根目录下的组件、`cartridge.json`、`plugin.json`、`plugin.toml`、`maps/`、`reference/observations.json`、`viz/viz.js`、`viz/engine.json`）：包括 CLI（`registry.rs`、`serve.rs`、`cmd/mod.rs`）、Soma、Kalam 和 web 的 Dockerfile、`web/docs/Dockerfile` 以及 `web/docs/tutorials/build.sh`。重命名其中任何文件必须在同一批次中全文检索并更新所有关联项；它们只能通过发布包访问。
- **绝不可手工编辑生成文件**。`engine/plugin.toml` 是原始 ABI（`plugin.json` 为派生生成）；`mapgen/recipes/` 是原始设计，`maps/` 由其生成；`cartridge.json` 中的棋盘目录与 `limits.boards` 从棋盘数据派生；`engine/about.json` 是合并入清单的唯一手写输入。若 `maps/` 为空，`tools/package.py` 将直接报错拒绝。
- **基础棋盘是跨代码仓库的刚性契约**。它们的 ID 被硬编码在 ants-starter 的比赛文件、教程的棋盘页面以及基线测试中，重命名基础棋盘只能通过新发布同步到 starter。它们的跨度**即为** `limits.boards`：缩小跨度会导致 Soma 拒绝此前合规的赛季棋盘；扩大跨度则意味着参考测试集以及所有准入适配器必须覆盖新的极端尺寸。Kalam 最多支持 8 个座位，这也是规格包络的上限。
- **赛季棋盘绝不可提交至本仓库或包含在发布包中**。赛季棋盘由管理员在外部独立生成并上传到对应赛季中，仅在赛季关闭后备份到私有归档仓库中。
- **若样式规则未限定在 `.tb-viz` 范围内，`viz/check.mjs` 将导致构建失败**：查看器的单一 `<style>` 标签会被注入宿主页面文档中，未限定作用域的选择器会污染宿主全局样式。此外，**它还会拒绝由其他引擎生成的最后一帧固件**：固件中记录了其对弈运行的引擎摘要，若与 `dist/viz/engine.json` 不匹配会导致 `viz/build.sh` 失败，直到运行 `node make-last-frame.mjs` 重新生成。没有任何工具会自动为你重新生成它，历史上它曾因为没有及时比对而落后了两个摘要版本。Web 按名称逐个复制查看器的模块（`viz.js shell.js render.js engine.js map.js graph.js`），因此在 `viz.js` 中新增静态 import 必须同步更新 web 的复制列表（`map.js` 和 `graph.js` 正因如此而改为首次使用时动态加载）。
  **`react.js` 刻意不在该列表中**：它仅在宿主环境已有 React 时提供包装层，是全仓库唯一允许提及 React 的文件，且导入了裸标识符 `"react"`——因此若列表中任何模块导入了 `react.js`，将导致站点上的所有回放直接崩溃。`viz/README.md` 是查看器专属的指南（包括 `mount` API、层级与 React peer 依赖）。
- **`baselines/` 是公共引用路径**。ants-starter 通过 `git+https://github.com/Tiny-Brains/ants#subdirectory=baselines` 使用 pip 安装 `tb_baselines`，因此重命名该目录或其包名会破坏所有 starter 克隆。它保持在发布包之外：`build.sh` 从不读取它，`tools/pack.py` 仅打包 `dist/`，因此修改该目录不会变动引擎摘要。
