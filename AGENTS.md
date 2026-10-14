# AGENTS.md

本文件为在本仓库中工作的 AI Agent（如 Antigravity、Claude Code、Codex 等）提供全方位的上下文、开发规范、核心约束与操作指引。

---

## 1. 仓库定位与架构全景

`ants` 是 **TinyBrains** 平台的官方基准游戏卡带（game cartridge），实现了基于二维环面网格（wrapping torus grid）的确定性蚁群对抗游戏（源自 2011 年 Google AI Challenge）。

### 1.1 核心组件与生态关系

```
+---------------------------------------------------------------------------------+
|                                TinyBrains 平台生态                               |
|                                                                                 |
|   +-------------------+    +--------------------+    +----------------------+   |
|   |   Soma (评测/天梯)  |    |  Kalam (对局执行)  |    | web / docs (网站/手册)|  |
|   +---------+---------+    +---------+----------+    +----------+-----------+   |
|             ^                        ^                          ^               |
|             |                        |                          |               |
|             +------------------------+--------------------------+               |
|                                      | 读取 release 构件                        |
|                                      v                                          |
|                     +---------------------------------+                         |
|                     |     ants 仓库发布的 dist/ 构件   |                         |
|                     +---------------------------------+                         |
+--------------------------------------^------------------------------------------+
                                       | 构建打包 (build.sh)
+--------------------------------------+------------------------------------------+
|                              本仓库核心子系统                                    |
|                                                                                 |
|  [engine/]       Rust crate -> WebAssembly 组件 (tb-ants.wasm, ABI 1.0.0)       |
|                  负责游戏规则、回合结算、迷雾、观测与回放重模拟，无外部系统依赖     |
|                                                                                 |
|  [mapgen/]       Rust crate (仅宿主) -> 棋盘工厂与公平性验证工具                   |
|                  负责基础棋盘的配方渲染、生成、校验与全对称 sweep 扫描            |
|                                                                                 |
|  [viz/]          Node / 前端 -> 回放查看器                                       |
|                  使用 jco 转译的 Wasm 组件对局回放，供 Web 端、手册与 CLI 统一使用|
|                                                                                 |
|  [baselines/]    Python -> 选手模型训练管线与状态空间编码 (tb_baselines)          |
|                  提供神经网络观测编码 (planes.py) 与一致性测试                   |
+---------------------------------------------------------------------------------+
```

- **零外部依赖**：`engine/` 编译出的 Wasm 组件不依赖任何网络、数据库、文件系统、时钟或环境随机源。
- **发布契约**：CI 通过 `.github/workflows/build.yml` 构建并打包 `dist/` 为 `ants-artifacts.tar.gz`，作为全平台消费者（Soma、Kalam、Web、CLI 及参赛模板 ants-starter）唯一定向的输入源。

---

## 2. 导出接口（Export ABI）

引擎通过 `engine/src/lib.rs` 仅对外导出以下五个函数（接口规范见 `engine/plugin.toml`）：

```
tb.ants.worldgen(seeds[], map | maps, players?, max_turns?)  → wave_state
tb.ants.observe(wave_state, refs)      → 每个存活座位的观测视角，回显不透明 ref
tb.ants.step(wave_state, actions)      → { wave_state, done[], ended[], replay_delta }
tb.ants.finish(wave_state)             → 各场次排名、得分、结束原因及棋盘
tb.ants.replay-decode(payload, turn | from,to)  → 单帧数据，或指定区间的连续帧
```

> **注意**：
>
> - `replay-decode` 中的连字符 `-` 必须保留（Orion 平台要求）。
> - `wave_state` 对所有外部调用者完全不透明（打包并 base64 编码）。
> - `refs` 为扁平列表，原样匹配并回显，严禁解析。

---

## 3. Agent 开发与验证命令集

所有命令应在正确的目录下运行（`engine/` 与 `mapgen/` 为两个独立的 Cargo 工作区，根目录无 `Cargo.toml`）：

### 3.1 日常构建与全量检查

```sh
# 核心发布级构建与门禁（包含 deny.sh 确定性检查、engine 宿主测试、mapgen 测试并生成 dist/）
./build.sh

# 构建回放查看器（依赖 ./build.sh 的输出），并执行样式作用域与逻辑检查
viz/build.sh

# 重新生成回放最后一帧固件（在引擎摘要变更后必须运行）
(cd viz && node make-last-frame.mjs)
```

### 3.2 静态检查与单元测试

```sh
# Rust 代码风格与 Clippy 严格静态检查
(cd engine && cargo fmt --check && cargo clippy --locked --all-targets -- -D warnings)
(cd mapgen && cargo fmt --check && cargo clippy --locked --all-targets -- -D warnings)

# 仅运行 engine 宿主单元测试
(cd engine && cargo test)

# 运行指定的单个测试
(cd engine && cargo test a_replay_re_simulates_the_match_it_recorded)

# 运行规范中推导的场景测试
(cd engine && cargo test spec_scenario_)

# 运行 baselines 编码一致性测试
(cd baselines && pytest tests/ -q)
```

### 3.3 棋盘生成与公平性验证

```sh
cd mapgen
# 验证 maps/ 下所有已提交的基础棋盘是否与 recipes/ 配方逐字节一致
cargo run -- check

# 执行对弈冒烟测试
cargo run --release -- check --play 6

# 执行 8 座位多风格公平性全量扫描（验证对称性无 Bug）
cargo run --release -- sweep --seats 8 --styles cave,rooms --boards 10 --turns 1000 --json out.json --export dir
```

### 3.4 发布验证与等价性对比

```sh
# 打包 dist/ 为正式发布归档并打印摘要与建议 tag
python3 tools/pack.py <目标输出目录>

# 比对当前构建与旧发布版本的逐回合对战一致性
python3 tools/equivalence.py <旧发布解压路径> dist --allow ids
```

---

## 4. 核心规范与模块职责划分

`engine/` 目录下的各模块职责划分清晰，必须严格维持该分层：

| 文件 / 路径          | 专属职责范围                                                                                                           | 严禁包含内容                 |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| `src/lib.rs`       | 插件分发与 JSON 结构转换                                                                                               | 严禁包含游戏规则             |
| `src/grid.rs`      | 环面拓扑几何、曼哈顿/欧氏距离、对称性映射、方向枚举、半径计算、位图结构 (`Bits`) 与随机数 (`Rng`)                  | 不含具体对局状态             |
| `src/maps.rs`      | 棋盘数据文件解析、连通性校验、平移对称性校验、蚁穴轨道解析                                                             | 不含对局状态                 |
| `src/state.rs`     | 单场对弈运行时数据结构、蚂蚁 ID 分配与每个座位的`next_id`、步数截止计数                                              | 不直接处理回合步进规则       |
| `src/turn.rs`      | **固定顺序的回合结算**：移动 → 攻击 → 摧毁蚁穴 → 生成蚂蚁 → 收集食物 → 生成新食物；对局结束判定与最终排名   | 严禁调换结算阶段顺序         |
| `src/food.rs`      | 隐式速率下的食物分布生成：速率计算、对称分布集、打乱旋转、队列缓冲                                                     | 不含蚂蚁移动逻辑             |
| `src/observe.rs`   | 座位视角观测生成：战争迷雾（fog）、已知水域（known water）、可见性矩阵（`vis`）、蚂蚁编号（`ids`）、观察者相对阵营 | 严禁向视野注入`memory` 键  |
| `src/codec.rs`     | `wave_state` 的紧凑序列化/反序列化与 Base64 编解码                                                                   | 严禁在外部直接解析其内部结构 |
| `src/replay.rs`    | 动作流记录与解算为可视化动画帧                                                                                         | 不包含比赛评分规则           |
| `src/authoring.rs` | 仅宿主运行：生成供选手适配器准入校验的参考观测数据                                                                     | 保证不编译入 Wasm 组件       |

---

## 5. Agent 必须遵守的铁律（Invariants）

### 5.1 绝对禁止浮点运算（Integer-Only Arithmetic）

- 游戏逻辑全部位于 `engine/src/`（排除 `tests/` 与 `bin/`），**必须使用纯整数运算**。
- `tools/deny.sh` 会在构建时执行 AST/正则检测，一旦出现浮点操作直接终止构建。
- **原因**：浮点数在跨架构（x86 vs ARM vs Wasm）与跨环境（Native vs 浏览器 JIT）下无法保证二进制逐位一致，而 Ants 依赖纯动作流实现确定性重放。

### 5.2 严禁擅自变动引擎摘要（Engine Digest Sensitivity）

- **引擎摘要**是 `tb-ants.wasm` 的 SHA-256 散列值。
- 在 `engine/src/`（除 `tests/` 和 `bin/`）或 `engine/plugin.toml` 下的**任何编辑（包括修改注释和增减空行）**，都会改变 panic 行号并导致生成完全不同的引擎摘要。
- 引擎摘要的变动属于**全平台发布与天梯重置事件**。如需重构，必须用 `tools/equivalence.py` 对比新旧版本行为严格一致。
- 官方摘要在 `aarch64-unknown-linux-gnu` 上生成，本地电脑生成的摘要仅供开发测试。

### 5.3 严格保持回合结算次序（Turn Order is Invariant）

`engine/src/turn.rs` 中的回合阶段结算顺序是游戏契约的核心部分，**绝对不可调换**：

1. **移动（Move）**：同时执行所有移动意图，检测并处理碰撞与碰撞阻挡；
2. **攻击（Attack）**：根据聚焦战斗（Focus Battle）规则，计算各方优势并消灭劣势蚂蚁；
3. **摧毁（Raze）**：存活蚂蚁若位于敌方蚁穴上，摧毁该蚁穴；
4. **生成（Spawn）**：存活且未被摧毁的蚁穴根据储备生成新蚂蚁；
5. **收集（Gather）**：蚂蚁移入食物所在格子收集食物并增加储备；
6. **补给（Spawn Food）**：按照隐式速率和对称集合投放新食物。

### 5.4 棋盘是数据，绝不可手工修改

- 基础棋盘保存在 `maps/` 中（共 5 张：`basic-tiny-2p` 到 `basic-xlarge-8p`），由 `mapgen/recipes/*.toml` 配方生成。
- **严禁直接编辑 `maps/*.json`**。所有棋盘必须由 `mapgen` 工具渲染并由 `cargo run -- check` 逐字节校验。
- 基础棋盘的尺寸与座位跨度定义了平台的包络（`limits.boards`：2-8 座位，边长 24-124）。

### 5.5 单一引擎原则（Single Engine Principle）

- 前端查看器（`viz/`）必须且只能通过 Wasm 组件转译产物（`jco`）对对局进行重模拟。
- 严禁用 JavaScript 或 Python 重新实现游戏规则，严禁尝试解析 `wave_state`。

### 5.6 规则分歧以 2011 官方引擎为准

- 若公开文档描述与 2011 年 Google AI Challenge 官方 Python 引擎（`aichallenge/ants/ants.py`）发生冲突，**以官方 Python 引擎行为为准**。
- 代码注释需保留 `ants.py:NNN` 引用；新增规则改动需补充测试用例。

### 5.7 观测格式改动必须全局联动

- 观测视野（`observe`）格式的变动属于跨仓库契约变更。
- 若修改视野字段，必须在同一次变更中同步修改：
  1. `web/docs` 中的参赛者规范；
  2. `baselines/src/tb_baselines/planes.py` 神经网络特征面编码；
  3. `dist/reference/observations.json` 参考观测数据。

---

## 6. 关键避坑指南（Gotchas）

1. **查看器 CSS 隔离**：
   `viz/` 的样式会被注入宿主宿主页面中，`viz/check.mjs` 会强制检查：**所有 CSS 规则必须带 `.tb-viz` 前缀**，否则构建报错。
2. **`react.js` 依赖隔离**：
   `viz/react.js` 仅供有 React 环境的宿主使用，`viz.js` 及其他核心模块严禁 `import` 该文件，否则会导致站点回放全面崩溃。
3. **`dist/` 目录规范**：
   `dist/` 属于纯构建产物输出，已被 `.gitignore` 忽略。严禁向 git 提交 `dist/` 中的任何文件，也严禁在 `dist/` 之外生成临时构建文件。
4. **Git 提交风格**：
   提交信息使用完整的陈述句（平台语气，如 `"Enforce strict torus wrapping distance calculation"`），不使用 conventional-commit（如 `feat:` / `fix:`）前缀。
5. **注释与解释性**：
   编写代码注释时注重解释**为什么**（说明被否决的替代方案、所防范的潜在 Bug、对应的竞赛规则行号），而不是简单复述代码在做什么。
