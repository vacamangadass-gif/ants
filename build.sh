#!/bin/sh
# 质量关卡门禁，之后将所有构件输出到 dist/。
#
# 依赖环境：rustup（rust-toolchain.toml 指定了精确的编译器版本与 wasm32 目标架构）、
# WASM_TOOLS_VERSION 指定版本的 `wasm-tools`，以及 Python 3.11 或更高版本。
# 回放查看器由 viz/build.sh 单独构建并输出至 dist/viz/。
#
# dist/ 即为交付构件集，而本脚本就是用于正式发布的构建脚本。一次完整的发布由
# .github/workflows/build.yml 运行本脚本与 viz/build.sh，并由 tools/pack.py
# 将 dist/ 打包为一个归档文件。同一个构件在任何地方都不会有第二种构建方式。
#
# 引擎摘要（ENGINE DIGEST）取决于四要素：源码、精确的 rustc、精确的 wasm-tools，
# 以及运行 rustc 的宿主环境（HOST）。前三项已在此处与 rust-toolchain.toml 中固定锁定，
# 构建路径也在下方进行了重新映射。宿主环境无法被锁定：在 Apple-silicon Mac 与 arm64 Linux 上，
# 即使使用相同的源码、构建参数与嵌入路径，函数排布顺序依然不同（Cargo 会将包含 host triple 的
# rustc 版本字符串混入每个 crate 的符号哈希中）。因此正式发布构建是在 aarch64-unknown-linux-gnu
# 上完成的，且只有在该宿主上构建才能得到官方发布的摘要；在 Mac 上的构建是相同逻辑但字节码不同。
#
# 宿主测试是核心门禁：运行时没有任何机制会捕获错误实现的游戏规则。
# 确定性检查首先运行，因为它是源码的静态属性，而非动态测试。
set -eu
here=$(cd "$(dirname "$0")" && pwd)
dist="$here/dist"
target="${CARGO_TARGET_DIR:-$here/engine/target}"

# `component new` 会在组件中记录自身版本，因此使用不同版本的 wasm-tools 会产生不同的摘要。
WASM_TOOLS_VERSION=1.258.0

have=$(wasm-tools --version | cut -d' ' -f2)
if [ "$have" != "$WASM_TOOLS_VERSION" ]; then
  echo "warning: wasm-tools $have, not $WASM_TOOLS_VERSION -- this component's digest will not be a release's" >&2
fi

"$here/tools/deny.sh"
cd "$here/engine"
cargo test --locked

# 基础棋盘：maps/ 下的每个棋盘完全由 mapgen/recipes/ 下的配方逐字节生成，
# 且符合每个棋盘都必须遵守的规则。它是一个独立的 crate，因此调整生成器或重新生成棋盘
# 都不会改变引擎摘要——Wasm 组件本身不包含任何棋盘。
# 赛季棋盘使用相同的工具在本仓库之外生成。
(cd "$here/mapgen" && cargo test --locked)

# 全新构建：从 maps/ 中删除的棋盘绝不能残留在 dist/maps/ 中，
# 且从上一版本组件转译的查看器对应的是其他引擎版本。
rm -rf "$dist"
mkdir -p "$dist/reference"

# 摘要绝不能依赖于代码检出的绝对路径。rustc 会将 panic 可能引用的每个源码文件的绝对路径
# 烧录到二进制中：crates.io 源码，以及（安装了 rust-src 组件时）标准库源码（否则 rustc 会命名为 /rustc/<commit>）。
# 将这三者全部重映射为固定路径，可确保组件仅由源码和宿主决定，且这些名称也是平台发布一直采用的名称。
# 该环境变量是直接设置而非追加：避免用户 shell 中导出的额外参数影响构建一致性。
sysroot=$(rustc --print sysroot)
commit=$(rustc -vV | sed -n 's/^commit-hash: //p')
RUSTFLAGS="--remap-path-prefix=${CARGO_HOME:-$HOME/.cargo}/registry/src=/cargo --remap-path-prefix=$sysroot/lib/rustlib/src/rust=/rustc/$commit --remap-path-prefix=$here=/ants" \
  cargo build --locked --release --target wasm32-unknown-unknown --lib
wasm-tools component new "$target/wasm32-unknown-unknown/release/tb_ants.wasm" -o "$dist/tb-ants.wasm"
wasm-tools validate "$dist/tb-ants.wasm" --features component-model

# 清单文件由程序自动生成，因为无人手工编辑的构件绝不会与生成它的代码产生偏差。
# 准入机制用于验证适配器的参考观测（reference observations）同样是引擎直接输出：
# 人工维护的副本可能会测试游戏不再生成的格式。
# 它们基于基础棋盘绘制，这使得这些棋盘成为选手上传内容必须契合的规格包络（envelope）。
cargo run --locked --quiet --bin manifest > "$dist/cartridge.json"
cargo run --locked --quiet --bin reference > "$dist/reference/observations.json"

python3 "$here/tools/package.py"
echo "    built on        $(rustc -vV | sed -n 's/^host: //p') (a release builds on aarch64-unknown-linux-gnu)"
