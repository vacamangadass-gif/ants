#!/bin/sh
# 确定性法，机械检查：游戏逻辑中禁止使用浮点数。
#
# 它是使平台构建与浏览器构建逐位一致的原因，也正是这使得回放成为动作流，
# 而不是多出好几倍的帧。Rust 自身不强制执行这一规则，且循环边界中的任何 `as f64`
# 都可能在两次构建结果不一致时才显现。
#
# 仅检查 engine/src/，不包括其 tests/ 或 bin/：这些是宿主工具，永远不会进入组件。
set -eu
src=$(cd "$(dirname "$0")/../engine/src" && pwd)
bad=$(grep -rnE '\bf32\b|\bf64\b|\.sqrt\(|\.powi\(|\.powf\(|[0-9]\.[0-9]' \
        "$src" --include='*.rs' --exclude-dir=tests --exclude-dir=bin | grep -v '://' || true)
if [ -n "$bad" ]; then
  echo "floating point in game logic -- the determinism law forbids it:" >&2
  echo "$bad" >&2
  exit 1
fi
echo "determinism: no floating point in game logic"
