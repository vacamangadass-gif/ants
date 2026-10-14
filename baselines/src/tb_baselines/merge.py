"""将并行采集的多个数据集合并为一个，同时保持每个座位的行数据相互独立。

    python -m tb_baselines.merge data/teacher-[1-5].jsonl.gz --out data/teacher.jsonl.gz

当单个进程运行教师模型过慢时，可通过多个进程基于各自的随机种子并行采集，
而 `train/seq.py` 会按 `k = [episode, seat, turn]` 将座位的行重新按序读回。每个采集进程都从 0 开始对 episode 进行编号，
因此两个文件中的 episode 3 代表两场不同的比赛：合并工具通过将每个文件的 episode 偏移其位置的十万倍来区分它们，
并在文件头中记录各个分卷信息。若分卷文件头在引擎摘要、教师模型或记忆定义上存在分歧，则会直接拒绝：这些必须属于同一个数据集。
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

STRIDE = 100_000


def merge(parts: list[Path], out: Path) -> dict:
    headers = []
    for part in parts:
        with gzip.open(part, "rt") as f:
            headers.append(json.loads(f.readline()))
    header = dict(headers[0])
    for part, h in zip(parts[1:], headers[1:]):
        for key in ("engine_digest", "teacher", "memory", "evaluator"):
            if h.get(key) != header.get(key):
                raise SystemExit(f"{part}: {key} 为 {h.get(key)!r}，而首个分卷为 {header.get(key)!r}")
    header["parts"] = [{"file": p.name, "env_seed": h.get("env_seed"), "episode_offset": i * STRIDE}
                       for i, (p, h) in enumerate(zip(parts, headers))]
    header.pop("env_seed", None)
    rows = 0
    with gzip.open(out, "wt") as w:
        w.write(json.dumps(header) + "\n")
        for i, part in enumerate(parts):
            with gzip.open(part, "rt") as f:
                f.readline()
                try:
                    for line in f:
                        if not line.endswith("\n"):
                            break                       # 采集器异常中断导致的残缺末行
                        if i == 0:
                            w.write(line)
                        else:
                            row = json.loads(line)
                            row["k"][0] += i * STRIDE
                            w.write(json.dumps(row, separators=(",", ":")) + "\n")
                        rows += 1
                except EOFError:
                    pass
    return {"rows": rows, "parts": len(parts), "out": str(out)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("parts", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    print(json.dumps(merge(a.parts, a.out), indent=2))


if __name__ == "__main__":
    main()
