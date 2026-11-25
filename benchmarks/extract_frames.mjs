// benchmarks/extract_frames.mjs
// 通过官方 Wasm 引擎回放解码器提取多席位、多蚁穴对局的逐帧与行为遥测指标。

import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const DIST_DIR = process.env.ANTS_DIST_DIR
  ? resolve(process.env.ANTS_DIST_DIR)
  : resolve(HERE, '../dist');
const ENGINE_JS = join(DIST_DIR, 'viz', 'engine.js');
if (!existsSync(ENGINE_JS)) {
  throw new Error(`未找到官方 Wasm 回放解码器: ${ENGINE_JS} (请先执行 ./build.sh && viz/build.sh 或设置 ANTS_DIST_DIR)`);
}
const { allFrames } = await import(pathToFileURL(ENGINE_JS).href);

const ANT_CHECKPOINTS = [25, 50, 100, 150, 200, 300, 500, 750, 1000];

export function analyzeReplay(content) {
  const replay = typeof content === 'string' ? JSON.parse(content) : content;
  const frames = allFrames(replay);
  const rawSeats = (replay.seats || []).slice().sort((a, b) => a.seat - b.seat);
  const seats = rawSeats.map((s, idx) => ({
    seat: s.seat ?? idx,
    label: s.label || `seat_${s.seat ?? idx}`,
    infer_us_total: s.infer_us_total ?? 0,
    infer_us_max: s.infer_us_max ?? 0,
    strikes: Number(s.strikes ?? 0),
    forfeit: Boolean(s.forfeit),
  }));
  const ranks = replay.engine_ranks || [];
  const scores = replay.scores || [];
  const reason = replay.reason;
  const turns = Number(replay.turns ?? 0);
  const numSeats = seats.length;

  const collisions = Array(numSeats).fill(0);
  const combatDeaths = Array(numSeats).fill(0);
  const combatKills = Array(numSeats).fill(0);
  const razedHills = [];
  const antIds = Array.from({ length: numSeats }, () => new Set());
  const initialAnts = Array(numSeats).fill(0);
  const initialHills = Array(numSeats).fill(0);
  const peakAnts = Array(numSeats).fill(0);
  const finalAnts = Array(numSeats).fill(0);
  const antsAtTurns = Array.from({ length: numSeats }, () => ({}));

  // 按蚂蚁 id 追踪最近 3 帧位置，计算 2-步往返震荡与 3-步原地停滞
  const posHist = new Map();
  const osc2Steps = Array(numSeats).fill(0);
  const osc2Eligible = Array(numSeats).fill(0);
  const idle3Steps = Array(numSeats).fill(0);
  const idle3Eligible = Array(numSeats).fill(0);

  // 第 0 帧统计各席位初始蚂蚁数与初始蚁穴数（兼容 1h/2h/3h 与任意席位）
  if (frames.length > 0) {
    const f0 = frames[0];
    for (const a of f0.ants || []) {
      if (a[2] >= 0 && a[2] < numSeats) {
        initialAnts[a[2]]++;
      }
    }
    for (const h of f0.hills || []) {
      if (h[2] >= 0 && h[2] < numSeats) {
        initialHills[h[2]]++;
      }
    }
  }

  for (let t = 0; t < frames.length; t++) {
    const f = frames[t];
    const counts = Array(numSeats).fill(0);
    for (const a of f.ants || []) {
      const [r, c, owner, id] = a;
      if (owner >= 0 && owner < numSeats) {
        antIds[owner].add(id);
        counts[owner]++;
        let hist = posHist.get(id);
        if (!hist) {
          hist = [];
          posHist.set(id, hist);
        }
        hist.push([r, c]);
        if (hist.length > 3) hist.shift();
        if (hist.length === 3) {
          osc2Eligible[owner]++;
          idle3Eligible[owner]++;
          const same02 = hist[0][0] === hist[2][0] && hist[0][1] === hist[2][1];
          const same01 = hist[0][0] === hist[1][0] && hist[0][1] === hist[1][1];
          if (same02 && !same01) osc2Steps[owner]++;
          if (same02 && same01) idle3Steps[owner]++;
        }
      }
    }
    for (let s = 0; s < numSeats; s++) {
      if (counts[s] > peakAnts[s]) peakAnts[s] = counts[s];
      if (ANT_CHECKPOINTS.includes(f.turn)) {
        antsAtTurns[s][f.turn] = counts[s];
      }
    }
    if (t === frames.length - 1) {
      for (let s = 0; s < numSeats; s++) {
        finalAnts[s] = counts[s];
      }
    }

    for (const d of f.deaths || []) {
      const owner = d.ant[2];
      if (owner >= 0 && owner < numSeats) {
        if (!d.by || d.by.length === 0) {
          collisions[owner]++;
        } else {
          combatDeaths[owner]++;
          const killers = new Set(
            d.by.map(k => k[2]).filter(k => k >= 0 && k < numSeats && k !== owner)
          );
          if (killers.size > 0) {
            const share = 1.0 / killers.size;
            for (const k of killers) {
              combatKills[k] += share;
            }
          }
        }
      }
    }

    for (const r of f.razed || []) {
      // r 格式为 [row, col, owner, by]
      razedHills.push({ turn: f.turn, row: r[0], col: r[1], owner: r[2], razer: r[3] });
    }
  }

  // 构建回合号到动作增量 delta 的映射，统计移动受阻、HOLD 与己方蚁穴驻留阻塞
  const deltaByTurn = new Map();
  for (let i = 0; i < (replay.deltas || []).length; i++) {
    const d = replay.deltas[i];
    if (d && d.a) {
      deltaByTurn.set(d.t ?? i, d);
    }
  }

  const totalOrders = Array(numSeats).fill(0);
  const blockedOrders = Array(numSeats).fill(0);
  const allOrders = Array(numSeats).fill(0);
  const holdOrders = Array(numSeats).fill(0);
  const onOwnHillSteps = Array(numSeats).fill(0);
  const onOwnHillHoldOrders = Array(numSeats).fill(0);
  const ownHillOccupiedTurns = Array(numSeats).fill(0);

  for (let t = 0; t < frames.length - 1; t++) {
    const fNow = frames[t];
    const fNext = frames[t + 1];
    const delta = deltaByTurn.get(fNow.turn) || (replay.deltas ? replay.deltas[t] : null);
    if (!delta || !delta.a) continue;

    const nextPos = new Map();
    for (const a of fNext.ants || []) {
      nextPos.set(a[3], [a[0], a[1]]);
    }

    const cols = fNow.size ? fNow.size[1] : (replay.map ? replay.map.cols : 0);
    const ownHillsNow = Array.from({ length: numSeats }, () => new Set());
    for (const h of fNow.hills || []) {
      if (h[2] >= 0 && h[2] < numSeats) {
        ownHillsNow[h[2]].add(h[0] * cols + h[1]);
      }
    }

    for (let s = 0; s < numSeats; s++) {
      const orders = delta.a[s] || '';
      const myAnts = (fNow.ants || [])
        .filter(a => a[2] === s)
        .sort((a, b) => (a[0] * cols + a[1]) - (b[0] * cols + b[1]) || (a[3] - b[3]));
      let occupiedThisTurn = false;
      for (let i = 0; i < myAnts.length; i++) {
        const a = myAnts[i];
        const ord = orders[i] || '-';
        const cellKey = a[0] * cols + a[1];
        const onOwnHill = ownHillsNow[s].has(cellKey);
        allOrders[s]++;
        if (onOwnHill) {
          onOwnHillSteps[s]++;
          occupiedThisTurn = true;
        }
        if (ord === '-') {
          holdOrders[s]++;
          if (onOwnHill) {
            onOwnHillHoldOrders[s]++;
          }
        } else if (ord === 'N' || ord === 'E' || ord === 'S' || ord === 'W') {
          totalOrders[s]++;
          const np = nextPos.get(a[3]);
          if (np && np[0] === a[0] && np[1] === a[1]) {
            blockedOrders[s]++;
          }
        }
      }
      if (occupiedThisTurn) {
        ownHillOccupiedTurns[s]++;
      }
    }
  }

  // 多席位通用的名次判定与归一化排名得分（2p 下严格退化为 1.0 / 0.5 / 0.0）
  const bestRank = ranks.length > 0 ? Math.min(...ranks) : 1;
  const winnersCount = ranks.filter(r => r === bestRank).length;

  const perSeatMetrics = [];
  for (let s = 0; s < numSeats; s++) {
    const myRank = ranks[s];
    const betterCount = ranks.filter(r => r > myRank).length;
    const tiedCount = Math.max(0, ranks.filter(r => r === myRank).length - 1);
    const rankPct = numSeats > 1 ? (betterCount + 0.5 * tiedCount) / (numSeats - 1) : 1.0;

    const won = myRank === bestRank && winnersCount === 1;
    const drawn = myRank === bestRank && winnersCount > 1;
    const lost = myRank > bestRank;
    const pts = rankPct;

    const razingsByMe = razedHills.filter(rz => rz.razer === s && rz.owner !== s);
    const firstRazeTurn = razingsByMe.length > 0 ? Math.min(...razingsByMe.map(rz => rz.turn)) : null;
    const lostByMe = razedHills.filter(rz => rz.owner === s);
    const firstLostTurn = lostByMe.length > 0 ? Math.min(...lostByMe.map(rz => rz.turn)) : null;

    const antsSpawned = Math.max(0, antIds[s].size - initialAnts[s]);
    const kdRatio = combatDeaths[s] > 0
      ? combatKills[s] / combatDeaths[s]
      : (combatKills[s] > 0 ? combatKills[s] : 0.0);

    perSeatMetrics.push({
      seat: s,
      name: seats[s].label,
      rank: myRank,
      rank_pct: rankPct,
      score: scores[s],
      won,
      drawn,
      lost,
      points: pts,
      initial_hills: initialHills[s],
      initial_ants: initialAnts[s],
      hills_razed: razingsByMe.length,
      hills_lost: lostByMe.length,
      first_hill_destroy_turn: firstRazeTurn,
      first_hill_lost_turn: firstLostTurn,
      collisions: collisions[s],
      combat_deaths: combatDeaths[s],
      combat_kills: Number(combatKills[s].toFixed(4)),
      kd_ratio: Number(kdRatio.toFixed(4)),
      ants_spawned: antsSpawned,
      peak_ants: peakAnts[s],
      final_ants: finalAnts[s],
      ants_at_turns: antsAtTurns[s],
      total_orders: totalOrders[s],
      blocked_orders: blockedOrders[s],
      blocked_rate: totalOrders[s] > 0 ? blockedOrders[s] / totalOrders[s] : 0.0,
      all_orders: allOrders[s],
      hold_orders: holdOrders[s],
      hold_rate: allOrders[s] > 0 ? holdOrders[s] / allOrders[s] : 0.0,
      on_own_hill_steps: onOwnHillSteps[s],
      on_own_hill_hold_orders: onOwnHillHoldOrders[s],
      on_own_hill_hold_rate: onOwnHillSteps[s] > 0 ? onOwnHillHoldOrders[s] / onOwnHillSteps[s] : 0.0,
      own_hill_occupied_turns: ownHillOccupiedTurns[s],
      idle3_steps: idle3Steps[s],
      idle3_eligible: idle3Eligible[s],
      idle3_rate: idle3Eligible[s] > 0 ? idle3Steps[s] / idle3Eligible[s] : 0.0,
      osc2_steps: osc2Steps[s],
      osc2_eligible: osc2Eligible[s],
      osc2_rate: osc2Eligible[s] > 0 ? osc2Steps[s] / osc2Eligible[s] : 0.0,
      infer_us_total: seats[s].infer_us_total,
      infer_us_max: seats[s].infer_us_max,
      strikes: seats[s].strikes,
      forfeit: seats[s].forfeit,
    });
  }

  return {
    match_id: replay.match_id || '',
    map_id: replay.map_id || (replay.map && replay.map.id) || '',
    rows: replay.map ? replay.map.rows : (frames[0]?.size ? frames[0].size[0] : null),
    cols: replay.map ? replay.map.cols : (frames[0]?.size ? frames[0].size[1] : null),
    players: numSeats,
    hills_per_seat: numSeats > 0 && initialHills.length > 0 ? initialHills[0] : 0,
    seed: replay.seed,
    turns,
    max_turns: replay.max_turns ?? 1000,
    engine_digest: replay.engine_digest || '',
    reason,
    ranks,
    scores,
    razed_hills: razedHills,
    seats: perSeatMetrics,
  };
}

function main() {
  const args = process.argv.slice(2);
  if (args.length === 0) {
    console.error('Usage: node extract_frames.mjs <replay_file_or_dir> [more_files...]');
    process.exit(1);
  }

  const files = [];
  for (const arg of args) {
    const p = resolve(arg);
    const st = statSync(p);
    if (st.isDirectory()) {
      for (const fn of readdirSync(p)) {
        if (fn.endsWith('.json') && !fn.startsWith('_batch_')) {
          files.push(join(p, fn));
        }
      }
    } else if (p.endsWith('.json') && !p.includes('_batch_')) {
      files.push(p);
    }
  }

  files.sort();

  const results = [];
  for (const f of files) {
    const content = readFileSync(f, 'utf8');
    const data = analyzeReplay(content);
    data.file = f;
    results.push(data);
  }

  console.log(JSON.stringify(results));
}

if (process.argv[1] && resolve(process.argv[1]) === resolve(new URL(import.meta.url).pathname)) {
  main();
}
