import { readFileSync, readdirSync, statSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { allFrames } from '../dist/viz/engine.js';

function torusDistance(r1, c1, r2, c2, rows, cols) {
  const dr = Math.abs(r1 - r2);
  const dc = Math.abs(c1 - c2);
  const minDr = Math.min(dr, rows - dr);
  const minDc = Math.min(dc, cols - dc);
  return minDr + minDc;
}

export function analyzeBehaviorReplay(content, targetName = 'exp083') {
  const replay = typeof content === 'string' ? JSON.parse(content) : content;
  const frames = allFrames(replay);
  const numSeats = (replay.seats || []).length;
  const rows = replay.map ? replay.map.rows : (frames[0]?.size ? frames[0].size[0] : 0);
  const cols = replay.map ? replay.map.cols : (frames[0]?.size ? frames[0].size[1] : 0);

  let targetSeat = null;
  let opponentSeat = null;
  let targetLabel = '';
  let opponentLabel = '';

  for (let s = 0; s < numSeats; s++) {
    const lbl = replay.seats[s]?.label || `seat_${s}`;
    if (lbl === targetName) {
      targetSeat = s;
      targetLabel = lbl;
    } else {
      opponentSeat = s;
      opponentLabel = lbl;
    }
  }

  // 如果对局是自博弈（双方都是 targetName），则统计双座
  const seatsToAnalyze = [];
  if (targetSeat !== null) {
    seatsToAnalyze.push(targetSeat);
  }
  if (opponentSeat !== null && opponentLabel === targetName) {
    seatsToAnalyze.push(opponentSeat);
  }

  // 如果指定了目标名称但两座均非该目标，则跳过该对局
  if (seatsToAnalyze.length === 0) {
    return null;
  }

  const enemyHills = [];
  if (frames[0] && frames[0].hills) {
    for (const h of frames[0].hills) {
      const owner = h.length >= 3 ? h[2] : 0;
      enemyHills.push({ row: h[0], col: h[1], owner });
    }
  } else if (replay.map && replay.map.hills) {
    for (let idx = 0; idx < replay.map.hills.length; idx++) {
      const h = replay.map.hills[idx];
      const owner = h.length >= 3 ? h[2] : idx;
      enemyHills.push({ row: h[0], col: h[1], owner });
    }
  }

  // 1. 轨迹记录：ant_id -> array of { turn, r, c }
  const antHistories = new Map();
  // ant_id -> consecutive blocked count
  const antConsecutiveBlocked = new Map();
  // 记录每个 ant 的最大连续受阻步数
  const antMaxBlockedStreak = new Map();
  // 记录震荡事件：连续 3 次及以上往返 (>= 6 steps)
  const antOscillations = [];
  // 记录受阻移动
  let totalTargetOrders = 0;
  let blockedTargetOrders = 0;
  let consecutiveBlockedEpisodes = 0; // 连续受阻 >= 3 步事件数
  let mutualHeadOnBlocks = 0; // 蚂蚁对向互撞受阻事件数

  // 食物搜集与敌穴距离追踪
  let initialEnemyHillDist = null;
  let minEnemyHillDist = 999999;
  let enemyHillReached = false;
  let enemyHillRazed = false;
  let razeTurn = null;
  let foodCollectedByTarget = 0;

  for (const rz of (replay.razed || [])) {
    if (seatsToAnalyze.includes(rz[3])) {
      enemyHillRazed = true;
      razeTurn = rz[0];
    }
  }

  for (let t = 0; t < frames.length; t++) {
    const f = frames[t];
    const targetAntsThisTurn = [];

    for (const a of f.ants) {
      const r = a[0];
      const c = a[1];
      const s = a[2];
      const id = a[3];

      if (seatsToAnalyze.includes(s)) {
        targetAntsThisTurn.push({ r, c, id, s });
        if (!antHistories.has(id)) {
          antHistories.set(id, []);
          antConsecutiveBlocked.set(id, 0);
          antMaxBlockedStreak.set(id, 0);
        }
        antHistories.get(id).push({ turn: t, r, c });

        // 统计距离敌方蚁穴的曼哈顿环面距离
        for (const eh of enemyHills) {
          if (!seatsToAnalyze.includes(eh.owner)) {
            const dist = torusDistance(r, c, eh.row, eh.col, rows, cols);
            if (initialEnemyHillDist === null && t === 0) {
              initialEnemyHillDist = dist;
            }
            if (dist < minEnemyHillDist) {
              minEnemyHillDist = dist;
            }
            if (dist === 0) {
              enemyHillReached = true;
            }
          }
        }
      }
    }

    // 检查是否有毁灭敌穴
    for (const rz of f.razed) {
      if (seatsToAnalyze.includes(rz[3])) {
        enemyHillRazed = true;
        if (razeTurn === null) {
          razeTurn = f.turn;
        }
      }
    }

    // 检查动作与阻塞
    if (t < frames.length - 1 && replay.deltas && replay.deltas[t]) {
      const delta = replay.deltas[t];
      const fNext = frames[t + 1];
      const nextPosMap = new Map();
      for (const a of fNext.ants) {
        nextPosMap.set(a[3], [a[0], a[1]]);
      }

      for (const s of seatsToAnalyze) {
        const orders = delta.a ? (delta.a[s] || '') : '';
        const myAnts = f.ants
          .filter(a => a[2] === s)
          .sort((a, b) => (a[0] * cols + a[1]) - (b[0] * cols + b[1]) || (a[3] - b[3]));

        const attemptedMoves = []; // { id, from: [r,c], to: [tr, tc] }

        for (let i = 0; i < myAnts.length; i++) {
          const ord = orders[i] || '-';
          const a = myAnts[i];
          const id = a[3];

          if (ord === 'N' || ord === 'E' || ord === 'S' || ord === 'W') {
            totalTargetOrders++;

            let tr = a[0];
            let tc = a[1];
            if (ord === 'N') tr = (tr - 1 + rows) % rows;
            else if (ord === 'S') tr = (tr + 1) % rows;
            else if (ord === 'E') tc = (tc + 1) % cols;
            else if (ord === 'W') tc = (tc - 1 + cols) % cols;

            attemptedMoves.push({ id, from: [a[0], a[1]], to: [tr, tc] });

            const np = nextPosMap.get(id);
            const isBlocked = (np && np[0] === a[0] && np[1] === a[1]);

            if (isBlocked) {
              blockedTargetOrders++;
              const curBlocked = (antConsecutiveBlocked.get(id) || 0) + 1;
              antConsecutiveBlocked.set(id, curBlocked);
              const maxB = antMaxBlockedStreak.get(id) || 0;
              if (curBlocked > maxB) {
                antMaxBlockedStreak.set(id, curBlocked);
              }
              if (curBlocked === 3) {
                consecutiveBlockedEpisodes++;
              }
            } else {
              antConsecutiveBlocked.set(id, 0);
            }
          } else {
            antConsecutiveBlocked.set(id, 0);
          }
        }

        // 检测两只友方蚂蚁互相向对方格子移动引发的死锁/互撞
        for (let j = 0; j < attemptedMoves.length; j++) {
          for (let k = j + 1; k < attemptedMoves.length; k++) {
            const m1 = attemptedMoves[j];
            const m2 = attemptedMoves[k];
            // 互换位置意图：m1.from == m2.to && m1.to == m2.from
            if (m1.from[0] === m2.to[0] && m1.from[1] === m2.to[1] &&
                m1.to[0] === m2.from[0] && m1.to[1] === m2.from[1]) {
              mutualHeadOnBlocks++;
            }
          }
        }
      }
    }
  }

  // 2. 震荡检测：针对每个蚂蚁的移动历史检测 length-2 往返振荡 (A -> B -> A -> B...)
  let totalOscillationEpisodes = 0;
  let maxOscillationStreak = 0;
  const oscillatingAnts = new Set();

  for (const [id, hist] of antHistories.entries()) {
    if (hist.length < 6) continue;

    let currentOscillationStreak = 0; // 半周期数
    for (let i = 2; i < hist.length; i++) {
      const pNow = hist[i];
      const pPrev = hist[i - 1];
      const pPrev2 = hist[i - 2];

      // 严格检查：当前坐标等于前两步坐标，且不等于前一步坐标
      if (pNow.r === pPrev2.r && pNow.c === pPrev2.c &&
          (pNow.r !== pPrev.r || pNow.c !== pPrev.c)) {
        if (currentOscillationStreak === 0) {
          currentOscillationStreak = 3; // 经历 A -> B -> A
        } else {
          currentOscillationStreak++;
        }

        if (currentOscillationStreak > maxOscillationStreak) {
          maxOscillationStreak = currentOscillationStreak;
        }

        // 连续 6 步往返即为 3 个完整振荡周期
        if (currentOscillationStreak === 6) {
          totalOscillationEpisodes++;
          oscillatingAnts.add(id);
          antOscillations.push({
            ant_id: id,
            start_turn: hist[i - 5].turn,
            end_turn: pNow.turn,
            pos_a: [pPrev.r, pPrev.c],
            pos_b: [pNow.r, pNow.c],
          });
        }
      } else {
        currentOscillationStreak = 0;
      }
    }
  }

  // 统计采食数量
  for (const s of seatsToAnalyze) {
    foodCollectedByTarget += (replay.scores ? replay.scores[s] : 0);
  }

  const blockedRate = totalTargetOrders > 0 ? (blockedTargetOrders / totalTargetOrders) : 0;
  const turns = replay.turns || frames.length - 1;

  return {
    match_id: replay.match_id || '',
    map_id: replay.map_id || (replay.map?.id) || '',
    seed: replay.seed,
    turns,
    target_seats: seatsToAnalyze,
    total_orders: totalTargetOrders,
    blocked_orders: blockedTargetOrders,
    blocked_rate: blockedRate,
    blocked_rate_pct: blockedRate * 100.0,
    consecutive_blocked_episodes: consecutiveBlockedEpisodes,
    mutual_head_on_blocks: mutualHeadOnBlocks,
    total_oscillation_episodes: totalOscillationEpisodes,
    max_oscillation_streak: maxOscillationStreak,
    oscillating_ants_count: oscillatingAnts.size,
    total_ants_tracked: antHistories.size,
    initial_enemy_hill_dist: initialEnemyHillDist,
    min_enemy_hill_dist: minEnemyHillDist,
    enemy_hill_reached: enemyHillReached,
    enemy_hill_razed: enemyHillRazed,
    raze_turn: razeTurn,
    food_collected: foodCollectedByTarget,
    food_per_100_turns: turns > 0 ? (foodCollectedByTarget / turns * 100.0) : 0,
    oscillations: antOscillations,
  };
}

function main() {
  const args = process.argv.slice(2);
  if (args.length === 0) {
    console.error('Usage: node tools/analyze_behavior.mjs <target_name> <path_or_dir> [...]');
    process.exit(1);
  }

  const targetName = args[0];
  const inputPaths = args.slice(1);
  const files = [];

  for (const inp of inputPaths) {
    const p = resolve(inp);
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
    try {
      const content = readFileSync(f, 'utf8');
      const res = analyzeBehaviorReplay(content, targetName);
      if (res !== null) {
        res.file = f;
        results.push(res);
      }
    } catch (e) {
      console.error(`处理 ${f} 失败:`, e.message);
    }
  }

  console.log(JSON.stringify(results));
}

if (process.argv[1] && resolve(process.argv[1]) === resolve(new URL(import.meta.url).pathname)) {
  main();
}
