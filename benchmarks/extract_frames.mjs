// benchmarks/extract_frames.mjs
// Extracts R4 match evaluation metrics from TinyBrains Ants tournament replays using the official Wasm engine.

import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { allFrames } from '../dist/viz/engine.js';

export function analyzeReplay(content) {
  const replay = typeof content === 'string' ? JSON.parse(content) : content;
  const frames = allFrames(replay);
  const seats = (replay.seats || []).map(s => ({
    seat: s.seat,
    label: s.label || `seat_${s.seat}`,
    infer_us_total: s.infer_us_total ?? 0,
    infer_us_max: s.infer_us_max ?? 0
  }));
  const ranks = replay.engine_ranks;
  const scores = replay.scores;
  const reason = replay.reason;
  const turns = Number(replay.turns ?? 0);
  const numSeats = seats.length;

  const collisions = Array(numSeats).fill(0);
  const combatDeaths = Array(numSeats).fill(0);
  const razedHills = [];
  const antIds = Array.from({ length: numSeats }, () => new Set());

  for (let t = 0; t < frames.length; t++) {
    const f = frames[t];
    for (const a of f.ants) {
      if (a[2] < numSeats) {
        antIds[a[2]].add(a[3]);
      }
    }
    for (const d of f.deaths) {
      const owner = d.ant[2];
      if (owner < numSeats) {
        if (d.by.length === 0) {
          collisions[owner]++;
        } else {
          combatDeaths[owner]++;
        }
      }
    }
    for (const r of f.razed) {
      // r is [row, col, owner, by]
      razedHills.push({ turn: f.turn, row: r[0], col: r[1], owner: r[2], razer: r[3] });
    }
  }

  // Count blocked move orders (order into water or food where position stays identical)
  const totalOrders = Array(numSeats).fill(0);
  const blockedOrders = Array(numSeats).fill(0);

  for (let t = 0; t < frames.length - 1; t++) {
    const fNow = frames[t];
    const fNext = frames[t + 1];
    const delta = replay.deltas ? replay.deltas[t] : null;
    if (!delta || !delta.a) continue;

    const nextPos = new Map();
    for (const a of fNext.ants) {
      nextPos.set(a[3], [a[0], a[1]]);
    }

    for (let s = 0; s < numSeats; s++) {
      const orders = delta.a[s] || '';
      const myAnts = fNow.ants.filter(a => a[2] === s);
      for (let i = 0; i < myAnts.length; i++) {
        const ord = orders[i] || '-';
        if (ord === 'N' || ord === 'E' || ord === 'S' || ord === 'W') {
          totalOrders[s]++;
          const a = myAnts[i];
          const np = nextPos.get(a[3]);
          if (np && np[0] === a[0] && np[1] === a[1]) {
            blockedOrders[s]++;
          }
        }
      }
    }
  }

  const perSeatMetrics = [];
  for (let s = 0; s < numSeats; s++) {
    const opp = 1 - s;
    const won = ranks[s] < ranks[opp];
    const drawn = ranks[s] === ranks[opp];
    const lost = ranks[s] > ranks[opp];
    const pts = won ? 1.0 : (drawn ? 0.5 : 0.0);

    const razingsByMe = razedHills.filter(rz => rz.razer === s && rz.owner !== s);
    const firstRazeTurn = razingsByMe.length > 0 ? Math.min(...razingsByMe.map(rz => rz.turn)) : null;

    // Initial ants on hill = 1 for 2p basic maps
    const initialAnts = 1;
    const antsSpawned = Math.max(0, antIds[s].size - initialAnts);

    perSeatMetrics.push({
      seat: s,
      name: seats[s].label,
      rank: ranks[s],
      score: scores[s],
      won,
      drawn,
      lost,
      points: pts,
      hills_razed: razingsByMe.length,
      first_hill_destroy_turn: firstRazeTurn,
      collisions: collisions[s],
      combat_deaths: combatDeaths[s],
      ants_spawned: antsSpawned,
      total_orders: totalOrders[s],
      blocked_orders: blockedOrders[s],
      blocked_rate: totalOrders[s] > 0 ? blockedOrders[s] / totalOrders[s] : 0.0,
      infer_us_total: seats[s].infer_us_total,
      infer_us_max: seats[s].infer_us_max
    });
  }

  return {
    match_id: replay.match_id || '',
    map_id: replay.map_id || (replay.map && replay.map.id) || '',
    seed: replay.seed,
    turns,
    max_turns: replay.max_turns ?? 300,
    engine_digest: replay.engine_digest || '',
    reason,
    ranks,
    scores,
    razed_hills: razedHills,
    seats: perSeatMetrics
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

  // Sort files for deterministic processing
  files.sort();

  const results = [];
  for (const f of files) {
    try {
      const content = readFileSync(f, 'utf8');
      const data = analyzeReplay(content);
      data.file = f;
      results.push(data);
    } catch (err) {
      console.error(`Failed to analyze ${f}:`, err);
    }
  }

  console.log(JSON.stringify(results));
}

if (process.argv[1] && resolve(process.argv[1]) === resolve(new URL(import.meta.url).pathname)) {
  main();
}
