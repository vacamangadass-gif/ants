from __future__ import annotations

import argparse
import contextlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import nets
from ..env import Env, orders_from_indices
from ..export import budget, classes
from ..planes import N_MOVES, MOVES
from .bc import device

ANT_BUCKET = 256


def _bucket(n: int, q: int = ANT_BUCKET) -> int:
    return max(q, -(-n // q) * q)


def _autocast(dev: torch.device, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    return torch.autocast(dev.type, dtype=torch.float16)


@dataclass
class RewardConfig:
    score: float = 1.0
    rival: float = 0.5
    ants: float = 0.02
    win: float = 1.0
    gamma: float = 0.997
    lam: float = 0.95
    shaping_weight: float = 1.0
    cf_weight: float = 0.5


class PotentialShaping:
    def __init__(self, gamma: float = 0.997):
        self.gamma = gamma

    def compute_potential(self, obs: dict) -> float:
        mine = np.asarray(obs["mine"], dtype=np.int32).reshape(-1, 2)
        n_ants = len(mine)
        if n_ants == 0:
            return 0.0

        rows, cols = obs["size"]
        food = np.asarray(obs["food"], dtype=np.int32).reshape(-1, 2)
        food_potential = 0.0
        if len(food) > 0:
            dr = np.abs(mine[:, None, 0] - food[None, :, 0])
            dc = np.abs(mine[:, None, 1] - food[None, :, 1])
            dr = np.minimum(dr, rows - dr)
            dc = np.minimum(dc, cols - dc)
            dist_food = dr + dc
            min_dist = dist_food.min(axis=1)
            food_potential = float(np.sum(1.0 / (1.0 + min_dist)))

        congestion_penalty = 0.0
        if n_ants > 1:
            dr_ants = np.abs(mine[:, None, 0] - mine[None, :, 0])
            dc_ants = np.abs(mine[:, None, 1] - mine[None, :, 1])
            dr_ants = np.minimum(dr_ants, rows - dr_ants)
            dc_ants = np.minimum(dc_ants, cols - dc_ants)
            dist_ants = dr_ants + dc_ants
            close_pairs = int(np.sum((dist_ants > 0) & (dist_ants <= 1))) // 2
            congestion_penalty = float(close_pairs)

        enemy_hills = [h for h in obs.get("hills", []) if h[2] != 0]
        advance_potential = 0.0
        if len(enemy_hills) > 0:
            hills_arr = np.asarray([[h[0], h[1]] for h in enemy_hills], dtype=np.int32)
            dr_hill = np.abs(mine[:, None, 0] - hills_arr[None, :, 0])
            dc_hill = np.abs(mine[:, None, 1] - hills_arr[None, :, 1])
            dr_hill = np.minimum(dr_hill, rows - dr_hill)
            dc_hill = np.minimum(dc_hill, cols - dc_hill)
            dist_hill = dr_hill + dc_hill
            min_hill_dist = dist_hill.min(axis=1)
            advance_potential = float(np.sum(1.0 / (1.0 + min_hill_dist)))

        phi = 0.5 * float(n_ants) + 1.0 * food_potential - 0.2 * congestion_penalty + 0.5 * advance_potential
        return phi

    def step_shaping(self, phi_cur: float, phi_nxt: float, is_terminal: bool) -> float:
        if is_terminal:
            target_phi = 0.0
        else:
            target_phi = phi_nxt
        return self.gamma * target_phi - phi_cur


class CounterfactualCreditAssigner:
    DIR_DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]

    def evaluate_actions(self, obs: dict, water_mask: np.ndarray, food_set: set[tuple[int, int]]) -> np.ndarray:
        mine = np.asarray(obs["mine"], dtype=np.int32).reshape(-1, 2)
        n_ants = len(mine)
        if n_ants == 0:
            return np.zeros((0, 5), dtype=np.float32)

        rows, cols = obs["size"]
        values = np.zeros((n_ants, 5), dtype=np.float32)

        for a_idx, (dr, dc) in enumerate(self.DIR_DELTAS):
            for i in range(n_ants):
                nr = (mine[i, 0] + dr) % rows
                nc = (mine[i, 1] + dc) % cols
                if water_mask[nr, nc]:
                    values[i, a_idx] -= 1.0
                    continue

                is_adjacent_food = False
                for f_dr, f_dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:
                    check_r = (nr + f_dr) % rows
                    check_c = (nc + f_dc) % cols
                    if (check_r, check_c) in food_set:
                        is_adjacent_food = True
                        break

                if is_adjacent_food:
                    values[i, a_idx] += 1.5

                if a_idx == 4:
                    values[i, a_idx] -= 0.05
                else:
                    values[i, a_idx] += 0.05

        return values

    def assign_credit(
        self,
        action_values: np.ndarray,
        chosen_actions: np.ndarray,
        action_probs: np.ndarray,
        team_advantage: float,
        cf_weight: float = 0.5,
    ) -> np.ndarray:
        n_ants = len(chosen_actions)
        if n_ants == 0:
            return np.zeros(0, dtype=np.float32)

        counterfactual_baseline = np.sum(action_probs * action_values, axis=1)
        chosen_values = action_values[np.arange(n_ants), chosen_actions]
        marginal_contribution = chosen_values - counterfactual_baseline
        individual_advantages = team_advantage + cf_weight * marginal_contribution
        return individual_advantages.astype(np.float32)


class CentralizedCritic(nn.Module):
    def __init__(self, in_channels: int = 28, privileged_dim: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.privileged_dim = privileged_dim
        self.conv_pool = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(32 * 2 + privileged_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, features: torch.Tensor, privileged: torch.Tensor) -> torch.Tensor:
        reduced = self.conv_pool(features)
        mean_pooled = reduced.mean(dim=(2, 3))
        max_pooled = reduced.amax(dim=(2, 3))
        combined = torch.cat([mean_pooled, max_pooled, privileged.float()], dim=1)
        return self.value_head(combined).squeeze(-1)


class MAPPOActorCritic(nn.Module):
    def __init__(self, trunk: nn.Module, privileged_dim: int = 8):
        super().__init__()
        self.trunk = trunk
        self.base_trunk = getattr(trunk, "trunk", trunk)
        self.critic = CentralizedCritic(self.base_trunk.channels, privileged_dim=privileged_dim)

    def forward_actor(self, board: torch.Tensor) -> torch.Tensor:
        f = self.base_trunk.features(board.float())
        return self.base_trunk.head(f)

    def forward_critic(self, board: torch.Tensor, privileged: torch.Tensor) -> torch.Tensor:
        f = self.base_trunk.features(board.float())
        return self.critic(f, privileged)

    def forward(self, board: torch.Tensor, privileged: torch.Tensor | None = None):
        f = self.base_trunk.features(board.float())
        logits = self.base_trunk.head(f)
        if privileged is not None:
            values = self.critic(f, privileged)
            return logits, values
        return logits


@dataclass
class MAPPOTrajectory:
    obs: list[dict] = field(default_factory=list)
    boards: list[np.ndarray] = field(default_factory=list)
    privileged: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    action_probs: list[np.ndarray] = field(default_factory=list)
    logp: list[np.ndarray] = field(default_factory=list)
    action_values: list[np.ndarray] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    potentials: list[float] = field(default_factory=list)
    last_score: int = 0
    last_rival: int = 0
    last_ants: int = 0
    primed: bool = False
    finished: bool = False


def extract_privileged_vector(obs: dict, scores: list[int] | None, seat: int, turn: int) -> np.ndarray:
    own_score = float(scores[seat]) if scores else 1.0
    rival_score = float(max(s for i, s in enumerate(scores) if i != seat)) if scores and len(scores) > 1 else 1.0
    score_diff = own_score - rival_score
    own_ants = float(len(obs.get("mine", [])))
    rival_ants = float(len(obs.get("foes", [])))
    ant_diff = own_ants - rival_ants
    food_count = float(len(obs.get("food", [])))
    turn_ratio = float(turn) / 300.0
    return np.array([
        own_score,
        rival_score,
        score_diff,
        own_ants,
        rival_ants,
        ant_diff,
        food_count,
        turn_ratio,
    ], dtype=np.float32)


def gae(rewards: list[float], values: list[float], bootstrap: float, gamma: float, lam: float):
    adv, out = [], 0.0
    nxt = bootstrap
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * nxt - values[t]
        out = delta + gamma * lam * out
        adv.append(out)
        nxt = values[t]
    adv.reverse()
    returns = [a + v for a, v in zip(adv, values)]
    return adv, returns


class _AntIndex:
    def __init__(self, seats, indices):
        mine = [np.asarray(seats[i].obs["mine"], dtype=np.int64).reshape(-1, 2) for i in indices]
        self.counts = [len(m) for m in mine]
        self.n = sum(self.counts)
        idx = np.zeros((3, _bucket(self.n)), dtype=np.int64)
        if self.n:
            idx[0, : self.n] = np.repeat(np.arange(len(mine)), self.counts)
            idx[1:, : self.n] = np.concatenate(mine).T
        self._idx = torch.from_numpy(idx)
        self.board, self.r, self.c = self._idx

    def to(self, dev):
        self._idx = self._idx.to(dev)
        self.board, self.r, self.c = self._idx
        return self


class MAPPORunner:
    def __init__(
        self,
        model: MAPPOActorCritic,
        env: Env,
        dev: torch.device,
        reward_config: RewardConfig,
        seed: int = 0,
        amp: bool = False,
    ):
        self.model = model
        self.env = env
        self.dev = dev
        self.reward = reward_config
        self.amp = amp
        self.rng = np.random.default_rng(seed)
        self.step = env.reset()
        self.open: dict[tuple[int, int], MAPPOTrajectory] = {}
        self.memories: dict[tuple[int, int], np.ndarray] = {}
        self.done: list[MAPPOTrajectory] = []
        self.returns: list[float] = []
        self.potential_shaping = PotentialShaping(gamma=reward_config.gamma)
        self.credit_assigner = CounterfactualCreditAssigner()

    @torch.no_grad()
    def act(self, boards: np.ndarray, privileged: np.ndarray, rows: _AntIndex):
        t_boards = torch.from_numpy(boards).to(self.dev)
        t_priv = torch.from_numpy(privileged).to(self.dev)
        with _autocast(self.dev, self.amp):
            logits, values = self.model(t_boards, t_priv)
        per_ant = logits[rows.board, :, rows.r, rows.c].float()
        logp_all = torch.log_softmax(per_ant, dim=-1)
        probs_all = torch.softmax(per_ant, dim=-1)
        host_logp = logp_all.cpu().numpy()
        host_probs = probs_all.cpu().numpy()
        host_values = values.float().cpu().numpy()

        picks = np.zeros(rows.n, dtype=np.int64)
        ant_logp = np.zeros(rows.n, dtype=np.float32)
        ant_probs = np.zeros((rows.n, N_MOVES), dtype=np.float32)

        if rows.n > 0:
            sub_logp = host_logp[: rows.n]
            sub_probs = host_probs[: rows.n]
            cdf = np.cumsum(sub_probs, axis=1)
            u = self.rng.random(rows.n) * cdf[:, -1]
            picks = np.minimum((cdf <= u[:, None]).sum(axis=1), N_MOVES - 1)
            ant_logp = sub_logp[np.arange(rows.n), picks]
            ant_probs = sub_probs

        return picks, ant_logp, ant_probs, host_values

    def collect(self, steps: int) -> dict:
        self.done, self.returns = [], []
        for _ in range(steps):
            actions: list[str | None] = [None] * len(self.step.seats)

            for group in self.step.groups:
                priv_list = []
                board_9_list = []
                for k, idx in enumerate(group.indices):
                    seat = self.step.seats[idx]
                    scores = self.step.scores.get(seat.ep)
                    priv_vec = extract_privileged_vector(seat.obs, scores, seat.seat, seat.turn)
                    priv_list.append(priv_vec)

                    key = (seat.ep, seat.seat)
                    rows, cols = seat.obs["size"]
                    if key not in self.memories or self.memories[key].shape[2:] != (rows, cols):
                        self.memories[key] = np.zeros((1, 2, rows, cols), dtype=np.int8)
                    mem = self.memories[key]
                    b7 = group.boards[k : k + 1]
                    b9 = np.concatenate([b7, mem], axis=1)
                    board_9_list.append(b9)

                    seen = b7[:, [2, 5]]
                    vis = (b7[:, 6:7] != 0)
                    self.memories[key] = np.where(vis, seen, mem).astype(np.int8)

                group_priv = np.stack(priv_list, axis=0)
                group_boards_9 = np.concatenate(board_9_list, axis=0)

                rows = _AntIndex(self.step.seats, group.indices).to(self.dev)
                picks, ant_logp, ant_probs, values = self.act(group_boards_9, group_priv, rows)
                orders = orders_from_indices(picks, rows.counts)

                at = 0
                for k, i in enumerate(group.indices):
                    n = rows.counts[k]
                    actions[i] = orders[k]
                    seat = self.step.seats[i]
                    seat_picks = picks[at : at + n]
                    seat_logp = ant_logp[at : at + n]
                    seat_probs = ant_probs[at : at + n]

                    water_mask = self._get_water_mask(seat.obs)
                    food_set = {tuple(f) for f in seat.obs.get("food", [])}
                    action_values = self.credit_assigner.evaluate_actions(seat.obs, water_mask, food_set)

                    self._record(
                        seat,
                        group_boards_9[k : k + 1],
                        group_priv[k : k + 1],
                        seat_picks,
                        seat_logp,
                        seat_probs,
                        action_values,
                        float(values[k]),
                    )
                    at += n

            prev = self.step
            self.step = self.env.step([a if a is not None else "" for a in actions])
            self._reward(prev)
            self._close(self.step.ended)

        return {
            "episodes": len(self.returns),
            "mean_return": float(np.mean(self.returns)) if self.returns else 0.0,
        }

    def _get_water_mask(self, obs: dict) -> np.ndarray:
        rows, cols = obs["size"]
        mask = np.zeros((rows, cols), dtype=bool)
        water_spec = obs.get("water", {})
        if "rle" in water_spec:
            rle = water_spec["rle"]
            flat = np.zeros(rows * cols, dtype=bool)
            idx = 0
            val = False
            for count in rle:
                if val:
                    flat[idx : idx + count] = True
                idx += count
                val = not val
            mask = flat.reshape(rows, cols)
        return mask

    def _record(self, seat, board, priv, picks, logp, probs, action_values, value) -> None:
        key = (seat.ep, seat.seat)
        traj = self.open.setdefault(key, MAPPOTrajectory())
        if not traj.obs:
            traj.last_ants = len(seat.obs["mine"])
        traj.obs.append(seat.obs)
        traj.boards.append(board)
        traj.privileged.append(priv)
        traj.actions.append(picks.copy())
        traj.logp.append(logp.copy())
        traj.action_probs.append(probs.copy())
        traj.action_values.append(action_values.copy())
        traj.values.append(value)
        phi = self.potential_shaping.compute_potential(seat.obs)
        traj.potentials.append(phi)

    def _reward(self, prev) -> None:
        r = self.reward
        next_seats = {(s.ep, s.seat): s for s in self.step.seats}
        for seat in prev.seats:
            key = (seat.ep, seat.seat)
            traj = self.open.get(key)
            if traj is None or len(traj.rewards) >= len(traj.obs):
                continue
            scores = self.step.scores.get(seat.ep) or {e.ep: e.scores for e in self.step.ended}.get(seat.ep)
            if scores is None:
                traj.rewards.append(0.0)
                continue
            mine = scores[seat.seat]
            rival = max(s for i, s in enumerate(scores) if i != seat.seat)
            ants = len(next_seats[key].obs["mine"]) if key in next_seats else len(seat.obs["mine"])
            if not traj.primed:
                traj.last_score, traj.last_rival, traj.last_ants = mine, rival, len(seat.obs["mine"])
                traj.primed = True

            env_reward = (
                r.score * (mine - traj.last_score)
                - r.rival * (rival - traj.last_rival)
                + r.ants * (ants - traj.last_ants)
            )
            traj.last_score, traj.last_rival, traj.last_ants = mine, rival, ants

            cur_idx = len(traj.rewards)
            phi_cur = traj.potentials[cur_idx]
            if key in next_seats:
                phi_nxt = self.potential_shaping.compute_potential(next_seats[key].obs)
                shaping_r = self.potential_shaping.step_shaping(phi_cur, phi_nxt, is_terminal=False)
            else:
                shaping_r = self.potential_shaping.step_shaping(phi_cur, 0.0, is_terminal=True)

            total_r = env_reward + r.shaping_weight * shaping_r
            traj.rewards.append(float(total_r))

    def _close(self, ended) -> None:
        for e in ended:
            for seat in range(e.seat_count):
                self.memories.pop((e.ep, seat), None)
                traj = self.open.pop((e.ep, seat), None)
                if traj is None or not traj.obs:
                    continue
                best = min(e.ranks)
                outcome = 1.0 if e.ranks[seat] == best and e.ranks.count(best) == 1 else (
                    0.0 if e.ranks[seat] == best else -1.0
                )
                while len(traj.rewards) < len(traj.obs):
                    traj.rewards.append(0.0)

                traj.rewards[-1] += self.reward.win * outcome
                traj.finished = True
                self.done.append(traj)
                self.returns.append(float(sum(traj.rewards)))

    def harvest(self) -> list[MAPPOTrajectory]:
        batch = list(self.done)
        for key, traj in list(self.open.items()):
            if len(traj.rewards) >= len(traj.obs) and traj.obs:
                batch.append(traj)
                self.open[key] = MAPPOTrajectory(
                    last_score=traj.last_score,
                    last_rival=traj.last_rival,
                    last_ants=traj.last_ants,
                    primed=True,
                )
        return batch


@dataclass
class _MAPPOMinibatch:
    boards: torch.Tensor
    privileged: torch.Tensor
    board: torch.Tensor
    r: torch.Tensor
    c: torch.Tensor
    act: torch.Tensor
    ret: torch.Tensor
    old_lp: torch.Tensor
    adv: torch.Tensor
    mask: torch.Tensor
    n: int


class _MAPPOGroup:
    def __init__(self, samples, members: list[int], dev):
        self.n = len(members)
        mine = [np.asarray(samples[i]["obs"]["mine"], dtype=np.int64).reshape(-1, 2) for i in members]
        self.counts = np.array([len(m) for m in mine], dtype=np.int64)
        self.starts = np.cumsum(self.counts) - self.counts
        rc = np.concatenate(mine)
        self.r, self.c = rc[:, 0], rc[:, 1]
        self.act = np.concatenate([samples[i]["actions"] for i in members]).astype(np.int64)
        self.old_lp = np.concatenate([samples[i]["logp"] for i in members]).astype(np.float32)
        self.adv = np.concatenate([samples[i]["individual_adv"] for i in members]).astype(np.float32)
        self.ret = np.array([samples[i]["ret"] for i in members], dtype=np.float32)

        self.boards = torch.from_numpy(np.concatenate([samples[i]["board"] for i in members], 0)).to(dev)
        self.privileged = torch.from_numpy(np.concatenate([samples[i]["privileged"] for i in members], 0)).to(dev)

    def minibatches(self, order: np.ndarray, size: int, dev):
        for start in range(0, len(order), size):
            sel = order[start : start + size]
            cnt = self.counts[sel]
            n = int(cnt.sum())
            m = _bucket(n)
            ant = np.arange(n) + np.repeat(self.starts[sel] - (np.cumsum(cnt) - cnt), cnt)

            ib = np.zeros(len(sel) + 4 * m, dtype=np.int64)
            ib[: len(sel)] = sel
            per = ib[len(sel) :].reshape(4, m)
            per[0, :n] = np.repeat(np.arange(len(sel)), cnt)
            per[1, :n] = self.r[ant]
            per[2, :n] = self.c[ant]
            per[3, :n] = self.act[ant]

            fb = np.zeros(len(sel) + 3 * m, dtype=np.float32)
            fb[: len(sel)] = self.ret[sel]
            fper = fb[len(sel) :].reshape(3, m)
            fper[0, :n] = self.old_lp[ant]
            fper[1, :n] = self.adv[ant]
            fper[2, :n] = 1.0

            ib_t = torch.from_numpy(ib).to(dev)
            fb_t = torch.from_numpy(fb).to(dev)

            b_len = len(sel)
            board, r, c, act = ib_t[b_len:].view(4, m)
            old_lp, adv, mask = fb_t[b_len:].view(3, m)

            yield _MAPPOMinibatch(
                self.boards.index_select(0, ib_t[:b_len]),
                self.privileged.index_select(0, ib_t[:b_len]),
                board,
                r,
                c,
                act,
                fb_t[:b_len],
                old_lp,
                adv,
                mask,
                n,
            )


def update_mappo(
    model: MAPPOActorCritic,
    opt: torch.optim.Optimizer,
    batch: list[MAPPOTrajectory],
    r: RewardConfig,
    dev: torch.device,
    epochs: int,
    clip: float,
    entropy: float,
    minibatch: int,
    amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
) -> dict:
    samples = []
    credit_assigner = CounterfactualCreditAssigner()

    for traj in batch:
        bootstrap = 0.0 if traj.finished else traj.values[-1]
        team_advs, rets = gae(traj.rewards, traj.values, bootstrap, r.gamma, r.lam)
        for t in range(len(traj.obs)):
            if len(traj.actions[t]) == 0:
                continue

            ind_adv = credit_assigner.assign_credit(
                traj.action_values[t],
                traj.actions[t],
                traj.action_probs[t],
                team_advs[t],
                cf_weight=r.cf_weight,
            )
            samples.append({
                "obs": traj.obs[t],
                "board": traj.boards[t],
                "privileged": traj.privileged[t],
                "actions": traj.actions[t],
                "logp": traj.logp[t],
                "team_adv": team_advs[t],
                "individual_adv": ind_adv,
                "ret": rets[t],
            })

    if not samples:
        return {"samples": 0}

    all_ind_adv = np.concatenate([s["individual_adv"] for s in samples])
    std = float(all_ind_adv.std())
    mean = float(all_ind_adv.mean())
    if std > 1e-8:
        for s in samples:
            s["individual_adv"] = (s["individual_adv"] - mean) / (std + 1e-8)

    by_size: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        by_size[tuple(s["obs"]["size"])].append(i)

    groups = [_MAPPOGroup(samples, members, dev) for members in by_size.values()]

    rows = []
    for _ in range(epochs):
        for group in groups:
            for mb in group.minibatches(np.random.permutation(group.n), minibatch, dev):
                with _autocast(dev, amp):
                    logits, values = model(mb.boards, mb.privileged)
                per_ant = logits[mb.board, :, mb.r, mb.c].float()
                logp = torch.log_softmax(per_ant, dim=-1)
                new_lp = logp.gather(1, mb.act.unsqueeze(1)).squeeze(1)

                ratio = torch.exp(new_lp - mb.old_lp)
                surr1 = ratio * mb.adv
                surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * mb.adv
                policy_loss = -(torch.min(surr1, surr2) * mb.mask).sum() / mb.n
                value_loss = F.mse_loss(values.float(), mb.ret)

                p_log_p = logp.exp() * logp.clamp(min=torch.finfo(logp.dtype).min)
                ent = (-p_log_p.sum(-1) * mb.mask).sum() / mb.n
                loss = policy_loss + 0.5 * value_loss - entropy * ent

                opt.zero_grad(set_to_none=True)
                if scaler is None:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    opt.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    scaler.step(opt)
                    scaler.update()

                clipped = (((ratio - 1.0).abs() > clip).float() * mb.mask).sum() / mb.n
                rows.append(torch.stack([policy_loss, value_loss, ent, clipped]).detach())

    if not rows:
        return {"samples": len(samples)}

    means = torch.stack(rows).cpu().numpy().astype(np.float64).mean(axis=0)
    return dict(zip(("policy", "value", "entropy", "clipped"), map(float, means))) | {
        "samples": len(samples),
        "adv_mean": mean,
        "adv_std": std,
    }
