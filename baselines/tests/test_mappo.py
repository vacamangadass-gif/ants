import numpy as np
import pytest
import torch

from tb_baselines import nets
from tb_baselines.export import classes
from tb_baselines.train.mappo import (
    CentralizedCritic,
    CounterfactualCreditAssigner,
    MAPPOActorCritic,
    PotentialShaping,
)


def test_potential_shaping_telescoping_property():
    gamma = 0.997
    shaping = PotentialShaping(gamma=gamma)

    obs_sequence = [
        {"mine": [[2, 2], [2, 3]], "size": [24, 24], "food": [[5, 5]], "hills": []},
        {"mine": [[3, 3], [3, 4]], "size": [24, 24], "food": [[5, 5]], "hills": []},
        {"mine": [[4, 4], [4, 5]], "size": [24, 24], "food": [[5, 5]], "hills": []},
        {"mine": [[5, 5]], "size": [24, 24], "food": [], "hills": []},
    ]

    potentials = [shaping.compute_potential(obs) for obs in obs_sequence]
    assert potentials[0] > 0.0

    step_rewards = []
    t_len = len(obs_sequence)
    for t in range(t_len):
        phi_cur = potentials[t]
        is_terminal = (t == t_len - 1)
        phi_nxt = 0.0 if is_terminal else potentials[t + 1]
        r_step = shaping.step_shaping(phi_cur, phi_nxt, is_terminal=is_terminal)
        step_rewards.append(r_step)

    discounted_sum = sum((gamma ** t) * step_rewards[t] for t in range(t_len))
    assert np.isclose(discounted_sum, -potentials[0], atol=1e-5)


def test_counterfactual_baseline_unbiased_expectation():
    assigner = CounterfactualCreditAssigner()
    rng = np.random.default_rng(42)

    n_ants = 10
    action_values = rng.uniform(-1.0, 2.0, size=(n_ants, 5)).astype(np.float32)
    raw_logits = rng.normal(size=(n_ants, 5))
    exp_logits = np.exp(raw_logits - raw_logits.max(axis=1, keepdims=True))
    action_probs = (exp_logits / exp_logits.sum(axis=1, keepdims=True)).astype(np.float32)

    team_adv = 1.25
    expected_marginal = np.zeros(n_ants, dtype=np.float32)
    for a in range(5):
        chosen_actions = np.full(n_ants, a, dtype=np.int64)
        ind_adv = assigner.assign_credit(action_values, chosen_actions, action_probs, team_adv, cf_weight=1.0)
        marginal = ind_adv - team_adv
        expected_marginal += action_probs[:, a] * marginal

    assert np.allclose(expected_marginal, 0.0, atol=1e-6)


def test_mappo_actor_critic_shapes_and_separation():
    spec = classes()["micro"]
    trunk = nets.build(spec, memory="where")
    model = MAPPOActorCritic(trunk, privileged_dim=8)

    batch_size = 2
    planes = 9
    height, width = 32, 32

    boards = torch.zeros((batch_size, planes, height, width), dtype=torch.int8)
    priv = torch.zeros((batch_size, 8), dtype=torch.float32)

    logits, values = model(boards, priv)
    assert logits.shape == (batch_size, 5, height, width)
    assert values.shape == (batch_size,)

    actor_logits = model.forward_actor(boards)
    assert actor_logits.shape == (batch_size, 5, height, width)

    critic_val = model.forward_critic(boards, priv)
    assert critic_val.shape == (batch_size,)


def test_mappo_runner_telescoping_sum_conservation():
    from tb_baselines.env import Env
    from tb_baselines.train.mappo import MAPPORunner, RewardConfig

    spec = classes()["micro"]
    trunk = nets.build(spec, memory="where")
    model = MAPPOActorCritic(trunk, privileged_dim=8)
    cfg = RewardConfig(score=0.0, rival=0.0, ants=0.0, win=0.0, shaping_weight=1.0)

    with Env(waves=1, matches_per_wave=1, maps=["basic-tiny-2p"], seed=1, max_turns=5) as env:
        runner = MAPPORunner(model, env, torch.device("cpu"), cfg, seed=1)
        runner.collect(10)

        assert len(runner.done) > 0
        for traj in runner.done:
            potentials = traj.potentials
            gamma = cfg.gamma
            t_len = len(potentials)
            expected_sum = -potentials[0]
            actual_sum = sum((gamma ** t) * traj.rewards[t] for t in range(t_len))
            assert np.isclose(actual_sum, expected_sum, atol=1e-6)
