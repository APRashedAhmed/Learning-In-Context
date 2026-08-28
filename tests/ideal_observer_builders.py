"""Deterministic event-grammar builder for the ideal-observer property suite.

events -> (B, T, 5) sample tensor. Geometry lives on the x-axis (y held
constant in-bounds), so a declared velocity change is a sign flip of the
per-step velocity and a declared wall places that sample outside [RADIUS,
SIZE-RADIUS]. Correctness is pinned by the builder-validation meta-test
(hand-derived anchor) in test_ideal_observer_properties.py — iterate this
file until that meta-test is green before trusting any estimator assertion.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

SCALE = 255
MASK_COLOR = (127.0, 127.0, 127.0)
PAD_VALUE = -1.0
SIZE = 256
RADIUS = 10
Y_CONST = 128.0  # constant, in-bounds: all geometry is on x
STEP = 4.0  # base per-step speed
WALL_X = SIZE - RADIUS  # 246: a sample at WALL_X + 2 is out of bounds


@dataclass
class Event:
    velocity_change: bool = False
    at_wall: bool = (
        False  # velocity_change & at_wall => bounce; velocity_change & ~at_wall => random
    )
    color: int | None = 0  # 0/1/2 cyclic; ignored when occluded or pad
    occluded: bool = False
    pad: bool = False


def beta_mean(prior_alpha: float, prior_beta: float, successes: float, total: float) -> float:
    """Analytic posterior mean of a Beta(prior_alpha, prior_beta) after `successes`
    of `total` Bernoulli trials: (alpha + successes) / (alpha + beta + total)."""
    return (prior_alpha + successes) / (prior_alpha + prior_beta + total)


def _x_trajectory(script: list[Event]) -> list[float]:
    """Synthesise x so that the second difference is nonzero exactly at declared
    velocity-change steps and the sample is OOB exactly at declared wall steps.

    A velocity change at position index t means v[t+1] != v[t]; realise it as a
    sign flip of the running velocity entering step t+1. A wall step is centred
    on WALL_X so the flip there lands the sample just outside the box."""
    n = len(script)
    x = [0.0] * n
    v = STEP
    # anchor the first position: start at WALL_X-STEP if the first real event is a
    # wall, else mid-box, so wall flips land OOB and non-wall flips stay in-bounds.
    x[0] = SIZE / 2.0
    for t in range(1, n):
        # a declared velocity_change at position index (t-1) flips the velocity
        # going into segment t; alignment: model.velocity_change[k] <- declared[k+1].
        if script[t - 1].velocity_change:
            v = -v
        x[t] = x[t - 1] + v
        # relocate wall steps to straddle the near wall without touching the
        # second-difference sign already realised above.
        if script[t].at_wall:
            x[t] = WALL_X + 2.0  # OOB (> SIZE - RADIUS)
    return x


def build_samples(scripts: list[list[Event]], scale: int = SCALE) -> torch.Tensor:
    """Render a batch of Event scripts to a (B, T_max, 5) tensor; shorter scripts
    right-pad with PAD_VALUE on all five channels."""
    batch = len(scripts)
    t_max = max(len(s) for s in scripts)
    samples = torch.full((batch, t_max, 5), float(PAD_VALUE))
    mask_color = torch.tensor(MASK_COLOR)
    for b, script in enumerate(scripts):
        x = _x_trajectory(script)
        for t, ev in enumerate(script):
            if ev.pad:
                continue  # leave the PAD_VALUE row
            samples[b, t, 0] = x[t]
            samples[b, t, 1] = Y_CONST
            if ev.occluded:
                samples[b, t, 2:] = mask_color
            else:
                onehot = torch.zeros(3)
                onehot[ev.color] = 1.0
                samples[b, t, 2:] = onehot * scale
    return samples
