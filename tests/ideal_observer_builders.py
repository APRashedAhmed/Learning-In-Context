"""Deterministic event-grammar builder for the ideal-observer property suite.

events -> (B, T, 5) sample tensor. Geometry lives on the x-axis (y held
constant in-bounds), so a declared velocity change is a change of the
per-step velocity and a declared wall places that sample outside [RADIUS,
SIZE-RADIUS]. Correctness is pinned by the builder-validation meta-tests
(hand-derived anchors) in test_ideal_observer_properties.py — iterate this
file until those meta-tests are green before trusting any estimator assertion.
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
WALL_X = SIZE - RADIUS  # 246: the model calls a sample OOB iff x > WALL_X or x < RADIUS

# Anchors, expressed relative to WALL_X so they cannot drift apart from it.
# The model's threshold comes from bouncing_ball_task.defaults.TaskParameters
# (size_frame=(256, 256), ball_radius=10), i.e. exactly `> 246` / `< 10`.
#
#   X_HOME  = 244: in bounds (244 <= 246) yet ONE STEP from the wall zone
#                  (244 + 4 = 248 > 246), so entering and leaving the wall zone
#                  costs a velocity of order STEP rather than a ~120-unit jump.
#   X_WALL0 = 248: the first out-of-bounds rung; used for x[0] when frame 0 is
#                  itself declared a wall (there is no earlier frame to move
#                  from, so the anchor has to start outside).
X_HOME = WALL_X - STEP / 2.0  # 244.0
X_WALL0 = WALL_X + STEP / 2.0  # 248.0


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


def _is_oob(x: float) -> bool:
    """The model's own out-of-bounds test, restricted to the x axis (y is pinned
    to Y_CONST = 128, which is in bounds on every frame)."""
    return x > WALL_X or x < RADIUS


def _velocity_for(p: float, want_oob: bool, v_prev: float | None) -> float:
    """Segment velocity carrying the ball from `p` to the required side of the
    wall, guaranteed distinct from `v_prev`.

    Only called at indices where the model is FREE to observe a new velocity
    (see `_x_trajectory`), so returning something != `v_prev` is always legal —
    and, when `v_prev` is not None, mandatory: that index is a declared change.

    Every value returned is a multiple of STEP/2 = 2, so any two distinct
    velocities differ by at least 2. That matters: the detector is
    `(velocity_diff ** 2).max(-1).to(int).to(bool)`, and `.to(int)` TRUNCATES,
    so a second difference whose square is < 1 would read as NO change. Keep
    every rung on the STEP/2 lattice (min |diff| = 2, min square = 4) if this
    ladder is ever extended. The lattice is also exactly representable in
    binary floating point, so "second difference is exactly 0" on a non-change
    step is exact, not approximate.
    """
    if want_oob:
        # Smallest +STEP rung that clears the wall. Larger rungs clear it too,
        # so every fallback candidate still satisfies the wall requirement.
        k = 1
        while p + k * STEP <= WALL_X:
            k += 1
        base = k * STEP
        candidates = (base, base + STEP / 2.0, base + STEP)
    else:
        # Smallest -STEP rung that lands at or below X_HOME. Travelling in -x
        # leaves the whole box as runway for the FORCED steps that follow (B6):
        # from X_HOME a -STEP run stays in bounds for another ~58 frames, and
        # the widest velocity this ladder hands out on a one-frame wall run is
        # -STEP, on a two-frame wall run -2*STEP (~29 frames of runway).
        k = 1
        while p - k * STEP > X_HOME:
            k += 1
        base = -k * STEP
        # `base + STEP/2` undershoots the anchor by a half step: it lands at
        # most X_HOME + STEP/2 == WALL_X, still in bounds. `base - STEP/2`
        # overshoots inward, also in bounds. Both are safe fallbacks.
        candidates = (base, base + STEP / 2.0, base - STEP / 2.0)
    for v in candidates:
        if v_prev is None or v != v_prev:
            return v
    raise AssertionError("unreachable: the candidate rungs are pairwise distinct")


def _x_trajectory(script: list[Event]) -> list[float]:
    """Synthesise x so that the second difference is nonzero EXACTLY at declared
    velocity-change steps and the sample is OOB EXACTLY at declared wall steps.

    Alignment. The model computes
        velocity[k]      = x[k+1] - x[k]
        velocity_diff[k] = velocity[k+1] - velocity[k]
    so `velocity_change[k]` compares segment k+2 against segment k+1 and is
    "centred" on position index k+1 (that is also how the model pairs it with
    `positions_oob[:, 1:-1]`). Writing seg[t] = x[t] - x[t-1]:

      * seg[1] is UNCONSTRAINED — no earlier segment exists to differ from, so
        the model can never see a change at position index 0.
      * for t >= 2, seg[t] must equal seg[t-1] unless script[t-1].velocity_change,
        in which case it must differ.

    Hence `velocity_change[k] == script[k+1].velocity_change` by construction,
    for k in 0..n-3. A velocity_change declared on the FIRST or LAST frame has
    no corresponding model index and is silently unobservable — that is a
    property of the detector, not of this builder.

    Geometry (this is the B2 comment, now implemented rather than merely
    described). x[0] is the interior anchor X_HOME, or X_WALL0 when frame 0 is
    itself declared a wall. Every later position is reached by ADDING a velocity
    — there is no relocation/teleport step. The wall zone is entered and left at
    indices where a new velocity is legal, using the smallest STEP rung that
    does the job, so a wall never manufactures an undeclared second-difference
    kink (defect B1: the old builder rewrote wall samples to WALL_X + 2, and
    that ~120-unit jump was itself a kink the models scored as an extra bounce).

    Prefix causality (property B5: build_samples([script[:k]]) must equal
    build_samples([script])[:, :k] for every k). x[0] reads script[0] only; for
    t >= 1, x[t] is a function of (x[t-1], v, script[t-1].velocity_change,
    script[t].at_wall). By induction x[t] depends on script[0..t] and NOTHING
    later. Note in particular that the wall decision at step t consults
    script[t].at_wall — an event that any truncation reaching index t must
    retain — never script[t+1] or beyond. So truncating at k reruns the same
    recurrence over the same inputs for every t < k and reproduces x[:k]
    exactly.

    Bounds (B6). Non-wall positions travel in -x from X_HOME = 244 at |v| = STEP
    in the common case, so a 32-frame event-free script ends at 244 - 4*31 = 120,
    comfortably inside [RADIUS, WALL_X]; the box affords ~58 such frames. A
    declared wall costs one wider rung on the way back in (see `_velocity_for`).

    Raises ValueError if a script is geometrically infeasible — e.g. a wall
    declared mid-run with no preceding velocity_change and no outward motion to
    continue. Better a loud failure than a silently dishonest trajectory.
    """
    n = len(script)
    if n == 0:
        return []
    x = [0.0] * n
    x[0] = X_WALL0 if script[0].at_wall else X_HOME
    v: float | None = None
    for t in range(1, n):
        free = t == 1 or script[t - 1].velocity_change
        if free:
            v = _velocity_for(x[t - 1], script[t].at_wall, v)
        x[t] = x[t - 1] + v
        # Pad frames are never rendered by build_samples, so their synthesised
        # position is meaningless and must not be checked.
        if not script[t].pad and _is_oob(x[t]) != script[t].at_wall:
            raise ValueError(
                f"infeasible script at index {t}: x={x[t]} is "
                f"{'out of' if _is_oob(x[t]) else 'in'} bounds but at_wall="
                f"{script[t].at_wall}. A wall frame must start the script, follow a "
                "declared velocity_change, or continue an already-outward run."
            )
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
