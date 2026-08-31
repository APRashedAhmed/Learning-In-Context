"""Tier 1+2 property (specification) suite for the ideal-observer models.

Cells #1-#10 follow the sibling test-suite spec (C1-C12, status matrix); the two
`v2_*` cells are new E2 cells added by the model-fixes work-unit. Bug-cells are
`xfail(strict, raises=AssertionError)` — the living bug-ledger: fixing a bug
flips its cell to a loud XPASS, signalling the marker's removal.

NB the current IdealCountingObserver crashes (`ValueError: max() arg is an empty
sequence`) on any batch with ZERO grayzone frames (Tier-3, deferred), so every
ICO-fed batch below carries at least one occluded frame in some row.

OPEN (C11 agreement): the model-side off-by-one that the B1 builder fix exposed —
ICO paired `velocity_change[k]` (centred on position k+1) with the colour change
arriving at position k+2 — is FIXED (ICO-A), together with the dead
explained-exit-change branch (ICO-B). ICO and V2 now agree on the underlying
event tally for `_AGREEMENT_SCRIPT` (0 contingent failures, 1 contingent
success), but their posterior means still differ: ICO's `get_dist_params` runs
with `offset=2`, which DUPLICATES rows 0 and 1 of every count vector, and this
script's only event sits at index 1. **C11 therefore remains UNCOVERED**, now
for a single, precisely-named reason — defect E5, ruled an accident and kept
inert/deferred by the PI (design doc, Q5). See `test_ico_v2_agreement`.
"""

import pytest
import torch

from learning_in_context.models.ideal_observer import (
    IdealBayesianObserver,
    IdealCountingObserver,
    IdealCountingObserverV2,
)
from tests.ideal_observer_builders import Event, beta_mean, build_samples

# IBO's __init__ has three REQUIRED positional args
# (probability_color_change_no_velocity_change,
#  probability_color_change_on_velocity_change, probability_velocity_change);
# ICO wants prog_bar=False; V2 takes neither. One helper keeps every call site
# from re-deriving that and from calling a bare IdealBayesianObserver().
_IBO_RATES = (0.05, 0.5, 0.05)  # same values as the golden test


def _construct(ctor):
    if ctor is IdealBayesianObserver:
        return ctor(*_IBO_RATES)
    if ctor is IdealCountingObserver:
        return ctor(prog_bar=False)
    return ctor()


def _gray(**kwargs):
    """An occluded (grayzone) frame; color is ignored while occluded."""
    return Event(color=1, occluded=True, **kwargs)


# --- Module-level Event scripts for the Tier-1 cells --------------------------

# #3 batch invariance: the Task-4 audit's verified-non-vacuous SHORT/LONG shape.
# SHORT carries a velocity change INSIDE its grayzone (index >= 2) and a colour
# change ACROSS it; LONG's grayzone run is strictly longer, so it drives the old
# batch-global max_grayzone_diff and perturbs SHORT's estimates while broken.
_GRAYZONE_BOUNCE_SCRIPT = [
    Event(color=0),
    Event(color=0),
    _gray(velocity_change=True),
    _gray(),
    _gray(),
    Event(color=2),
    Event(color=2),
]
_LONGER_GRAYZONE_SCRIPT = [
    Event(color=0),
    Event(color=0),
    _gray(),
    _gray(),
    _gray(),
    _gray(),
    Event(color=2),
    Event(color=2),
]

# #7-V2 rate recovery: FULLY VISIBLE (V2 tolerates no-grayzone input; a grayzone
# is not count-inert for V2), event-free first two frames, no bounces anywhere.
# Hand count: transitions 0->0 u, 0->1 C, 1->1 u, 1->2 C, 2->2 u, 2->2 u.
_V2_VISIBLE_RANDOM_SCRIPT = [
    Event(color=0),
    Event(color=0),
    Event(color=1),
    Event(color=1),
    Event(color=2),
    Event(color=2),
    Event(color=2),
]
_K_CHANGES, _N_STEPS = 2, 6

# #8-V2 dissociation: the ONLY colour change coincides with the only bounce.
# Geometry after the B1 builder fix: x = [248, 252, 244, 236, 228] (frame 0 is
# now itself rendered OOB, which the pre-fix builder silently dropped), so
# velocity_change = [T, F, F] and positions_oob = [T, T, F, F, F]. V2 reads
# positions only through _derive_bounce, whose bounce[:, 0] is forced False, so
# bounce = [F, T, F, F, F] — bit-identical to the pre-fix batch. This cell is
# INERT to the fix.
_BOUNCE_ONLY_VISIBLE_SCRIPT = [
    Event(color=0, at_wall=True),
    Event(color=1, velocity_change=True, at_wall=True),
    Event(color=1),
    Event(color=1),
    Event(color=1),
]

# #9 causality: an early grayzone run (len 4, velocity change at run start) whose
# exit cell sits 3 cells from the change, followed by a strictly longer FUTURE
# run (len 6). Full-script window = 1 + 6//2 = 4 reaches the exit cell; the
# truncated script's window = 1 + 4//2 = 3 does not — so the attribution (and the
# estimate) at _T_CUT depends on the future run. Verified non-vacuous pre-fix:
# m_ovc[0, _T_CUT] full 0.3333 vs truncated 0.5.
_CAUSAL_SCRIPT = [
    Event(color=0),
    Event(color=0),
    _gray(velocity_change=True),
    _gray(),
    _gray(),
    _gray(),
    Event(color=1),
    Event(color=1),
    _gray(),
    _gray(),
    _gray(),
    _gray(),
    _gray(),
    _gray(),
    Event(color=2),
    Event(color=2),
]
_T_CUT = 7

# #10 agreement: a visible bounce-coincident colour change plus a grayzone with
# a colour change across it (the grayzone also keeps ICO from crashing).
# ONE velocity change is declared, at script index 2. After the B1 builder fix
# the trajectory is x = [244, 248, 252, 244, 236, 228, 220], giving
# velocity_change [F, T, F, F, F] and exactly ONE detected bounce; before the
# fix the wall relocation added a phantom bounce at velocity index 0. See
# test_builder_validation_agreement_script_single_bounce for the hand table and
# test_ico_v2_agreement for what the extra bounce had been propping up.
_AGREEMENT_SCRIPT = [
    Event(color=0),
    Event(color=0, at_wall=True),
    Event(color=1, velocity_change=True, at_wall=True),
    Event(color=1),
    _gray(),
    Event(color=2),
    Event(color=2),
]

# E2 cells: pinned to exactly three frames — the [2, 2, 1, 1] expectation in
# test_v2_grayzone_classified_occluded is derived for this script and does not
# survive changing its length.
_BENIGN_GRAYZONE_SCRIPT = [
    Event(color=0),
    Event(color=0, occluded=True),  # the single [127,127,127] frame
    Event(color=0),
]


class TestBuilderValidation:
    """Hand-derived reference: x positions and their second differences are written
    out by hand; the expected oob/velocity_change/bounce/random are stated WITHOUT
    calling the model, then checked against the model's detectors. Breaks the
    builder<->detector tautology (sibling spec A13).

    Each literal batch carries a second, grayzone-bearing row (same positions,
    one occluded frame) purely so the ICO does not hit its no-grayzone ValueError
    (Tier-3); the hand table is asserted on row 0, and the detectors under test
    are position-derived, so the extra row cannot perturb them.

    The first two cells feed LITERAL tensors, never the builder — they are the
    detector's own anchor and must stay untouched by builder changes. The
    builder-fed cells below assert EXACT boolean lists rather than `.any()`:
    `.any()` cannot tell one declared bounce from two, which is exactly how
    builder defect B1 (a wall relocation that was itself a second-difference
    kink) survived undetected here.
    """

    def test_builder_validation_random_change(self):
        # Hand table (x on a 256 box, radius 10, y=128 in-bounds):
        #   x      = [100, 110, 120, 124, 128]
        #   v=diff = [ 10,  10,   4,   4]
        #   v2diff = [  0,  -6,   0]   -> nonzero at position index 2 only
        #   oob    = all in [10, 246]  -> no walls
        # => velocity_change at position index 2; no bounce (oob False there).
        samples = torch.tensor(
            [
                [
                    [100.0, 128, 1, 0, 0],
                    [110.0, 128, 1, 0, 0],
                    [120.0, 128, 1, 0, 0],
                    [124.0, 128, 1, 0, 0],
                    [128.0, 128, 1, 0, 0],
                ],
                [
                    [100.0, 128, 1, 0, 0],
                    [110.0, 128, 1, 0, 0],
                    [120.0, 128, 127, 127, 127],
                    [124.0, 128, 1, 0, 0],
                    [128.0, 128, 1, 0, 0],
                ],
            ]
        )
        ico = IdealCountingObserver(prog_bar=False)
        ico(samples, return_means=True)
        # alignment: model.velocity_change[k] corresponds to position index k+1.
        assert ico.velocity_change[0].tolist() == [False, True, False]
        assert ico.positions_oob[0].tolist() == [False, False, False, False, False]
        assert ico.velocity_change_bounce[0].tolist() == [False, False, False]
        assert ico.velocity_change_random[0].tolist() == [False, True, False]

    def test_builder_validation_bounce(self):
        # Hand table: a wall bounce on the right edge (WALL_X = 246).
        #   x      = [240, 248, 240, 232, 224]
        #   v=diff = [  8,  -8,  -8,  -8]
        #   v2diff = [-16,   0,   0]   -> nonzero at position index 1 only
        #   oob    = [F, T, F, F, F]   (248 > 246) -> wall at index 1
        # => velocity_change at index 1 AND oob there => bounce.
        samples = torch.tensor(
            [
                [
                    [240.0, 128, 1, 0, 0],
                    [248.0, 128, 1, 0, 0],
                    [240.0, 128, 1, 0, 0],
                    [232.0, 128, 1, 0, 0],
                    [224.0, 128, 1, 0, 0],
                ],
                [
                    [240.0, 128, 1, 0, 0],
                    [248.0, 128, 1, 0, 0],
                    [240.0, 128, 127, 127, 127],
                    [232.0, 128, 1, 0, 0],
                    [224.0, 128, 1, 0, 0],
                ],
            ]
        )
        ico = IdealCountingObserver(prog_bar=False)
        ico(samples, return_means=True)
        assert ico.velocity_change[0].tolist() == [True, False, False]
        assert ico.positions_oob[0].tolist() == [False, True, False, False, False]
        assert ico.velocity_change_bounce[0].tolist() == [True, False, False]
        assert ico.velocity_change_random[0].tolist() == [False, False, False]

    def test_builder_validation_matches_declared_events(self):
        # The builder, given the equivalent Event scripts, reproduces the hand
        # tables. EXACT lists, not `.any()` (weakness W1): `.any()` could not
        # distinguish one declared bounce from the two the pre-B1-fix builder
        # actually produced, which is how defect B1 survived this cell.
        #
        # Builder constants: X_HOME = 244 (in bounds), X_WALL0 = 248 (OOB),
        # STEP = 4, WALL_X = 246; OOB iff x > 246 or x < 10. Detector alignment:
        # velocity_change[k] = (x[k+2]-x[k+1]) != (x[k+1]-x[k]), centred on
        # position index k+1, and paired with positions_oob[:, 1:-1].
        random_script = [
            Event(color=1),
            Event(color=1),
            Event(color=1, velocity_change=True),
            Event(color=1),
            Event(color=1),
        ]
        # Hand-derived trajectory for random_script (no walls, vc at index 2):
        #   t=0            -> x = X_HOME                      = 244
        #   t=1 free       -> v = -4  (244-4 = 240 <= X_HOME) = 240
        #   t=2 forced     -> v = -4                          = 236
        #   t=3 free (vc)  -> -4 collides with v_prev, next rung -2
        #                                                     = 234
        #   t=4 forced     -> v = -2                          = 232
        #   x        = [244, 240, 236, 234, 232]
        #   seg      = [     -4,  -4,  -2,  -2]
        #   2nd diff = [          0,  +2,   0]  -> nonzero at velocity index 1
        #   oob      = [  F,   F,   F,   F,   F]
        # => velocity_change [F, T, F]; index 1 <-> position index 2 <-> the
        #    declared change at script index 2. No wall => random, not bounce.
        bounce_script = [
            Event(color=1, at_wall=True),
            Event(color=1, velocity_change=True, at_wall=True),
            Event(color=1),
            Event(color=1),
            Event(color=1),
        ]
        # Hand-derived trajectory for bounce_script (walls at 0 and 1, vc at 1):
        #   t=0            -> frame 0 is a wall     -> x = X_WALL0 = 248 (OOB)
        #   t=1 free, wall -> v = +4 (248+4 = 252 > 246)      = 252 (OOB)
        #   t=2 free (vc), no wall -> smallest rung landing <= 244 is -8
        #                                                     = 244
        #   t=3 forced     -> v = -8                          = 236
        #   t=4 forced     -> v = -8                          = 228
        #   x        = [248, 252, 244, 236, 228]
        #   seg      = [     +4,  -8,  -8,  -8]
        #   2nd diff = [        -12,   0,   0]  -> nonzero at velocity index 0
        #   oob      = [  T,   T,   F,   F,   F]
        #   oob[1:-1]= [       T,   F,   F]
        # => velocity_change [T, F, F]; bounce = oob[1:-1] & vc = [T, F, F].
        # NB positions_oob[0] is now True: the pre-fix builder silently dropped
        # a wall declared on frame 0 (x[0] was always mid-box). It is inert for
        # V2 (bounce[:, 0] is forced False) but the builder no longer lies.
        # third row: event-free, one occluded frame (ICO no-grayzone guard only).
        grayzone_script = [
            Event(color=1),
            Event(color=1),
            _gray(),
            Event(color=1),
            Event(color=1),
        ]
        # Hand-derived trajectory for grayzone_script (no events at all):
        #   x        = [244, 240, 236, 232, 228]   (constant v = -4 throughout)
        #   2nd diff = [          0,   0,   0]
        #   oob      = [  F,   F,   F,   F,   F]
        # => velocity_change [F, F, F]; no bounce, no random. Occlusion touches
        #    only the colour channels, never the positions.
        samples = build_samples([random_script, bounce_script, grayzone_script])
        ico = IdealCountingObserver(prog_bar=False)
        ico(samples, return_means=True)

        # row 0: one random change, no walls.
        assert ico.velocity_change[0].tolist() == [False, True, False]
        assert ico.positions_oob[0].tolist() == [False, False, False, False, False]
        assert ico.velocity_change_bounce[0].tolist() == [False, False, False]
        assert ico.velocity_change_random[0].tolist() == [False, True, False]

        # row 1: one bounce, walls on frames 0 and 1.
        assert ico.velocity_change[1].tolist() == [True, False, False]
        assert ico.positions_oob[1].tolist() == [True, True, False, False, False]
        assert ico.velocity_change_bounce[1].tolist() == [True, False, False]
        assert ico.velocity_change_random[1].tolist() == [False, False, False]

        # row 2: event-free; the occluded frame must not perturb the geometry.
        assert ico.velocity_change[2].tolist() == [False, False, False]
        assert ico.positions_oob[2].tolist() == [False, False, False, False, False]
        assert ico.velocity_change_bounce[2].tolist() == [False, False, False]
        assert ico.velocity_change_random[2].tolist() == [False, False, False]

    def test_builder_validation_agreement_script_single_bounce(self):
        # Regression guard for defect B1 (the reason this cell exists). The
        # pre-fix builder rewrote wall samples to WALL_X + 2 from a mid-box
        # start; that ~120-unit relocation was itself a second-difference kink,
        # so _AGREEMENT_SCRIPT — which declares ONE velocity change, at script
        # index 2 — produced velocity_change [T, T, F, F, F] and TWO detected
        # bounces. Exactly one bounce must now be detected.
        #
        # Hand-derived trajectory (walls at 1 and 2, vc at 2):
        #   t=0            -> x = X_HOME                        = 244
        #   t=1 free, wall -> v = +4 (244+4 = 248 > 246)        = 248 (OOB)
        #   t=2 forced     -> v = +4, still outward             = 252 (OOB)
        #   t=3 free (vc), no wall -> smallest rung landing <= 244 is -8
        #                                                       = 244
        #   t=4..6 forced  -> v = -8                     = 236, 228, 220
        #   x        = [244, 248, 252, 244, 236, 228, 220]
        #   seg      = [     +4,  +4,  -8,  -8,  -8,  -8]
        #   2nd diff = [          0, -12,   0,   0,   0]
        #   oob      = [  F,   T,   T,   F,   F,   F,   F]
        #   oob[1:-1]= [       T,   T,   F,   F,   F]
        # => velocity_change [F, T, F, F, F] (index 1 <-> position index 2,
        #    the declared change), bounce = oob[1:-1] & vc = [F, T, F, F, F],
        #    random = ~oob[1:-1] & vc = [F, F, F, F, F].
        samples = build_samples([_AGREEMENT_SCRIPT])
        ico = IdealCountingObserver(prog_bar=False)
        ico(samples, return_means=True)
        assert ico.velocity_change[0].tolist() == [False, True, False, False, False]
        assert ico.positions_oob[0].tolist() == [False, True, True, False, False, False, False]
        assert ico.velocity_change_bounce[0].tolist() == [False, True, False, False, False]
        assert ico.velocity_change_random[0].tolist() == [False, False, False, False, False]


class TestTier2Invariants:
    # row 0 carries an occluded frame so ICO-fed batches have a grayzone.
    # Both rows are INERT to the B1 builder fix. Hand-derived trajectories:
    #   SCRIPTS[0] (vc at 2, no walls): [244, 240, 236, 234]
    #       velocity_change [F, T], positions_oob [F, F, F, F]
    #   SCRIPTS[1] (vc AND wall both at 1): [244, 248, 244, 240]
    #       velocity_change [T, F], positions_oob [F, T, F, F]
    # SCRIPTS[1] escaped the defect pre-fix because the declared change fell on
    # the same index as the relocation, so the phantom kink and the real one
    # coincided; the masks it hands the models are unchanged by the fix.
    SCRIPTS = [
        [Event(color=0), _gray(), Event(color=1, velocity_change=True), Event(color=1)],
        [
            Event(color=2),
            Event(color=2, velocity_change=True, at_wall=True),
            Event(color=0),
            Event(color=0),
        ],
    ]

    def _samples(self):
        return build_samples(self.SCRIPTS)

    @pytest.mark.parametrize(
        "ctor", [IdealBayesianObserver, IdealCountingObserver, IdealCountingObserverV2]
    )
    def test_distribution_valid(self, ctor):
        samples = self._samples()
        # IdealBayesianObserver.__init__ takes THREE required positional args
        # (pccnvc, pccovc, pvc) with no defaults — a bare ctor() raises TypeError,
        # which pytest reports as ERROR, not xfail. Use the golden test's values,
        # matching the pccnvc/pccovc passed to forward() below.
        model = _construct(ctor)
        out = (
            model(samples)
            if ctor is not IdealBayesianObserver
            else model(samples, pccnvc=0.05, pccovc=0.5)
        )
        beliefs = (
            out["beliefs"] if isinstance(out, dict) else (out if torch.is_tensor(out) else out[0])
        )
        # pad mask is derived from the INPUT script, never from model output (A15).
        pad = torch.tensor(
            [[ev.pad for ev in s] + [True] * (beliefs.shape[1] - len(s)) for s in self.SCRIPTS]
        )
        valid = ~pad
        assert (beliefs[valid] >= -1e-6).all()
        assert torch.allclose(beliefs[valid].sum(-1), torch.ones(int(valid.sum())), atol=1e-5)

    def test_transition_matrix_cyclic(self):
        # #2 (C4). V2 exposes T as a callable method taking the rates directly,
        # so its structure is checkable standalone. ICO/IBO inherit T from
        # IdealObserverModel as a @property computed from self.probability_bounce
        # / pvc / pccnvc / pccovc, which forward() populates — so it cannot be
        # exercised without first running a sequence. ICO.T is therefore covered
        # indirectly via the downstream belief-update tests; the direct
        # T-structure check for ICO/IBO is deferred per sibling spec C4.
        v2 = IdealCountingObserverV2()
        for p in (0.0, 0.3, 1.0):
            T = v2.T(
                prob_no_change=torch.tensor([1 - p]), prob_change=torch.tensor([p]), batch_size=1
            )[0]
            assert torch.allclose(T.sum(-1), torch.ones(3), atol=1e-6)  # row-stochastic
            assert torch.allclose(
                torch.diagonal(T), torch.full((3,), 1 - p), atol=1e-6
            )  # i->i mass
            # only i->i or i->(i+1) mod 3 carry mass (no backward / skip)
            forward = torch.tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=T.dtype)
            assert torch.allclose(T, (1 - p) * torch.eye(3) + p * forward, atol=1e-6)

    def test_determinism(self):
        samples = self._samples()
        # _construct, not a bare ctor(): IdealBayesianObserver() would TypeError.
        for model in (
            _construct(IdealBayesianObserver),
            _construct(IdealCountingObserver),
            _construct(IdealCountingObserverV2),
        ):
            a = (
                model(samples.clone(), pccnvc=0.05, pccovc=0.5)
                if isinstance(model, IdealBayesianObserver)
                else model(samples.clone())
            )
            b = (
                model(samples.clone(), pccnvc=0.05, pccovc=0.5)
                if isinstance(model, IdealBayesianObserver)
                else model(samples.clone())
            )
            ta = a if torch.is_tensor(a) else (a["beliefs"] if isinstance(a, dict) else a[0])
            tb = b if torch.is_tensor(b) else (b["beliefs"] if isinstance(b, dict) else b[0])
            assert torch.equal(ta, tb)

    # #4 (C5) padding ignored — green for ICO/V2; IBO excluded (A5). The mate
    # extends the SAME script with a visible, event-free tail, so the base row
    # gains trailing pad frames without changing the batch's grayzone maximum
    # (which would trip the #3 xfail bug instead).
    _PAD_BASE = [Event(color=0), Event(color=0), _gray(), Event(color=1), Event(color=1)]
    _PAD_MATE = _PAD_BASE + [Event(color=1), Event(color=1)]

    def test_padding_ignored_ico(self):
        n = len(self._PAD_BASE)
        ico = IdealCountingObserver(prog_bar=False)
        b_solo, nvc_solo, ovc_solo, pvc_solo = ico(
            build_samples([self._PAD_BASE]), return_means=True
        )
        b_pad, nvc_pad, ovc_pad, pvc_pad = ico(
            build_samples([self._PAD_BASE, self._PAD_MATE]), return_means=True
        )
        assert torch.allclose(b_pad[0, :n], b_solo[0], atol=1e-6)
        assert torch.allclose(nvc_pad[0, :n], nvc_solo[0], atol=1e-6)
        assert torch.allclose(ovc_pad[0, :n], ovc_solo[0], atol=1e-6)
        assert torch.allclose(pvc_pad[0, :n], pvc_solo[0], atol=1e-6)

    def test_padding_ignored_v2(self):
        n = len(self._PAD_BASE)
        out_solo = IdealCountingObserverV2()(build_samples([self._PAD_BASE]), return_means=True)
        out_pad = IdealCountingObserverV2()(
            build_samples([self._PAD_BASE, self._PAD_MATE]), return_means=True
        )
        assert torch.allclose(out_pad["betas"][0], out_solo["betas"][0], atol=1e-6)
        assert torch.allclose(out_pad["beliefs"][0, :n], out_solo["beliefs"][0], atol=1e-6)

    # #6 (C7) colour cyclic equivariance. Same event geometry as
    # _AGREEMENT_SCRIPT, so the B1 builder fix changes its trajectory the same
    # way (two detected bounces -> one). The equivariance VERDICT is unaffected:
    # `_rotated` touches only `color`, leaving velocity_change / positions_oob /
    # bounce bit-identical between orig and rot, and both colour detectors are
    # invariant under a cyclic relabel (ICO's `color_change` is
    # `abs().max(-1) != 0`; V2's `changed` is `(argmax - argmax) % 3 == 1`).
    # `occluded=True` is preserved by `_rotated`, so the grayzone frame is
    # byte-identical too. The underlying RATES do change with the fix; the
    # orig-vs-rot comparison these cells make does not.
    _EQUIV_SCRIPT = [
        Event(color=0),
        Event(color=0, at_wall=True),
        Event(color=1, velocity_change=True, at_wall=True),
        Event(color=1),
        _gray(),
        Event(color=2),
        Event(color=2),
    ]

    @staticmethod
    def _rotated(script):
        return [
            Event(
                velocity_change=ev.velocity_change,
                at_wall=ev.at_wall,
                color=None if ev.color is None else (ev.color + 1) % 3,
                occluded=ev.occluded,
                pad=ev.pad,
            )
            for ev in script
        ]

    def test_equivariance_rates_ico(self):
        # 6a-ICO: rate estimates invariant under a cyclic colour relabel.
        orig = build_samples([self._EQUIV_SCRIPT])
        rot = build_samples([self._rotated(self._EQUIV_SCRIPT)])
        ico = IdealCountingObserver(prog_bar=False)
        _, nvc_o, ovc_o, pvc_o = ico(orig, return_means=True)
        _, nvc_r, ovc_r, pvc_r = ico(rot, return_means=True)
        assert torch.allclose(nvc_r, nvc_o, atol=1e-6)
        assert torch.allclose(ovc_r, ovc_o, atol=1e-6)
        assert torch.allclose(pvc_r, pvc_o, atol=1e-6)

    def test_equivariance_rates_v2(self):
        # 6a-V2. The spec matrix called this "green (vacuous, O2)" — true only
        # while the dead detector masked O3. With E1 applied, the broken
        # visibility test reads the [127,127,127] frame as visible colour 0, so
        # a rotated script sees a spurious forward-cyclic change (2->0) that the
        # original (1->0) does not: rate equivariance breaks until E2 hides the
        # grayzone frame again.
        orig = build_samples([self._EQUIV_SCRIPT])
        rot = build_samples([self._rotated(self._EQUIV_SCRIPT)])
        out_o = IdealCountingObserverV2()(orig.clone(), return_means=True)
        out_r = IdealCountingObserverV2()(rot.clone(), return_means=True)
        assert torch.allclose(out_r["betas"], out_o["betas"], atol=1e-6)

    @pytest.mark.parametrize(
        "ctor",
        [
            IdealBayesianObserver,
            IdealCountingObserver,
            # NB pre-E2 the V2 case failed here (spec A3's flagged "green*"):
            # the grayzone frame was read as VISIBLE with an argmax tie-break to
            # colour 0 — label-asymmetric. E2's occlusion branch (uniform
            # emission on hidden frames) restored equivariance.
            IdealCountingObserverV2,
        ],
    )
    def test_equivariance_beliefs(self, ctor):
        # 6b: beliefs permute by the matching cyclic permutation:
        # beliefs_rot[..., (c+1) % 3] == beliefs_orig[..., c].
        orig = build_samples([self._EQUIV_SCRIPT])
        rot = build_samples([self._rotated(self._EQUIV_SCRIPT)])
        model = _construct(ctor)
        if ctor is IdealBayesianObserver:
            b_o = model(orig, pccnvc=0.05, pccovc=0.5)
            b_r = model(rot, pccnvc=0.05, pccovc=0.5)
        else:
            out_o, out_r = model(orig.clone()), model(rot.clone())
            b_o = out_o["beliefs"] if isinstance(out_o, dict) else out_o
            b_r = out_r["beliefs"] if isinstance(out_r, dict) else out_r
        assert torch.allclose(b_r[:, :, [1, 2, 0]], b_o, atol=1e-5)


class TestBugLedger:
    def test_ico_batch_invariance(self):
        # #3-ICO (C5). A sequence's ICO estimate must not depend on a batch-mate.
        # Non-vacuity verified pre-fix: solo-vs-batch |delta| at [0, n-1] is
        # nonzero on the unmodified model (the Task-4 SHORT/LONG shape).
        n = len(_GRAYZONE_BOUNCE_SCRIPT)
        solo = build_samples([_GRAYZONE_BOUNCE_SCRIPT])
        mate = build_samples([_GRAYZONE_BOUNCE_SCRIPT, _LONGER_GRAYZONE_SCRIPT])
        ico = IdealCountingObserver(prog_bar=False)
        _, m_nvc_solo, m_ovc_solo, _ = ico(solo, return_means=True)
        _, m_nvc_batch, m_ovc_batch, _ = ico(mate, return_means=True)
        assert torch.allclose(m_ovc_solo[0, n - 1], m_ovc_batch[0, n - 1], atol=1e-6)

    def test_v2_rate_recovery(self):
        # #7-V2 (C8). On a fully-visible script with K hand-counted random colour
        # changes in N no-bounce steps, V2's hz channel recovers the Beta mean.
        # Flipped green by E1 alone (the script is fully visible, so E2 never
        # enters); the plan expected the flip at E2 — recorded deviation.
        samples = build_samples([_V2_VISIBLE_RANDOM_SCRIPT])
        out = IdealCountingObserverV2()(samples, return_means=True)
        a_hz, b_hz = out["betas"][0, 0], out["betas"][0, 1]
        assert torch.isclose(
            a_hz / (a_hz + b_hz), torch.tensor(beta_mean(1, 1, _K_CHANGES, _N_STEPS)), atol=1e-4
        )

    def test_v2_dissociation(self):
        # #8-V2 (C9). Bounce-only colour changes must raise cont and leave hz at prior.
        samples = build_samples([_BOUNCE_ONLY_VISIBLE_SCRIPT])
        out = IdealCountingObserverV2()(samples, return_means=True)
        a_cont, b_cont = out["betas"][0, 2], out["betas"][0, 3]
        assert (a_cont / (a_cont + b_cont)) > 0.5 + 1e-3  # moved off the 0.5 prior

    def test_ico_causality(self):
        # #9-ICO (C10). Estimate at t from a script truncated at t equals the
        # full-script estimate at t (non-start-in-grayzone script).
        full = build_samples([_CAUSAL_SCRIPT])
        trunc = build_samples([_CAUSAL_SCRIPT[: _T_CUT + 1]])
        ico = IdealCountingObserver(prog_bar=False)
        _, _, m_ovc_full, _ = ico(full, return_means=True)
        _, _, m_ovc_trunc, _ = ico(trunc, return_means=True)
        assert torch.allclose(m_ovc_full[0, _T_CUT], m_ovc_trunc[0, _T_CUT], atol=1e-6)

    def test_ico_v2_agreement(self):
        # #10 (C11). On a controlled input the two counting observers agree.
        #
        # DISPOSITION (Task 6, 2026-08-28): PROMOTED TO GREEN. The O2 root
        # cause the original xfail cited is fixed (E1), and the agreement held
        # through E2 and E3/E4: both estimators land on 0.5 on this script
        # (ICO m_ovc[-1] 0.5; V2 cont 2/4). NB this is agreement on ONE
        # controlled input, not an equivalence proof — the estimators remain
        # different (Dirichlet 3-way vs per-channel Beta, hard vs expected
        # counts through occlusion), so a future divergence on a richer script
        # would be a finding, not a regression of this cell.
        #
        # ############################################################
        # DISPOSITION SUPERSEDED TWICE.
        #   (a) B1 builder fix: agreement broke, and the cell was pinned at
        #       ICO 0.25 / V2 2/3 while the model-side off-by-one it exposed
        #       awaited a ruling.
        #   (b) ICO-A + ICO-B (this work): the off-by-one is FIXED. Agreement is
        #       STILL NOT restored, so C11 stays UNCOVERED — but the residual
        #       gap now has one named cause (E5, deferred), not an unexplained
        #       misalignment. The values below are hand-derived pending
        #       empirical confirmation.
        # ############################################################
        #
        # WHY the original 0.5/0.5 agreement was an artefact. Pre-B1-fix, the
        # builder's relocation jump made velocity_change = [T, T, F, F, F] — a
        # PHANTOM bounce at velocity index 0 on top of the one declared at index
        # 1 — and ICO paired velocity_change[k] (centred on position index k+1)
        # with color_change[:, 1:][k] (the change ARRIVING at position k+2). The
        # declared bounce-coincident colour change (0 -> 1, arriving at position
        # index 2) therefore sat at color_change[:, 1:][0], where only the
        # phantom could meet it. Removing the phantom left color_change_bounce
        # identically zero; ICO-A removes the +1 skew instead.
        #
        # Hand derivation on the fixed trajectory and the fixed model.
        # x = [244, 248, 252, 244, 236, 228, 220] (see the builder-validation
        # cell above): velocity_change = [F, T, F, F, F], oob[1:-1] =
        # [T, T, F, F, F], bounce = [F, T, F, F, F], random = [F]*5.
        #
        # ICO means_pccovc. colours_inferred = [R, R, G, G, G, B, B] (the
        # grayzone at index 4 forward-fills to G). Post-ICO-A every (B, T-2)
        # mask lives in the event space, index j <-> frame j+1:
        #   color_change              = [F, T, F, F, T, F]
        #   color_change[:, :-1]      = [F, T, F, F, T]   (j=1 <-> frame 2,
        #                                                  j=4 <-> frame 5)
        #   mask_idx_after_grayzone[:, :-1] = [F, F, F, F, T]  (exit frame 5)
        #   color_change_bounce = vc & cc & ~exit          = [0, 1, 0, 0, 0]
        #   color_change_random = ~vc & cc & ~exit         = [0, 0, 0, 0, 0]
        #     (the exit-frame change at j=4 has no in-run vc to explain it —
        #      the run [4, 4] carries none — so it joins the RANDOM channel via
        #      the grayzone path, not the contingent one)
        #   velocity_change_shifted = [F, T, F, F, F] (no grayzone overlap)
        # pccovc pair = (vcs & ~ccb, ccb) = ([0,0,0,0,0], [0,1,0,0,0]); the
        # Dirichlet counts prepend rows 0-1, giving the 7-row sequence
        #   [(0,0), (0,1), (0,0), (0,1), (0,0), (0,0), (0,0)]
        # whose cumulative sum at the last row is (0, 2), so
        #   m_ovc[-1] = (1 + 2) / (1 + 1 + 0 + 2) = 3/4 = 0.75.
        # (The same arithmetic reproduces the two previously documented values —
        # pre-B1 (2, 2) -> 0.5, post-B1 (2, 0) -> 0.25 — which validates it.)
        #
        # WHY THEY STILL DIVERGE, exactly. Both estimators now see the SAME
        # contingent tally on this script: one trial, one success, zero
        # failures. V2 banks it once -> (alpha_cont, beta_cont) = (2, 1) ->
        # 2/3. ICO's get_dist_params runs with offset=2, which prepends a COPY
        # of count rows 0 and 1 before the cumulative sum; this script's only
        # event sits at row 1, so ICO banks it TWICE -> (0, 2) -> 3/4. Drop the
        # duplication and ICO would read (0, 1) -> (1+1)/(1+1+0+1) = 2/3,
        # agreeing exactly. The head duplication is defect E5, ruled an
        # accident and kept inert/deferred (design doc, Q5) — so this cell
        # pins the divergence AND its single remaining cause.
        #
        # V2 cont. bounce[:, 1:-1] = oob[1:-1] & vc = [F, T, F, F, F], so the
        # ONLY bounce frame is t = 2 (pre-fix t = 1 and t = 2 both bounced).
        #   t=1: visible R->R, no bounce  -> beta_hz  1 -> 2
        #   t=2: visible R->G, bounce     -> alpha_cont 1 -> 2
        #   t=3: visible G->G, no bounce  -> beta_hz  2 -> 3
        #   t=4,5: hidden, no bounce      -> hz accumulators only
        #   t=6: visible B->B, no bounce  -> beta_hz
        # cont is touched exactly once => (alpha_cont, beta_cont) = (2, 1) and
        #   cont mean = 2 / 3 ~= 0.6667.
        # (Pre-fix t=1 was a bounce, so beta_cont took the R->R "unchanged"
        # count: (2, 2) -> 0.5, again reproducing the documented value.)
        samples = build_samples([_AGREEMENT_SCRIPT])
        ico = IdealCountingObserver(prog_bar=False)
        _, _, m_ovc, _ = ico(samples, return_means=True)
        out = IdealCountingObserverV2()(samples, return_means=True)
        a_cont, b_cont = out["betas"][0, 2], out["betas"][0, 3]
        assert torch.isclose(m_ovc[0, -1], torch.tensor(0.75), atol=1e-4)
        assert torch.isclose(a_cont / (a_cont + b_cont), torch.tensor(2.0 / 3.0), atol=1e-4)
        # Pin the DIVERGENCE too, so a later change that restores agreement here
        # fails loudly and forces this cell to be re-dispositioned rather than
        # quietly re-acquiring a property nobody re-derived.
        assert not torch.isclose(m_ovc[0, -1], a_cont / (a_cont + b_cont), atol=1e-4)
        # ... and pin the ATTRIBUTED CAUSE, so E5's eventual fix flips this cell
        # rather than silently shifting the gap onto something else.
        # `counts_pccovc` is the (B, T, 2) CUMULATIVE sum of
        #   [row_0, row_1, row_0, row_1, ..., row_{T-3}]
        # so its slot at index 1 is exactly the duplicated head (row_0 + row_1)
        # and `counts[-1] - counts[1]` is the honest, once-counted tally.
        # Subtract the duplication and ICO reproduces V2's contingent mean.
        counts = ico.counts_pccovc[0]
        assert counts[-1].tolist() == [0.0, 2.0]  # one event, banked twice
        failures, successes = (counts[-1] - counts[1]).tolist()
        assert (failures, successes) == (0.0, 1.0)
        unduplicated = (1 + successes) / (1 + 1 + failures + successes)
        assert torch.isclose(torch.tensor(unduplicated), a_cont / (a_cont + b_cont), atol=1e-4)


class TestV2OcclusionAndMutation:
    def test_v2_grayzone_classified_occluded(self):
        # v2_grayzone (NEW; design E2 §). A [127,127,127] frame must run V2's
        # occlusion branch, not be read as a visible nongray_transition.
        #
        # The proxy asserts the occlusion branch's POSITIVE SIGNATURE, not its
        # absence. A grayzone is NOT count-inert for V2 (design E2 §): both
        # transitions of [colour_A, grayzone, colour_A] are hidden, so V2
        # accumulates EXPECTED counts (exp_change = mean_hz = 0.5 at the (1,1)
        # prior, twice) and flushes them at the exit frame, giving
        # alpha_hz = beta_hz = 1 + 2*0.5 = 2. No bounce, so cont stays at prior.
        #   broken (E2 unfixed, E1 either way) -> [1, 1, 1, 1]: the grayzone is
        #     read as a visible transition that is neither `changed` nor
        #     `unchanged`, so it contributes nothing and the betas never move.
        #   after E2                            -> [2, 2, 1, 1].
        # NB do NOT assert "betas stay at the (1,1,1,1) prior": that is the
        # BROKEN signature, so it would XPASS under strict and crash the suite.
        samples = build_samples([_BENIGN_GRAYZONE_SCRIPT])
        out = IdealCountingObserverV2()(samples, return_means=True)
        assert torch.allclose(out["betas"][0], torch.tensor([2.0, 2.0, 1.0, 1.0]), atol=1e-5)

    def test_v2_no_mutation(self):
        # v2_no_mutation (NEW; design E2 §). forward() must not modify the caller's
        # samples tensor (the split() view must not be written back).
        samples = build_samples([_BENIGN_GRAYZONE_SCRIPT])
        before = samples.clone()
        IdealCountingObserverV2()(samples, return_means=True)
        assert torch.equal(samples, before)
