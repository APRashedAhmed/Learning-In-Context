"""Tier 1+2 property (specification) suite for the ideal-observer models.

Cells #1-#10 follow the sibling test-suite spec (C1-C12, status matrix); the two
`v2_*` cells are new E2 cells added by the model-fixes work-unit. Bug-cells are
`xfail(strict, raises=AssertionError)` — the living bug-ledger: fixing a bug
flips its cell to a loud XPASS, signalling the marker's removal.

NB the current IdealCountingObserver crashes (`ValueError: max() arg is an empty
sequence`) on any batch with ZERO grayzone frames (Tier-3, deferred), so every
ICO-fed batch below carries at least one occluded frame in some row.
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
        # The builder, given the equivalent Event scripts, reproduces the hand tables.
        random_script = [
            Event(color=1),
            Event(color=1),
            Event(color=1, velocity_change=True),
            Event(color=1),
            Event(color=1),
        ]
        bounce_script = [
            Event(color=1, at_wall=True),
            Event(color=1, velocity_change=True, at_wall=True),
            Event(color=1),
            Event(color=1),
            Event(color=1),
        ]
        # third row: event-free, one occluded frame (ICO no-grayzone guard only).
        grayzone_script = [
            Event(color=1),
            Event(color=1),
            _gray(),
            Event(color=1),
            Event(color=1),
        ]
        samples = build_samples([random_script, bounce_script, grayzone_script])
        ico = IdealCountingObserver(prog_bar=False)
        ico(samples, return_means=True)
        # row 0: random change; row 1: bounce — declared vs detected (1:-1 alignment).
        assert ico.velocity_change_random[0].any() and not ico.velocity_change_bounce[0].any()
        assert ico.velocity_change_bounce[1].any()


class TestTier2Invariants:
    # row 0 carries an occluded frame so ICO-fed batches have a grayzone.
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

    # #6 (C7) colour cyclic equivariance.
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
    @pytest.mark.xfail(
        strict=True, raises=AssertionError, reason="O6: ICO max_grayzone_diff is batch-global"
    )
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

    @pytest.mark.xfail(
        strict=True, raises=AssertionError, reason="O1: ICO max_grayzone_diff uses future structure"
    )
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
        # Provisionally green: E1 alone brought both estimates to 0.5 on this
        # script (ICO 0.5, V2 cont 2/4) — the O2 root cause the xfail cited is
        # gone. Final disposition (green vs xfail-with-new-reason) is Task 6's,
        # after E2 and E3/E4 land; E2 in particular reshapes V2's grayzone
        # counts on this script and may re-open the divergence.
        samples = build_samples([_AGREEMENT_SCRIPT])
        ico = IdealCountingObserver(prog_bar=False)
        _, _, m_ovc, _ = ico(samples, return_means=True)
        out = IdealCountingObserverV2()(samples, return_means=True)
        a_cont, b_cont = out["betas"][0, 2], out["betas"][0, 3]
        assert torch.isclose(m_ovc[0, -1], a_cont / (a_cont + b_cont), atol=1e-4)


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
