"""Characterization tests for the migrated ideal observer models.

Pins the *current* public API and numeric behavior of the four observer classes
in :mod:`learning_in_context.models.ideal_observer` (migrated from hmdcpd
``iom.py`` at commit ``973624e``).

These are golden-output / characterization tests: they freeze present behavior to
catch regressions during the remaining migration work. They deliberately do **not**
assert numerical parity against the pre-migration hmdcpd implementation. The
contract named in the migration handoff — ``get_ico_outputs`` calling the ICO with
``counts=`` and ``vel_change_offset=0`` — targets an *older*, since-rewritten ICO
``forward`` signature (visible in stale hmdcpd tracebacks). The live signature, used
by ``iom.py``'s own ``__main__`` block, is
``forward(samples, beliefs=None, scale=255, return_means=False, prior_pvc=None,
prior_pccovc=None, prior_pccnvc=None)`` — which is what is pinned here.

Input fixture (deterministic, hand-built — see module constants):
    seq0 bounces off the right wall (x=248 > 246) with a coincident color change,
         so the bounce-contingent color channel (pccovc) is genuinely exercised,
         then is occluded by a grayzone.
    seq1 bounces off the top wall (y=8 < 10) and ends with a padded tail, so the
         padding-handling paths are exercised.
"""

import pytest
import torch
from bouncing_ball_task.defaults import TaskParameters

from learning_in_context.models import (
    IdealBayesianObserver,
    IdealCountingObserver,
    IdealCountingObserverV2,
    IdealObserverModel,
)

ATOL = 1e-4

# Colors are one-hot * 255; grayzone == mask_color (127); padding == -1.
_R = [255.0, 0.0, 0.0]
_G = [0.0, 255.0, 0.0]
_B = [0.0, 0.0, 255.0]
_GRAY = [127.0, 127.0, 127.0]
_PAD = [-1.0, -1.0, -1.0]

# Deterministic positions (B=2, T=14), hardcoded so the fixture needs no simulator.
# fmt: off
_POS0 = [[232.0, 120.0], [240.0, 120.0], [248.0, 120.0], [240.0, 120.0], [232.0, 120.0],
         [224.0, 120.0], [216.0, 120.0], [208.0, 120.0], [200.0, 120.0], [192.0, 120.0],
         [184.0, 120.0], [176.0, 120.0], [168.0, 120.0], [160.0, 120.0]]
_POS1 = [[120.0, 24.0], [120.0, 16.0], [120.0, 8.0], [120.0, 16.0], [120.0, 24.0],
         [120.0, 32.0], [120.0, 40.0], [120.0, 48.0], [120.0, 56.0], [120.0, 64.0],
         [120.0, 72.0], [120.0, 80.0], [120.0, 88.0], [120.0, 96.0]]
_COL0 = [_R, _R, _G, _G, _GRAY, _GRAY, _GRAY, _B, _B, _B, _B, _B, _B, _B]
_COL1 = [_B, _B, _R, _GRAY, _GRAY, _R, _R, _R, _G, _G, _G, _PAD, _PAD, _PAD]
# fmt: on


@pytest.fixture
def samples():
    """A fresh (2, 14, 5) samples tensor: [x, y, r, g, b] per timestep.

    Function-scoped for test isolation (V2's in-place color scaling, which
    once made this mandatory, was removed by the E2 fix).
    """
    positions = torch.tensor([_POS0, _POS1], dtype=torch.float)
    colors = torch.tensor([_COL0, _COL1], dtype=torch.float)
    return torch.cat([positions, colors], dim=-1)


class TestIdealObserverModel:
    """Base module: task geometry + the cyclic color-transition matrix."""

    def test_default_geometry(self):
        model = IdealObserverModel()
        assert torch.equal(model.mask_color, torch.tensor([127.0, 127.0, 127.0], dtype=float))
        assert torch.equal(model.size_frame, torch.tensor([256, 256]))
        assert model.ball_radius == 10
        assert model.dt == 0.1
        assert model.padding_value == -1

    def test_custom_task_parameters(self):
        model = IdealObserverModel(TaskParameters(ball_radius=5, dt=0.2))
        assert model.ball_radius == 5
        assert model.dt == 0.2

    def test_base_forward_is_noop(self):
        # The base class is abstract: forward/init_states do nothing and return None.
        assert IdealObserverModel().forward(object()) is None


class TestIdealBayesianObserver:
    """Oracle observer: hazard rates are given, not estimated."""

    def test_output_shape(self, samples):
        ibo = IdealBayesianObserver(0.01, 0.8, 0.05)
        pred = ibo(samples, pccnvc=0.01, pccovc=0.8)
        assert pred.shape == (2, 14, 3)

    def test_predictions_are_distributions_for_visible_sequence(self, samples):
        # seq0 is fully visible/occluded (never padded), so every predictive row
        # is a valid categorical distribution summing to 1.
        ibo = IdealBayesianObserver(0.01, 0.8, 0.05)
        pred = ibo(samples, pccnvc=0.01, pccovc=0.8)
        assert torch.allclose(pred[0].sum(-1), torch.ones(14), atol=ATOL)

    def test_golden_final_prediction(self, samples):
        ibo = IdealBayesianObserver(0.01, 0.8, 0.05)
        pred = ibo(samples, pccnvc=0.01, pccovc=0.8)
        # seq0: belief settled on blue, propagated one step under the transition.
        # seq1: padded final frame -> the observer emits zeros (characterized quirk:
        # the IBO handles grayzone masking but not padding).
        expected = torch.tensor([[0.0495, 0.0, 0.9505], [0.0, 0.0, 0.0]])
        assert torch.allclose(pred[:, -1], expected, atol=ATOL)


class TestIdealCountingObserver:
    """Counting observer: estimates pvc/pccnvc/pccovc from counts, then filters."""

    def test_return_means_tuple_structure(self, samples):
        ico = IdealCountingObserver(prog_bar=False)
        out = ico(samples, return_means=True)
        assert isinstance(out, tuple) and len(out) == 4
        beliefs, m_nvc, m_ovc, m_pvc = out
        assert beliefs.shape == (2, 14, 3)
        assert m_nvc.shape == m_ovc.shape == m_pvc.shape == (2, 14)

    def test_returns_bare_beliefs_without_means(self, samples):
        ico = IdealCountingObserver(prog_bar=False)
        out = ico(samples, return_means=False)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (2, 14, 3)

    def test_beliefs_are_distributions(self, samples):
        ico = IdealCountingObserver(prog_bar=False)
        beliefs = ico(samples, return_means=False)
        assert torch.allclose(beliefs.sum(-1), torch.ones(2, 14), atol=ATOL)

    def test_estimated_rates_are_probabilities(self, samples):
        ico = IdealCountingObserver(prog_bar=False)
        _, m_nvc, m_ovc, m_pvc = ico(samples, return_means=True)
        for m in (m_nvc, m_ovc, m_pvc):
            assert (m >= 0).all() and (m <= 1).all()

    def test_bounce_contingent_channel_is_exercised(self, samples):
        # seq0 has an OOB bounce with a coincident color change, so its pccovc
        # estimate must move off the (1, 1) prior mean of 0.5.
        # ICO-A: the channel is now exercised through its SUCCESS count. Before
        # the fix `color_change_bounce` was all-zero and the estimate moved only
        # via the failure count (the pairing was skewed by one --
        # `velocity_change[j]` is frame j+1 while `color_change[:, 1:][j]` was
        # frame j+2, so a genuinely coincident bounce+colour change could never
        # land in the same slot). The estimate therefore moves the OTHER WAY
        # now: 0.25 (pure failure) -> 0.75 (pure success).
        ico = IdealCountingObserver(prog_bar=False)
        _, _, m_ovc, _ = ico(samples, return_means=True)
        assert not torch.isclose(m_ovc[0, -1], torch.tensor(0.5), atol=ATOL)
        assert m_ovc[0, -1] > 0.5

    def test_golden_final_outputs(self, samples):
        ico = IdealCountingObserver(prog_bar=False)
        beliefs, m_nvc, m_ovc, m_pvc = ico(samples, return_means=True)
        # ICO-A (bounce/colour pairing skew) re-baselines BOTH rows. HAND-DERIVED
        # (empirically confirmed 2026-08-31, 47/47). Common index convention: every
        # (B, T-2) event mask now lives in the event space, index j <-> frame j+1.
        #
        # seq0: x = [232, 240, 248, 240, ...] -> oob at frame 2 only,
        #   velocity_change = [F, T, F, ...] (j=1 <-> frame 2) and that vc is a
        #   BOUNCE. colours [R, R, G, G, GRAY x3, B x7] forward-fill to
        #   [R, R, G, G, G, G, G, B, ...], so color_change (j <-> frame j+1) is
        #   True at j=1 (frame 2) and j=6 (frame 7, the grayzone exit).
        #     color_change_bounce = vc & cc & ~exit -> {j=1}   (was EMPTY: the
        #       frame-2 change used to be read one cell early, at old-space j=0,
        #       and misfiled as a random-channel success -- and j=0 is one of the
        #       two rows the head duplication copies, so it supplied 2 of the 3
        #       successes behind the old 4/17)
        #     color_change_random = {j=6} only, via the exit path (the run
        #       [4, 6] carries no vc, so the exit change stays random)
        #     velocity_change_shifted = {j=1}
        #   pccnvc pair (~vcs, ccr): the offset=2 head duplication prepends rows
        #     j=0, j=1, so counts = (1 + 11, 0 + 1) = (12, 1) -> 2/15
        #     (was (12, 3) -> 4/17).
        #   pccovc pair (vcs & ~ccb, ccb): j=1 is now a SUCCESS, so slot 0 is
        #     empty and the duplicated row j=1 counts it twice:
        #     (0, 2) -> 3/4 (was (2, 0) -> 1/4).
        #   pvc pair is built from the UNSHIFTED detectors -> unchanged, 1/14.
        #   beliefs[0, -1]: frame 13 is visible blue -> [0, 0, 1], propagated one
        #     step. probability_bounce = 0, so vc = pvc = 1/14 and
        #     p_transition = (2/15)(13/14) + (3/4)(1/14) = 149/840 = 0.177381.
        #
        # seq1: bounce at frame 2 (y = 8 < 10), colours [B, B, R, GRAY, GRAY,
        #   R, R, R, G, G, G, PAD x3] -> inferred [B, B, R, R, R, R, R, R, G, G,
        #   G, PAD x3]; color_change True at j=1 (frame 2, the bounce), j=7
        #   (frame 8) and j=10 (frame 11, the pre-existing G -> PAD artefact).
        #   The grayzone run [3, 4] exits at frame 5 with NO colour change, and
        #   carries no vc, so the exit path stays inert (DRIFT-1's verdict for
        #   this row is unchanged: velocity_change_shifted = {j=1}).
        #     color_change_bounce = {j=1}; color_change_random = {j=7, j=10}.
        #   pccnvc: (1 + 11, 0 + 2) = (12, 2) -> 3/16 (was (12, 4) -> 5/18; the
        #     frame-2 change left the random channel and the head duplication no
        #     longer doubles a success at j=0).
        #   pccovc: (0, 2) -> 3/4 (was (2, 0) -> 1/4), same arithmetic as seq0.
        #   beliefs[1, -1] is a PAD frame, never written in the loop, so it stays
        #     at the 1/3 init regardless.
        assert torch.allclose(
            beliefs[:, -1],
            torch.tensor([[149 / 840, 0.0, 691 / 840], [1 / 3, 1 / 3, 1 / 3]]),
            atol=ATOL,
        )
        assert torch.allclose(m_nvc[:, -1], torch.tensor([2 / 15, 3 / 16]), atol=ATOL)
        assert torch.allclose(m_ovc[:, -1], torch.tensor([0.75, 0.75]), atol=ATOL)
        assert torch.allclose(m_pvc[:, -1], torch.tensor([1 / 14, 1 / 14]), atol=ATOL)

    def test_deterministic(self, samples):
        ico = IdealCountingObserver(prog_bar=False)
        a = ico(samples, return_means=False)
        b = ico(samples, return_means=False)
        assert torch.equal(a, b)

    def test_visible_bounce_pairs_with_colour_change_at_the_bounce_frame(self):
        # ICO-A. The task's convention is that a bounce-contingent colour change
        # happens AT the bounce frame. Pins that pairing on EXACT boolean arrays
        # (no Dirichlet arithmetic), on a fully VISIBLE bounce so no grayzone
        # attribution is involved.
        #
        # row 0 -- bounce at frame 2 with the colour change at frame 2.
        #   x = [232, 240, 248, 240, 232, 224]
        #     v      = [ 8, 8, -8, -8, -8]
        #     v2diff = [ 0, -16,  0,  0]  -> velocity_change[1] (frame 2)
        #     oob    = [F, F, T, F, F, F] (248 > 246) -> the vc is a BOUNCE
        #   colours [R, R, G, G, G, G] -> color_change at frame 2, i.e. index
        #   j = 1 in the event space (index j <-> frame j+1) that ICO-A puts the
        #   colour arrays in.
        #     broken -> color_change_bounce[0] == [0, 0, 0, 0]  (the change was
        #               read at color_change[:, 1:][0], one frame LATE) and
        #               color_change_random[0]  == [1, 0, 0, 0]  (misfiled into
        #               the random channel, one cell early)
        #     fixed  -> color_change_bounce[0] == [0, 1, 0, 0], random all zero.
        #
        # row 1 -- inert companion carrying the grayzone frame the ICO's
        #   attribution path expects. Straight line (x = 100..120, no oob, no
        #   vc) and the occluded frame forward-fills to the same colour, so it
        #   declares no colour change at all and cannot perturb row 0 (every
        #   mask below is computed per sequence).
        # fmt: off
        pos_0 = [[232.0, 128.0], [240.0, 128.0], [248.0, 128.0],
                 [240.0, 128.0], [232.0, 128.0], [224.0, 128.0]]
        pos_1 = [[100.0, 128.0], [104.0, 128.0], [108.0, 128.0],
                 [112.0, 128.0], [116.0, 128.0], [120.0, 128.0]]
        col_0 = [_R, _R, _G, _G, _G, _G]
        col_1 = [_R, _R, _GRAY, _R, _R, _R]
        # fmt: on
        samples = torch.cat(
            [
                torch.tensor([pos_0, pos_1], dtype=torch.float),
                torch.tensor([col_0, col_1], dtype=torch.float),
            ],
            dim=-1,
        )
        ico = IdealCountingObserver(prog_bar=False)
        ico(samples, return_means=True)

        # the detector itself (unchanged by ICO-A): index j <-> frame j+1
        assert ico.velocity_change_bounce[0].tolist() == [False, True, False, False]
        assert ico.velocity_change_random[0].tolist() == [False] * 4

        # the pairing: bounce channel, NOT random.
        assert ico.color_change_bounce[0].tolist() == [0.0, 1.0, 0.0, 0.0]
        assert ico.color_change_random[0].tolist() == [0.0] * 4

        # the companion declares nothing in either channel.
        assert ico.color_change_bounce[1].tolist() == [0.0] * 4
        assert ico.color_change_random[1].tolist() == [0.0] * 4

    def test_attribution_window_is_spec_closed_interval(self):
        # DRIFT-1. Pins the SPEC's [s, e] attribution window on EXACT boolean
        # arrays (no Dirichlet arithmetic, no offset=2 head duplication), on the
        # two cases where [s, e] and the old [s-1, e-1] disagree.
        #
        # row A -- LENGTH-1 grayzone run at frame 4 carrying a bounce vc at
        #   frame 4, with a colour change across the run (R -> G at the exit).
        #   x = [236, 240, 244, 248, 252, 248, 244, 240]
        #     v      = [ 4, 4, 4, 4, -4, -4, -4]
        #     v2diff = [ 0, 0, 0, -8,  0,  0]  -> velocity_change[3] (frame 4)
        #     oob    = frames 3, 4, 5 (x > 246) -> the vc at frame 4 is a BOUNCE
        #   NB the oob span is deliberately 3 frames wide (the flip has to happen
        #   OUTSIDE the box for the vc frame to be oob); do not "tidy" these
        #   positions without re-deriving which frames are bounces.
        #   s = e = 4, exit 5. New window [4, 4] contains it; old window [3, 3]
        #   does not, so the exit-frame colour change used to be misfiled as a
        #   RANDOM colour change (colour changed with no attributable vc).
        #     broken -> ico.color_change_random[0] == [0, 0, 0, 1, 0, 0]
        #     fixed  -> all zeros (the change is attributed to the run's bounce).
        #   ICO-B additionally lands it in the CONTINGENT channel rather than
        #   nowhere: color_change_bounce[0] == [0, 0, 0, 0, 1, 0] (cell j = 4
        #   <-> the exit frame 5, in the post-ICO-A event space).
        #
        # row B -- run [4, 5] with a random vc on frame 3 (the last VISIBLE frame
        #   before entry) AND a random vc on frame 4 (inside the run).
        #   x = [100, 104, 108, 112, 108, 112, 116, 120]
        #     v      = [ 4, 4, 4, -4, 4, 4, 4]
        #     v2diff = [ 0, 0, -8, 8, 0, 0]  -> velocity_change[2], [3]
        #                                       (frames 3 and 4); never oob
        #   Only frame 4 is inside [s, e] = [4, 5]. Frame 3's vc must keep its
        #   own index (j = 2), while the run's aggregate lands on the exit cell.
        #   The old code registered frame 3's vc as the run's cause AND stripped
        #   it from its own index:
        #     broken       -> velocity_change_random_shifted[1] == [F,F,F,F,T,F]
        #     DRIFT-1 only -> [F, F, T, F, T, F]  (exit cell j = e-1 = 4, in the
        #                     pre-ICO-A frame-(j+2) colour space)
        #     + ICO-A      -> [F, F, T, F, F, T]  (exit cell j = e = 5 <-> the
        #                     exit frame 6, now that the colour arrays share the
        #                     velocity index space). The WINDOW verdict -- frame
        #                     3 in, frame 4 the cause -- is identical; only the
        #                     index space of the plant moved.
        #   ICO-B: the exit colour change is explained by the run's RANDOM vc, so
        #   it joins the contingent channel too (see the model-side note on the
        #   design doc's literal "random channel" wording):
        #     color_change_bounce[1] == [0, 0, 0, 0, 0, 1]
        # fmt: off
        pos_a = [[236.0, 128.0], [240.0, 128.0], [244.0, 128.0], [248.0, 128.0],
                 [252.0, 128.0], [248.0, 128.0], [244.0, 128.0], [240.0, 128.0]]
        pos_b = [[100.0, 128.0], [104.0, 128.0], [108.0, 128.0], [112.0, 128.0],
                 [108.0, 128.0], [112.0, 128.0], [116.0, 128.0], [120.0, 128.0]]
        col_a = [_R, _R, _R, _R, _GRAY, _G, _G, _G]
        col_b = [_R, _R, _R, _R, _GRAY, _GRAY, _G, _G]
        # fmt: on
        samples = torch.cat(
            [
                torch.tensor([pos_a, pos_b], dtype=torch.float),
                torch.tensor([col_a, col_b], dtype=torch.float),
            ],
            dim=-1,
        )
        ico = IdealCountingObserver(prog_bar=False)
        ico(samples, return_means=True)

        # detectors themselves are untouched by DRIFT-1 (alignment: index j <-> frame j+1)
        assert ico.velocity_change_bounce[0].tolist() == [False, False, False, True, False, False]
        assert ico.velocity_change_random[1].tolist() == [False, False, True, True, False, False]

        # row A: the run's own bounce now explains the exit-frame colour change
        assert ico.color_change_random[0].tolist() == [0.0] * 6
        # ... and a length-1 run leaves the aggregate on the exit cell (j = e = 4)
        expected_a = [False, False, False, False, True, False]
        assert ico.velocity_change_bounce_shifted[0].tolist() == expected_a
        # ICO-B: and the change is BANKED there instead of vanishing.
        assert ico.color_change_bounce[0].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]

        # row B: frame 3 (last visible pre-entry) keeps its own index j = 2;
        # the run's aggregate lands on the exit cell j = 5.
        expected_b = [False, False, True, False, False, True]
        assert ico.velocity_change_random_shifted[1].tolist() == expected_b
        assert ico.color_change_random[1].tolist() == [0.0] * 6
        assert ico.color_change_bounce[1].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


class TestIdealCountingObserverV2:
    """Re-architected counting observer: online Beta updating, dict output."""

    def test_output_dict_structure(self, samples):
        out = IdealCountingObserverV2()(samples, return_means=True)
        assert set(out) == {"beliefs", "betas", "p_change"}
        assert out["beliefs"].shape == (2, 14, 3)
        assert out["betas"].shape == (2, 4)
        assert out["p_change"].shape == (2, 14)

    def test_p_change_omitted_without_means(self, samples):
        out = IdealCountingObserverV2()(samples, return_means=False)
        assert set(out) == {"beliefs", "betas"}

    def test_independent_geometry_wiring(self):
        # V2 does not inherit IdealObserverModel; its geometry load is a separate
        # code path, so verify it honors custom task parameters too.
        model = IdealCountingObserverV2(TaskParameters(ball_radius=5))
        assert model.ball_radius == 5

    def test_betas_never_below_prior(self, samples):
        # Beta pseudo-counts start at the (1,1,1,1) prior and only accumulate.
        out = IdealCountingObserverV2()(samples, return_means=True)
        assert (out["betas"] >= 1.0).all()

    def test_p_change_is_probability(self, samples):
        out = IdealCountingObserverV2()(samples, return_means=True)
        assert (out["p_change"] >= 0).all() and (out["p_change"] <= 1).all()

    def test_forward_does_not_mutate_input_colors(self, samples):
        # E2 fix: forward no longer writes back through the split() view, so the
        # caller's tensor is left byte-for-byte unchanged.
        before = samples.clone()
        IdealCountingObserverV2()(samples, return_means=False)
        assert torch.equal(samples, before)

    def test_golden_final_outputs(self, samples):
        out = IdealCountingObserverV2()(samples, return_means=True)
        assert torch.allclose(
            out["betas"], torch.tensor([[2.0, 12.0, 2.0, 1.0], [3.0, 8.0, 2.0, 1.0]]), atol=ATOL
        )
        # seq0's final frame is VISIBLE (blue), so its belief is the one-hot of the
        # observation regardless of the recursion -- unchanged by the W8 fix.
        # seq1's final frame is PADDED. Padded rows are still propagated through T
        # every step (a characterization quirk that predates W8). Pre-fix every step
        # read the uniform init, so every non-visible slot -- grayzone and padded
        # alike -- came out uniform; post-W8 the last visible belief is carried
        # forward.
        # HAND-DERIVED (empirically confirmed 2026-08-31, 47/47): seq1's counts freeze at
        # alpha_hz=3, beta_hz=8 after t=10, so p_change = 3/11 for t=11..13 and the
        # belief evolves [0,1,0] -> [0,8,3]/11 -> [9,64,48]/121 -> [216,539,576]/1331.
        assert torch.allclose(
            out["beliefs"][:, -1],
            torch.tensor([[0.0, 0.0, 1.0], [216 / 1331, 539 / 1331, 576 / 1331]]),
            atol=ATOL,
        )
        assert torch.allclose(out["p_change"][:, -1], torch.tensor([0.153846, 0.0]), atol=ATOL)

    def test_opening_grayzone_contributes_nothing(self):
        # DEFECT-1. A grayzone run that OPENS the sequence has no observed entry
        # colour, so it must contribute nothing to the change/no-change counts.
        # Straight-line positions (no oob, constant velocity) => no bounces, so
        # only the hz channel can move and cont stays at the (1, 1) prior.
        #
        # HAND-DERIVED (empirically confirmed 2026-08-31, 47/47), colours
        # [GRAY, R, R, GRAY, R, R]:
        #   t=1  hidden (gray -> R), run is ANCHORLESS -> no accumulation.
        #        belief seeded from the exit colour by the visible-frame emission
        #        path: uniform prior * one-hot(R) -> [1, 0, 0].
        #   t=2  visible R -> R, unchanged, no bounce -> beta_hz = 2.
        #   t=3  hidden (R -> gray), ANCHORED. mean_hz = 1/3, belief [1, 0, 0]
        #        -> acc = (1/3, 2/3).
        #   t=4  hidden (gray -> R), still mean_hz = 1/3 (the flush lands after
        #        the accumulation) -> acc = (2/3, 4/3); exit flush ->
        #        alpha_hz = 1 + 2/3 = 5/3, beta_hz = 2 + 4/3 = 10/3.
        #   t=5  visible R -> R, unchanged -> beta_hz = 13/3.
        # Broken (accumulating through the opening run) adds exactly the orphaned
        # (0.5, 0.5) from t=1: [13/6, 29/6, 1, 1].
        positions = [[100.0 + 4 * t, 128.0] for t in range(6)]
        colors = [_GRAY, _R, _R, _GRAY, _R, _R]
        samples = torch.cat(
            [
                torch.tensor([positions], dtype=torch.float),
                torch.tensor([colors], dtype=torch.float),
            ],
            dim=-1,
        )
        out = IdealCountingObserverV2()(samples, return_means=True)
        assert torch.allclose(out["betas"][0], torch.tensor([5 / 3, 13 / 3, 1.0, 1.0]), atol=ATOL)

    def test_opening_grayzone_betas_are_batch_invariant(self):
        # DEFECT-1, the orphaned-flush half. On [GRAY, R, R] the exit guard
        # `visible_now & inside_gray` cannot fire at t=1 (`inside_gray` starts
        # False), so pre-fix the t=0->1 expected counts were neither flushed nor
        # zeroed. Solo they simply leaked away; in a padded batch the pad step
        # drives `inside_gray` True, and the end-of-sequence flush then banked
        # them -- so the SAME sequence produced different betas depending on its
        # batch-mates (broken: solo [1, 2, 1, 1] vs padded [1.5, 2.5, 1, 1]).
        # The equality is the discriminating assertion; the [1, 2, 1, 1] value is
        # shared with the broken solo path and is pinned only as a sanity anchor
        # (only the visible R -> R step at t=2 may count, and it is `unchanged`).
        pos_solo = [[100.0, 128.0], [104.0, 128.0], [108.0, 128.0]]
        col_solo = [_GRAY, _R, _R]
        solo = torch.cat(
            [
                torch.tensor([pos_solo], dtype=torch.float),
                torch.tensor([col_solo], dtype=torch.float),
            ],
            dim=-1,
        )
        # a strictly longer, fully visible mate: row 0 gains one padded frame.
        pos_mate = pos_solo + [[112.0, 128.0]]
        col_mate = [_R, _R, _R, _R]
        padded = torch.cat(
            [
                torch.tensor([pos_solo + [_PAD[:2]], pos_mate], dtype=torch.float),
                torch.tensor([col_solo + [_PAD], col_mate], dtype=torch.float),
            ],
            dim=-1,
        )
        betas_solo = IdealCountingObserverV2()(solo, return_means=True)["betas"][0]
        betas_padded = IdealCountingObserverV2()(padded, return_means=True)["betas"][0]
        assert torch.allclose(betas_padded, betas_solo, atol=ATOL)
        assert torch.allclose(betas_solo, torch.tensor([1.0, 2.0, 1.0, 1.0]), atol=ATOL)
