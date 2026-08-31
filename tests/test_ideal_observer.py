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

    def test_coincident_wall_and_random_change_keeps_beliefs_in_range(self):
        # Overflow fix. `probability_bounce` is now the UNION of the two
        # velocity-event flags, `a + b - a*b`; it used to be the bare sum
        # `positions_oob[:, t] + velocity_change_random[:, max(0, t - 2)]`.
        #
        # NON-VACUITY, measured 2026-08-31. At HEAD the bare sum could not
        # actually reach 2: both operands are BOOL tensors and torch's bool `+`
        # saturates as logical OR, so the pre-fix expression already WAS the
        # union. The live mutant this cell kills is the natural refactor that
        # casts before adding (`oob_t + vcr_t`, floats): verified against a
        # patched copy of the module, it drives `probability_bounce` to 2 at the
        # overflow frame and beliefs[0, 6] to [1.1167, 0, -0.1167].
        #
        # Fixture, built so the mutant genuinely ESCAPES [0, 1] (a low pccovc is
        # not enough -- p_transition = pccnvc*(pvc - 1) + pccovc*(2 - pvc) only
        # exceeds 1 once pccovc is well above 0.5, so BOTH velocity changes are
        # given a coincident colour change):
        #   x = [240, 248, 240, 232, 224, 216, 252, 288]
        #     v      = [ 8, -8, -8, -8, -8, 36, 36]
        #     v2diff = [-16,  0,  0,  0, 44,  0]  -> vc at frames 1 and 5
        #     oob    = [F, T, F, F, F, F, T, T]   (>246), oob[1:-1] = [T,F,F,F,F,T]
        #   => velocity_change_bounce = [T, F, F, F, F, F] (frame 1, at the wall)
        #      velocity_change_random = [F, F, F, F, T, F] (frame 5, mid-box)
        #   colours [R, G, G, GRAY, G, B, B, B] -> a change at frame 1 and at
        #   frame 5, each coincident with its velocity change, so both land in
        #   color_change_bounce and pccovc[6] = (1 + 2)/(2 + 0 + 2) = 0.75.
        #   The occluded frame 3 follows this suite's convention of giving
        #   ICO-fed batches a grayzone; it is inert here (its run exits with no
        #   colour change), and the ICO no longer requires one either way.
        #
        # THE OVERFLOW FRAME is t = 6: positions_oob[6] is True (288 > 246) AND
        # velocity_change_random[max(0, 6 - 2)] = velocity_change_random[4] is
        # True (frame 5). Union -> 1; bare float sum -> 2.
        # fmt: off
        pos = [[240.0, 128.0], [248.0, 128.0], [240.0, 128.0], [232.0, 128.0],
               [224.0, 128.0], [216.0, 128.0], [252.0, 128.0], [288.0, 128.0]]
        col = [_R, _G, _G, _GRAY, _G, _B, _B, _B]
        # fmt: on
        samples = torch.cat(
            [
                torch.tensor([pos], dtype=torch.float),
                torch.tensor([col], dtype=torch.float),
            ],
            dim=-1,
        )
        ico = IdealCountingObserver(prog_bar=False)
        beliefs = ico(samples, return_means=False)
        # the fixture really does hit the overflow pattern
        assert ico.positions_oob[0, 6].item() is True
        assert ico.velocity_change_random[0, 4].item() is True
        # T is row-stochastic and its rows are genuine probabilities, so the
        # propagated beliefs stay a valid categorical distribution.
        assert torch.allclose(beliefs.sum(-1), torch.ones(1, 8), atol=ATOL)
        assert (beliefs >= -ATOL).all() and (beliefs <= 1 + ATOL).all()
        # the discriminating value: under the mutant this row is
        # [1.1167, 0, -0.1167]. Row-sums stay 1 either way, so the [0, 1] bound
        # is what carries this cell -- do not drop it as redundant.
        assert torch.allclose(beliefs[0, 6], torch.tensor([0.75, 0.0, 0.25]), atol=ATOL)

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
        # now: 0.25 (pure failure) -> 0.75 (pure success). E5 (the offset=2 head
        # duplication) then removed the second, spurious copy of that success:
        # counts (0, 2) -> (0, 1), so the estimate is 2/3, still above 0.5.
        ico = IdealCountingObserver(prog_bar=False)
        _, _, m_ovc, _ = ico(samples, return_means=True)
        assert not torch.isclose(m_ovc[0, -1], torch.tensor(0.5), atol=ATOL)
        assert m_ovc[0, -1] > 0.5

    def test_golden_final_outputs(self, samples):
        ico = IdealCountingObserver(prog_bar=False)
        beliefs, m_nvc, m_ovc, m_pvc = ico(samples, return_means=True)
        # ICO-A (bounce/colour pairing skew) then E5 (the offset=2 head
        # duplication) re-baseline BOTH rows. HAND-DERIVED (empirically confirmed
        # 2026-08-31, 49/49). Common index convention: every (B, T-2) event mask
        # lives in the event space, index j <-> frame j+1.
        #
        # E5 ARITHMETIC (uniform across every value below). `get_dist_params`
        # used to prepend a COPY of event rows 0 and 1 before the cumulative sum;
        # it now prepends two PRIOR-NEUTRAL zero rows. Since the old
        # `counts[:, 1]` was exactly that duplicated head (row_0 + row_1), every
        # new terminal count is `old counts[-1] - old counts[1]`. Measured heads
        # for this fixture: pccnvc (1, 0) both rows, pccovc (0, 1) both rows,
        # pvc (1, 0) both rows -- i.e. each row's j=0 no-change opportunity and,
        # for pccovc, the j=1 bounce success were being banked twice.
        #
        # seq0: x = [232, 240, 248, 240, ...] -> oob at frame 2 only,
        #   velocity_change = [F, T, F, ...] (j=1 <-> frame 2) and that vc is a
        #   BOUNCE. colours [R, R, G, G, GRAY x3, B x7] forward-fill to
        #   [R, R, G, G, G, G, G, B, ...], so color_change (j <-> frame j+1) is
        #   True at j=1 (frame 2) and j=6 (frame 7, the grayzone exit).
        #     color_change_bounce = vc & cc & ~exit -> {j=1}   (was EMPTY: the
        #       frame-2 change used to be read one cell early, at old-space j=0,
        #       and misfiled as a random-channel success)
        #     color_change_random = {j=6} only, via the exit path (the run
        #       [4, 6] carries no vc, so the exit change stays random)
        #     velocity_change_shifted = {j=1}
        #   pccnvc pair (~vcs, ccr): 11 no-change opportunities (j != 1) and the
        #     one exit success at j=6 -> counts (11, 1) -> 2/14 = 1/7
        #     (E5: was (1 + 11, 0 + 1) = (12, 1) -> 2/15).
        #   pccovc pair (vcs & ~ccb, ccb): j=1 is a SUCCESS and nothing else is a
        #     contingent opportunity -> (0, 1) -> 2/3
        #     (E5: the duplicated row j=1 banked that success twice, (0, 2) -> 3/4).
        #   pvc pair (~vc, vc_random) is built from the UNSHIFTED detectors: 11
        #     no-vc frames, no random vc -> (11, 0) -> 1/13
        #     (E5: was (1 + 11, 0) = (12, 0) -> 1/14).
        #   beliefs[0, -1]: frame 13 is visible blue -> [0, 0, 1], propagated one
        #     step. probability_bounce = 0, so vc = pvc = 1/13 and
        #     p_transition = (1/7)(12/13) + (2/3)(1/13)
        #                  = 36/273 + 14/273 = 50/273 = 0.183150
        #     (E5: was (2/15)(13/14) + (3/4)(1/14) = 149/840 = 0.177381).
        #
        # seq1: bounce at frame 2 (y = 8 < 10), colours [B, B, R, GRAY, GRAY,
        #   R, R, R, G, G, G, PAD x3] -> inferred [B, B, R, R, R, R, R, R, G, G,
        #   G, PAD x3]; color_change True at j=1 (frame 2, the bounce), j=7
        #   (frame 8) and j=10 (frame 11, the pre-existing G -> PAD artefact).
        #   The grayzone run [3, 4] exits at frame 5 with NO colour change, and
        #   carries no vc, so the exit path stays inert (DRIFT-1's verdict for
        #   this row is unchanged: velocity_change_shifted = {j=1}).
        #     color_change_bounce = {j=1}; color_change_random = {j=7, j=10}.
        #   pccnvc: (11, 2) -> 3/15 = 1/5 (E5: was (1 + 11, 0 + 2) = (12, 2)
        #     -> 3/16).
        #   pccovc: (0, 1) -> 2/3 (E5: was (0, 2) -> 3/4), same arithmetic as seq0.
        #   pvc: (11, 0) -> 1/13 (E5: was (12, 0) -> 1/14).
        #   beliefs[1, -1] is a PAD frame, never written in the loop, so it stays
        #     at the 1/3 init regardless.
        assert torch.allclose(
            beliefs[:, -1],
            torch.tensor([[50 / 273, 0.0, 223 / 273], [1 / 3, 1 / 3, 1 / 3]]),
            atol=ATOL,
        )
        assert torch.allclose(m_nvc[:, -1], torch.tensor([1 / 7, 1 / 5]), atol=ATOL)
        assert torch.allclose(m_ovc[:, -1], torch.tensor([2 / 3, 2 / 3]), atol=ATOL)
        assert torch.allclose(m_pvc[:, -1], torch.tensor([1 / 13, 1 / 13]), atol=ATOL)

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
        # arrays (no Dirichlet arithmetic, no offset=2 head padding), on the
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


class TestIdealCountingObserverDegenerateInputs:
    """Inputs that used to CRASH the ICO, and what it now emits instead.

    Operator-approved semantics for all three: do not crash; skip the
    un-computable updates and leave the latent (colour) predictions at their
    initialisation -- uniform beliefs, prior-mean colour rates.

    Pre-existing crashes, located empirically at c0ef72e:
      all-gray  IndexError in the belief backfill -- `idx_timesteps_color` is
                (last gray index) + 1, which equals T when a row is occluded end
                to end, so the gather ran off the time axis.
      T = 2     RuntimeError from a zero-width Dirichlet, then IndexError on
                `velocity_change_random[:, max(0, t - 2)]` (a (B, 0) mask).
      T = 3     IndexError on `means_pvc[:, t]`: the rate curve came out 2 wide
                instead of 3.
    The T=2 Dirichlet and the whole T=3 crash were fixed STRUCTURALLY by the E5
    head-padding change -- `stacked_vectors[:, :offset]` was only
    `min(offset, T - 2)` rows wide, whereas the zero pad is always `offset` rows,
    so the curve is `offset + (T - 2) == T` wide for every T >= 2. Only two
    explicit guards remain (the backfill filter and the (B, 0) random-change
    read); T=3 needs none, and none was added.

    NB positions are NEVER occluded in this task -- only colours are -- so an
    all-gray row still carries a full velocity stream. Its pccnvc / pvc curves
    therefore keep accumulating genuine no-velocity-change FAILURES from the
    positions; that is the model's pre-existing convention for occluded frames
    (every grayzone cell is a `~velocity_change_shifted` opportunity), not
    something these guards introduce. What must stay untouched is the COLOUR
    evidence, which is what the cells below pin.
    """

    _LINE6 = [[100.0 + 4 * t, 128.0] for t in range(6)]

    @staticmethod
    def _make(rows):
        return torch.cat(
            [
                torch.tensor([p for p, _ in rows], dtype=torch.float),
                torch.tensor([c for _, c in rows], dtype=torch.float),
            ],
            dim=-1,
        )

    def test_all_grayzone_sequence_runs_with_latents_at_init(self):
        ico = IdealCountingObserver(prog_bar=False)
        beliefs, m_nvc, m_ovc, m_pvc = ico(
            self._make([(self._LINE6, [_GRAY] * 6)]), return_means=True
        )
        assert beliefs.shape == (1, 6, 3) and m_nvc.shape == (1, 6)
        # Beliefs stay at the 1/3 init for every frame. No frame is ever
        # visible, so the belief is only ever propagated through T -- and the
        # cyclic T is doubly stochastic, so uniform is its stationary point.
        assert torch.allclose(beliefs, torch.full((1, 6, 3), 1 / 3), atol=ATOL)
        # No colour evidence is ever banked: neither colour channel declares
        # anything, so the contingent rate sits at its (1, 1) prior mean for the
        # whole curve.
        assert (ico.color_change_bounce == 0).all()
        assert (ico.color_change_random == 0).all()
        assert torch.allclose(m_ovc, torch.full((1, 6), 0.5), atol=ATOL)
        assert (ico.counts_pccnvc[..., 1] == 0).all()  # zero pccnvc SUCCESSES
        assert (ico.counts_pccovc == 0).all()  # no contingent opportunity at all
        # Characterized, not endorsed (see the class docstring): the failure
        # side of the hazard/pvc channels still accrues from the visible
        # positions, so those curves decay 1/2, 1/2, 1/3, 1/4, 1/5, 1/6.
        expected_decay = torch.tensor([[0.5, 0.5, 1 / 3, 0.25, 0.2, 1 / 6]])
        assert torch.allclose(m_nvc, expected_decay, atol=ATOL)
        assert torch.allclose(m_pvc, expected_decay, atol=ATOL)

    def test_two_frame_sequence_runs_with_rates_at_priors(self):
        # T = 2: the second difference needs a neighbour on both sides, so there
        # is NO event cell at all. The entire count curve is the prior-neutral
        # head pad, and every rate is the bare (1, 1) prior mean.
        ico = IdealCountingObserver(prog_bar=False)
        beliefs, m_nvc, m_ovc, m_pvc = ico(
            self._make([([[100.0, 128.0], [104.0, 128.0]], [_R, _G])]), return_means=True
        )
        assert beliefs.shape == (1, 2, 3)
        for m in (m_nvc, m_ovc, m_pvc):
            assert m.shape == (1, 2)
            assert torch.allclose(m, torch.full((1, 2), 0.5), atol=ATOL)
        assert (ico.counts_pvc == 0).all()
        assert torch.allclose(beliefs.sum(-1), torch.ones(1, 2), atol=ATOL)

    def test_three_frame_sequence_runs_with_rates_at_priors(self):
        # T = 3: exactly ONE event cell (j = 0 <-> frame 1), reached only by the
        # last slot of the curve. The straight line declares no velocity change
        # there, so that slot banks a single pvc FAILURE -- (1 + 0)/(2 + 1 + 0)
        # = 1/3 -- and both colour channels, which have no opportunity, stay at
        # the 0.5 prior mean across the whole curve.
        ico = IdealCountingObserver(prog_bar=False)
        beliefs, m_nvc, m_ovc, m_pvc = ico(
            self._make([([[100.0, 128.0], [104.0, 128.0], [108.0, 128.0]], [_R, _G, _G])]),
            return_means=True,
        )
        assert beliefs.shape == (1, 3, 3)
        assert torch.allclose(m_nvc, torch.full((1, 3), 0.5), atol=ATOL)
        assert torch.allclose(m_ovc, torch.full((1, 3), 0.5), atol=ATOL)
        assert torch.allclose(m_pvc, torch.tensor([[0.5, 0.5, 1 / 3]]), atol=ATOL)
        assert torch.allclose(beliefs.sum(-1), torch.ones(1, 3), atol=ATOL)

    def test_degenerate_batchmate_leaves_normal_row_bit_identical(self):
        # The guards must be per-row, not batch-global: an all-gray row must not
        # perturb a well-formed batch-mate by so much as a ULP. The two rows are
        # the SAME length (T = 6) on purpose, so this cell isolates degeneracy
        # rather than re-testing padding invariance.
        normal = (self._LINE6, [_R, _R, _G, _GRAY, _G, _G])
        degenerate = (self._LINE6, [_GRAY] * 6)
        solo = IdealCountingObserver(prog_bar=False)(self._make([normal]), return_means=True)
        mixed = IdealCountingObserver(prog_bar=False)(
            self._make([degenerate, normal]), return_means=True
        )
        for solo_out, mixed_out in zip(solo, mixed, strict=True):
            assert torch.equal(solo_out[0], mixed_out[1])


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
