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
        ico = IdealCountingObserver(prog_bar=False)
        _, _, m_ovc, _ = ico(samples, return_means=True)
        assert not torch.isclose(m_ovc[0, -1], torch.tensor(0.5), atol=ATOL)

    def test_golden_final_outputs(self, samples):
        ico = IdealCountingObserver(prog_bar=False)
        beliefs, m_nvc, m_ovc, m_pvc = ico(samples, return_means=True)
        assert torch.allclose(
            beliefs[:, -1],
            torch.tensor([[0.236345, 0.0, 0.763655], [1 / 3, 1 / 3, 1 / 3]]),
            atol=ATOL,
        )
        assert torch.allclose(m_nvc[:, -1], torch.tensor([0.235294, 0.263158]), atol=ATOL)
        assert torch.allclose(m_ovc[:, -1], torch.tensor([0.25, 0.333333]), atol=ATOL)
        assert torch.allclose(m_pvc[:, -1], torch.tensor([0.071429, 0.071429]), atol=ATOL)

    def test_deterministic(self, samples):
        ico = IdealCountingObserver(prog_bar=False)
        a = ico(samples, return_means=False)
        b = ico(samples, return_means=False)
        assert torch.equal(a, b)


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
        assert torch.allclose(
            out["beliefs"][:, -1],
            torch.tensor([[0.0, 0.0, 1.0], [1 / 3, 1 / 3, 1 / 3]]),
            atol=ATOL,
        )
        assert torch.allclose(out["p_change"][:, -1], torch.tensor([0.153846, 0.0]), atol=ATOL)
