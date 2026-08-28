"""Ideal observer models for the bouncing-ball change-point-detection task.

Migrated from ``hmdcpd-analysis`` (``src/hmdcpd/iom.py``). The model classes are
self-contained, depending only on :mod:`torch` and the bouncing-ball task
geometry. Task parameters are sourced from
:class:`bouncing_ball_task.defaults.TaskParameters` (defaulting to
``TaskParameters()``) instead of the original ``dict`` interface.

Classes
-------
IdealObserverModel
    Base module holding task geometry and the bounce-transition matrix ``T``.
IdealBayesianObserver
    Bayesian belief-updating observer.
IdealCountingObserver, IdealCountingObserverV2
    Dirichlet count-based observers.
"""

import torch
import torch.nn.functional as F
from bouncing_ball_task.defaults import TaskParameters
from torch.distributions.dirichlet import Dirichlet
from tqdm.auto import tqdm


class IdealObserverModel(torch.nn.Module):
    def __init__(self, task_parameters=None, padding_value=-1):
        super().__init__()

        self.task_parameters = task_parameters if task_parameters is not None else TaskParameters()
        self.mask_color = torch.tensor(self.task_parameters.mask_color, dtype=float)
        self.size_frame = torch.tensor(self.task_parameters.size_frame)
        self.ball_radius = self.task_parameters.ball_radius
        self.dt = self.task_parameters.dt
        self.padding_value = padding_value

    @property
    def T(self):
        self.probability_bounce = self.probability_bounce.float()
        self.vc = self.probability_bounce + self.pvc * (1 - self.probability_bounce)
        self.probability_transition = self.pccnvc * (1 - self.vc) + self.pccovc * self.vc

        return torch.stack(
            [
                torch.stack(
                    [
                        1 - self.probability_transition,
                        self.probability_transition,
                        torch.zeros_like(self.vc),
                    ],
                    axis=-1,
                ),
                torch.stack(
                    [
                        torch.zeros_like(self.vc),
                        1 - self.probability_transition,
                        self.probability_transition,
                    ],
                    axis=-1,
                ),
                torch.stack(
                    [
                        self.probability_transition,
                        torch.zeros_like(self.vc),
                        1 - self.probability_transition,
                    ],
                    axis=-1,
                ),
            ],
            axis=1,
        )

    def init_states(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass


class IdealBayesianObserver(IdealObserverModel):
    def __init__(
        self,
        probability_color_change_no_velocity_change,
        probability_color_change_on_velocity_change,
        probability_velocity_change,
        task_parameters=None,
    ):
        super().__init__(task_parameters)

        self.pccnvc = 0
        self.pccovc = 0
        self.pvc = probability_velocity_change

    def init_states(self, batch_size):
        self.probability_bounce = torch.zeros(batch_size)

    def forward(self, samples, pccnvc, pccovc, belief=None, scale=255):
        # Bayesian update
        positions = samples[:, :, :2]
        colors = samples[:, :, 2:]
        self.pccnvc = pccnvc
        self.pccovc = pccovc
        batch_size, time_steps, _ = colors.shape
        self.init_states(batch_size)
        if belief is None:
            belief = torch.ones((batch_size, 3)) / 3
        predictions_list = []

        for t in range(time_steps):
            observation = colors[:, t, :]
            non_masked = torch.all(observation[:] != self.mask_color, dim=-1)

            belief[non_masked] = torch.round(observation[non_masked] / scale)

            self.probability_bounce = torch.logical_or(
                torch.any(
                    positions[:, t, :] > self.size_frame - self.ball_radius,
                    axis=-1,
                ),
                torch.any(
                    positions[:, t, :] < torch.tensor([self.ball_radius] * 2),
                    axis=-1,
                ),
            ).float()
            belief = torch.bmm(belief.unsqueeze(1).float(), self.T.float()).squeeze(1)
            predictions_list.append(belief.detach().clone())

        predictions = torch.stack(predictions_list, axis=1)

        return predictions


class IdealCountingObserverV2(torch.nn.Module):
    _CHANGE_MASK = torch.tensor(
        [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
        dtype=torch.float32,
    )
    _NO_CHANGE_MASK = torch.eye(3, dtype=torch.float32)

    def __init__(
        self,
        task_parameters=None,
        beta_priors: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        padding_value=-1,
    ):
        super().__init__()

        self.task_parameters = task_parameters if task_parameters is not None else TaskParameters()
        self.mask_color = torch.tensor(self.task_parameters.mask_color, dtype=float)
        self.size_frame = torch.tensor(self.task_parameters.size_frame)
        self.size_x, self.size_y = self.size_frame
        self.ball_radius = self.task_parameters.ball_radius
        self.dt = self.task_parameters.dt
        self.padding_value = padding_value

        # register buffers so they follow the module's device
        alpha_hz, beta_hz, alpha_cont, beta_cont = beta_priors
        self.register_buffer("alpha_hz_prior", torch.tensor(alpha_hz))
        self.register_buffer("beta_hz_prior", torch.tensor(beta_hz))
        self.register_buffer("alpha_cont_prior", torch.tensor(alpha_cont))
        self.register_buffer("beta_cont_prior", torch.tensor(beta_cont))

    def _derive_bounce(
        self,
        positions,
        radius: float,
        width: int,
        height: int,
    ):
        batch_size, timesteps, _ = positions.shape
        positions_oob = torch.logical_or(
            torch.any(
                positions > self.size_frame - self.ball_radius,
                axis=-1,
            ),
            torch.any(
                positions < torch.tensor([self.ball_radius] * 2),
                axis=-1,
            ),
        )
        velocity = positions[:, 1:, :] - positions[:, :-1, :]
        velocity_diff = velocity[:, 1:, :] - velocity[:, :-1, :]
        velocity_change = (velocity_diff[:, :] ** 2).max(dim=-1)[0].to(int).to(bool)

        bounce = torch.zeros((batch_size, timesteps)).to(bool)
        bounce[:, 1:-1] = positions_oob[:, 1:-1] & velocity_change
        return bounce

        # velocity = positions[:, 1:] - positions[:, :-1]
        # sign_flip = (velocity[:, 1:] * velocity[:, :-1]) < 0

        # near_left   = positions[:, 1:-1, 0] < radius
        # near_right  = positions[:, 1:-1, 0] > width - radius
        # near_top    = positions[:, 1:-1, 1] < radius
        # near_bottom = positions[:, 1:-1, 1] > height - radius
        # near_wall   = torch.stack(
        #     [near_left | near_right, near_top | near_bottom],
        #     dim=-1
        # )

        # bounce = torch.zeros_like(positions[..., 0], dtype=torch.bool)
        # bounce[:, 2:] = (sign_flip & near_wall).any(-1)

        # import ipdb; ipdb.set_trace()
        # return bounce

    def T(self, prob_no_change, prob_change, batch_size):
        return torch.stack(
            [
                torch.stack([prob_no_change, prob_change, prob_change.new_zeros(batch_size)], -1),
                torch.stack([prob_change.new_zeros(batch_size), prob_no_change, prob_change], -1),
                torch.stack([prob_change, prob_change.new_zeros(batch_size), prob_no_change], -1),
            ],
            1,
        )

        pass

    def forward(
        self,
        samples,
        scale=255,
        return_means: bool = False,
    ):
        positions, colors = samples.split([2, 3], dim=-1)  # positions (B,T,2), colors (B,T,3)
        colors /= scale
        mask_valid = colors.sum(dim=-1) >= 0  # True iff timestep is inside the trial

        device = colors.device
        batch_size, timesteps, _ = colors.shape  # NB: timesteps = T_max

        # -------- bounce flag (computed once for the whole padded batch) ----------
        bounce = self._derive_bounce(positions, self.ball_radius, self.size_x, self.size_y).to(
            device
        )  # (B, T_max)

        # ----------------------- initialise beliefs & parameters ------------------
        color_beliefs = torch.full((batch_size, timesteps, 3), 1.0 / 3.0, device=device)

        alpha_hz = self.alpha_hz_prior.expand(batch_size).clone()
        beta_hz = self.beta_hz_prior.expand(batch_size).clone()
        alpha_cont = self.alpha_cont_prior.expand(batch_size).clone()
        beta_cont = self.beta_cont_prior.expand(batch_size).clone()

        # accumulators active only while inside the gray zone
        acc_succ_hz = torch.zeros(batch_size, device=device)
        acc_fail_hz = torch.zeros(batch_size, device=device)
        acc_succ_cont = torch.zeros(batch_size, device=device)
        acc_fail_cont = torch.zeros(batch_size, device=device)
        inside_gray = torch.zeros(batch_size, dtype=torch.bool, device=device)

        p_change_now = torch.zeros(batch_size, timesteps, device=device) if return_means else None

        # ===================================================================== loop
        for t in range(1, timesteps):
            color_belief = color_beliefs[:, t]

            # --- masks for *this* pair of frames ----------------------------------
            valid_prev = mask_valid[:, t - 1]
            valid_now = mask_valid[:, t]
            step_valid = valid_prev & valid_now  # only these seqs matter at t

            # import ipdb; ipdb.set_trace()

            if not step_valid.any():  # nothing to update this step
                continue

            bounce_t = bounce[:, t] & step_valid  # invalidate bounces on padding

            # ------------- current expected transition probabilities --------------
            mean_hz = alpha_hz / (alpha_hz + beta_hz)
            mean_cont = alpha_cont / (alpha_cont + beta_cont)

            prob_no_change = torch.where(bounce_t, 1 - mean_cont, 1 - mean_hz)
            prob_change = torch.where(bounce_t, mean_cont, mean_hz)

            T = self.T(prob_no_change, prob_change, batch_size)  # (B,3,3)

            # ------------- expected advance for hidden-step bookkeeping -----------
            change_prob_per_state = (T * self._CHANGE_MASK.to(device)).sum(-1)
            exp_change = (color_belief * change_prob_per_state).sum(-1)
            exp_no_change = 1.0 - exp_change

            # ------------- classify transition: nongray / gray -------------------
            visible_prev = step_valid & (colors[:, t - 1, 0] != -1)
            visible_now = step_valid & (colors[:, t, 0] != -1)

            nongray_transition = visible_prev & visible_now  # fully visible
            gray_transition = step_valid & ~nongray_transition  # any part hidden

            # -------------------- (1) hidden step → accumulate expectations -------
            if gray_transition.any():
                mask_no_bounce = gray_transition & (~bounce_t)
                mask_bounce = gray_transition & bounce_t

                acc_succ_hz += exp_change * mask_no_bounce.float()
                acc_fail_hz += exp_no_change * mask_no_bounce.float()
                acc_succ_cont += exp_change * mask_bounce.float()
                acc_fail_cont += exp_no_change * mask_bounce.float()

            # -------------------- (2) visible step → deterministic counts ---------
            if nongray_transition.any():
                unchanged = (
                    ((colors[:, t] == colors[:, t - 1]) & nongray_transition.unsqueeze(-1))
                    .all(-1)
                    .float()
                )

                # E1 fix: argmax-cyclic detector. The old `%3==1` on one-hot vectors
                # required all three channels to equal 1 after `.all(-1)`, so it never
                # fired. argmax names the colour; a change is a single forward step.
                changed = (
                    ((colors[:, t].argmax(-1) - colors[:, t - 1].argmax(-1)) % 3 == 1)
                    & nongray_transition
                ).float()

                mask_no_bounce = nongray_transition & (~bounce_t)
                mask_bounce = nongray_transition & bounce_t

                alpha_hz += changed * mask_no_bounce.float()
                beta_hz += unchanged * mask_no_bounce.float()
                alpha_cont += changed * mask_bounce.float()
                beta_cont += unchanged * mask_bounce.float()

            exiting = visible_now & inside_gray
            if exiting.any():
                alpha_hz[exiting] += acc_succ_hz[exiting]
                beta_hz[exiting] += acc_fail_hz[exiting]
                alpha_cont[exiting] += acc_succ_cont[exiting]
                beta_cont[exiting] += acc_fail_cont[exiting]

                acc_succ_hz[exiting] = acc_fail_hz[exiting] = 0.0
                acc_succ_cont[exiting] = acc_fail_cont[exiting] = 0.0

            # update inside_gray for the next frame
            inside_gray = (~visible_now) | (inside_gray & gray_transition)

            color_pred = torch.einsum("bs,bsj->bj", color_belief, T)

            emission = torch.ones_like(color_pred)  # default: uniform
            emission[visible_now] = F.one_hot(  # overwrite where visible
                ((colors[visible_now, t] / scale).argmax(-1)), num_classes=3
            ).float()

            color_update = color_pred * emission

            # mask out padded rows so they never divide by zero
            row_sum = color_update.sum(1, keepdim=True)  # (B,1)
            is_valid_row = row_sum.squeeze(1) > 0  # True iff not a pure padding row
            row_sum = torch.where(is_valid_row.unsqueeze(1), row_sum, torch.ones_like(row_sum))
            color_belief = torch.where(
                is_valid_row.unsqueeze(1),  # update only valid sequences
                color_update / row_sum,  # normalise
                color_belief,  # leave padding rows unchanged
            )

            # color_belief = color_pred * emission
            # color_belief /= color_belief.sum(1, keepdim=True)

            # -------------------- (5) online hazard curve -------------------------
            if return_means:
                p_now = torch.where(bounce_t, mean_cont, mean_hz)
                p_change_now[:, t] = torch.where(step_valid, p_now, p_change_now[:, t])

            color_beliefs[:, t] = color_belief

        # ============================ flush incomplete grey at sequence end =======
        unfinished = inside_gray
        if unfinished.any():
            alpha_hz[unfinished] += acc_succ_hz[unfinished]
            beta_hz[unfinished] += acc_fail_hz[unfinished]
            alpha_cont[unfinished] += acc_succ_cont[unfinished]
            beta_cont[unfinished] += acc_fail_cont[unfinished]

        betas = torch.stack([alpha_hz, beta_hz, alpha_cont, beta_cont], dim=1)

        output = {"beliefs": color_beliefs, "betas": betas}
        if return_means:
            output["p_change"] = p_change_now
        return output


class IdealCountingObserver(IdealObserverModel):
    def __init__(
        self,
        task_parameters=None,
        prior_pvc=(1, 1),
        prior_pccovc=(1, 1),
        prior_pccnvc=(1, 1),
        prog_bar=True,
        *args,
        **kwargs,
    ):
        super().__init__(task_parameters, *args, **kwargs)
        self.prior_pvc = torch.tensor(prior_pvc)
        self.prior_pccovc = torch.tensor(prior_pccovc)
        self.prior_pccnvc = torch.tensor(prior_pccnvc)
        self.prog_bar = prog_bar
        self.n_count = 4

    def init_states(self, batch_size):
        self.probability_bounce = torch.zeros(batch_size)
        self.pccovc = torch.zeros(batch_size)
        self.pccnvc = torch.zeros(batch_size)
        self.pvc = torch.zeros(batch_size)

    def forward(
        self,
        samples,
        beliefs=None,
        scale=255,
        return_means=False,
        prior_pvc=None,
        prior_pccovc=None,
        prior_pccnvc=None,
    ):
        batch_size, timesteps, _ = samples.shape
        prior_pvc = torch.tensor(prior_pvc) if prior_pvc else self.prior_pvc
        prior_pccovc = torch.tensor(prior_pccovc) if prior_pccovc else self.prior_pccovc
        prior_pccnvc = torch.tensor(prior_pccnvc) if prior_pccnvc else self.prior_pccnvc

        positions = samples[:, :, :2]
        self.positions_oob = torch.logical_or(
            torch.any(
                positions > self.size_frame - self.ball_radius,
                axis=-1,
            ),
            torch.any(
                positions < torch.tensor([self.ball_radius] * 2),
                axis=-1,
            ),
        )

        # Calculate velocity and the different changes
        velocity = positions[:, 1:, :] - positions[:, :-1, :]
        velocity_diff = velocity[:, 1:, :] - velocity[:, :-1, :]
        self.velocity_change = (velocity_diff[:, :] ** 2).max(dim=-1)[0].to(int).to(bool)

        self.velocity_change_bounce = torch.logical_and(
            self.positions_oob[:, 1:-1],
            self.velocity_change,
        )
        self.velocity_change_random = torch.logical_and(
            ~self.positions_oob[:, 1:-1],
            self.velocity_change,
        )

        # Calculate velocity counts
        # self.counts_velocity_change = torch.cumsum(self.velocity_change, dim=1)
        # self.counts_no_velocity_change = torch.cumsum(~self.velocity_change, dim=1)

        # Color changes are more complex to obtain since they arent observable
        # in the grayzone
        colors = samples[:, :, 2:]

        # Grayzone masks
        mask_grayzone = (colors == self.mask_color).all(dim=-1)  # shape [B, T]
        mask_non_grayzone = ~mask_grayzone  # shape [B, T]
        mask_idx_after_grayzone = torch.logical_and(mask_non_grayzone[:, 1:], mask_grayzone[:, :-1])

        # Compute the longest grayzone length
        mask_grayzone_padded_int = torch.cat(
            [
                torch.zeros((batch_size, 1)),
                mask_grayzone.int(),
                torch.zeros((batch_size, 1)),
            ],
            dim=1,
        )
        mask_grayzone_diff = mask_grayzone_padded_int[:, 1:] - mask_grayzone_padded_int[:, :-1]
        grayzone_diffs = (mask_grayzone_diff == -1).nonzero(as_tuple=True)[1] - (
            mask_grayzone_diff == 1
        ).nonzero(as_tuple=True)[1]
        max_grayzone_diff = 1 + max(grayzone_diffs) // 2

        # Create an index array along the time axis.
        timesteps_arange = torch.arange(timesteps)  # shape [T]
        # For valid entries, keep their time index; for masked entries, set index to -1.
        idx_non_grayzone = torch.where(mask_non_grayzone, timesteps_arange, -1)  # shape [B, T]

        # Infer what the colors are to create the color change arrays
        colors_inferred = colors.clone()

        # Foward fill the colors according to what was last seen
        idx_last_observed_color = torch.cummax(idx_non_grayzone, dim=1)[0]
        idx_last_observed_color_valid = idx_last_observed_color != -1

        # Get the batch and time indices where valid forward-fill exists.
        idx_batch_valid, idx_timesteps_valid = idx_last_observed_color_valid.nonzero(as_tuple=True)

        colors_inferred[idx_batch_valid, idx_timesteps_valid, :] = colors[
            idx_batch_valid, idx_last_observed_color[idx_batch_valid, idx_timesteps_valid], :
        ]

        # Backfill initial colors if they started in the grayzone
        mask_grayzone_remaining = (colors_inferred == self.mask_color).all(dim=-1)
        idx_grayzone_batch, idx_grayzone_timesteps = mask_grayzone_remaining.nonzero(as_tuple=True)
        idx_batch_remaining, idx_batch_map = torch.unique(
            idx_grayzone_batch,
            sorted=True,
            return_inverse=True,
        )

        idx_timesteps_color = torch.full(
            (idx_batch_remaining.shape[0],),
            float("-inf"),
        ).to(idx_grayzone_timesteps.dtype)

        idx_timesteps_color = (
            idx_timesteps_color.scatter_reduce(
                0,
                idx_batch_map,
                idx_grayzone_timesteps,
                reduce="amax",
                include_self=True,
            )
            + 1
        )  # Plus one to get the idx after the highest grayzone idx

        backfill_colors = colors_inferred[idx_batch_remaining, idx_timesteps_color, :]
        backfill_mask = timesteps_arange.unsqueeze(0) < idx_timesteps_color.unsqueeze(1)

        # Assume if started in the grayzone, they have the same color as when observed
        colors_inferred[idx_batch_remaining] = torch.where(
            backfill_mask.unsqueeze(-1),  # shape: [N, T, 1] broadcast over color channels
            backfill_colors.unsqueeze(1).expand(-1, timesteps, -1),  # shape: [N, T, C]
            colors_inferred[idx_batch_remaining],
        )

        # Start computing color differences
        color_diff = colors_inferred[:, 1:, :] - colors_inferred[:, :-1, :]
        color_change = color_diff.abs().max(dim=-1)[0].to(bool)

        # Color changes that coincide with a bounce that are not immediately
        # following the grayzone
        self.color_change_bounce = torch.logical_and(
            torch.logical_and(
                self.velocity_change,
                color_change[:, 1:],
            ),
            ~mask_idx_after_grayzone[:, 1:],
        ).float()

        # Color changes that happen when there isnt a bounce that dont happen
        # immediately after exiting the grayzone
        self.color_change_random = torch.logical_and(
            ~self.velocity_change & color_change[:, 1:],
            ~mask_idx_after_grayzone[:, 1:],
        ).float()

        # Handle grayzone changes
        color_change_grayzone = torch.logical_and(
            color_change[:, 1:],
            mask_idx_after_grayzone[:, 1:],
        )

        mask_grayzone_with_velocity_change_bounce = self.find_overlapping_grayzone(
            mask_grayzone[:, 2:] | mask_idx_after_grayzone[:, 1:],
            self.velocity_change_bounce & mask_grayzone[:, 2:],
            max_grayzone_diff,
        )

        mask_grayzone_with_velocity_change_random = self.find_overlapping_grayzone(
            mask_grayzone[:, 2:] | mask_idx_after_grayzone[:, 1:],
            self.velocity_change_random & mask_grayzone[:, 2:],
            max_grayzone_diff,
        )

        mask_grayzone_with_velocity_change = torch.logical_or(
            mask_grayzone_with_velocity_change_bounce,
            mask_grayzone_with_velocity_change_random,
        )

        # Find color changes where the velocity didnt change
        mask_color_changes_with_no_grayzone_velocity_change = torch.logical_and(
            color_change_grayzone,
            ~mask_grayzone_with_velocity_change,
        )

        # Add those to random color changes
        self.color_change_random += mask_color_changes_with_no_grayzone_velocity_change.float()

        # #
        # # Need to handle color changes in the grayzone where the velocity
        # # changes as well
        # #
        # mask_color_changes_with_grayzone_velocity_change = torch.logical_and(
        #     color_change_grayzone,
        #     mask_grayzone_with_velocity_change,
        # )
        # self.color_change_random += mask_color_changes_with_grayzone_velocity_change
        # self.color_change_bounce += mask_color_changes_with_grayzone_velocity_change

        # Make a new change array that shifts grayzone velocity changes to be the
        # index after the grayzone so that color counts are updated properly
        self.velocity_change_bounce_shifted = self.velocity_change_bounce.clone()

        self.velocity_change_bounce_shifted[mask_grayzone_with_velocity_change_bounce] = False
        self.velocity_change_bounce_shifted[
            mask_grayzone_with_velocity_change_bounce & mask_idx_after_grayzone[:, 1:]
        ] = True

        self.velocity_change_random_shifted = self.velocity_change_random.clone()
        self.velocity_change_random_shifted[mask_grayzone_with_velocity_change_random] = False
        self.velocity_change_random_shifted[
            mask_grayzone_with_velocity_change_random & mask_idx_after_grayzone[:, 1:]
        ] = True

        self.velocity_change_shifted = torch.logical_or(
            self.velocity_change_bounce_shifted,
            self.velocity_change_random_shifted,
        )

        means_pccnvc, means_pccovc, means_pvc = self.get_all_dist_params(
            (~self.velocity_change_shifted, self.color_change_random),
            prior_pccnvc,
            (
                self.velocity_change_shifted & ~self.color_change_bounce.bool(),
                self.color_change_bounce,
            ),
            prior_pccovc,
            (~self.velocity_change, self.velocity_change_random),
            prior_pvc,
        )

        # for i in range(10):
        #     p_color_change_random = 1 - (
        #         1 - means_pccnvc[:, 2:][mask_color_changes_with_grayzone_velocity_change]
        #     )**(median_grayzone_diff - 1)
        #     p_color_change_bounce = 1 - (
        #         1 - means_pccovc[:, 2:][mask_color_changes_with_grayzone_velocity_change]
        #     )
        #     p_comb = p_color_change_random + p_color_change_bounce

        #     self.color_change_random[mask_color_changes_with_grayzone_velocity_change] = (
        #         p_color_change_random / p_comb
        #     )
        #     self.color_change_bounce[mask_color_changes_with_grayzone_velocity_change] = (
        #         p_color_change_bounce / p_comb
        #     )

        #     means_pccnvc, means_pccovc, means_pvc = self.get_all_dist_params(
        #         (~self.velocity_change_shifted, self.color_change_random),
        #         prior_pccnvc,
        #         (self.velocity_change_shifted & ~self.color_change_bounce.bool(), self.color_change_bounce),
        #         prior_pccovc,
        #         (~self.velocity_change, self.velocity_change_random),
        #         prior_pvc,
        #     )

        self.init_states(batch_size)
        if beliefs is None:
            beliefs = torch.ones((batch_size, timesteps, 3)) / 3

        if self.prog_bar:
            range_timesteps = tqdm(range(timesteps))
        else:
            range_timesteps = range(timesteps)

        for t in range_timesteps:
            observation = colors[:, t]

            # Filter to update non-padded timesteps
            non_padded = (observation != self.padding_value).all(dim=-1)
            # Filter to just update on sequences not in grayzone
            non_masked = (observation != self.mask_color).all(dim=-1)

            # Combined mask
            mask_non_grayzone_t = non_padded & non_masked
            mask_grayzone_t = non_padded & ~non_masked

            # Apply the filter to the mask
            beliefs[mask_non_grayzone_t, t] = torch.round(
                observation[mask_non_grayzone_t] / scale
            ).float()
            if t > 0:
                beliefs[mask_grayzone_t, t] = beliefs[mask_grayzone_t, t - 1]

            self.probability_bounce = (
                self.positions_oob[:, t] + self.velocity_change_random[:, max(0, t - 2)]
            ).float()
            self.pvc = self.means_pvc[:, t]
            self.pccovc = self.means_pccovc[:, t]
            self.pccnvc = self.means_pccnvc[:, t]

            # Make a prediction for the next timestep based on stats
            beliefs[non_padded, t] = (
                torch.bmm(beliefs[non_padded, t].unsqueeze(1).float(), self.T[non_padded])
                .squeeze(1)
                .float()
            )

        if return_means:
            # print(means_pccnvc.shape) #each row is (1,#timesteps)
            return beliefs, self.means_pccnvc, self.means_pccovc, self.means_pvc
        else:
            return beliefs

    def find_overlapping_grayzone(
        self,
        mask_grayzone,
        mask_change,
        max_grayzone_diff,
    ):
        timesteps_mask_change = mask_change.shape[1]
        timesteps_mask_change_arange = torch.arange(
            mask_change.shape[1],
            device=mask_change.device,
        )
        i_grid = timesteps_mask_change_arange.unsqueeze(1).expand(
            timesteps_mask_change,
            timesteps_mask_change,
        )
        j_grid = timesteps_mask_change_arange.unsqueeze(0).expand(
            timesteps_mask_change,
            timesteps_mask_change,
        )
        band = (i_grid >= (j_grid - max_grayzone_diff)) & (i_grid < (j_grid + max_grayzone_diff))
        return torch.logical_and(
            mask_grayzone, torch.logical_and(band.unsqueeze(0), mask_change.unsqueeze(1)).any(dim=2)
        )

    def get_all_dist_params(
        self,
        pccnvc_params,
        prior_pccnvc,
        pccovc_params,
        prior_pccovc,
        pvc_params,
        prior_pvc,
    ):
        self.dist_pccnvc, self.counts_pccnvc = self.get_dist_params(
            *pccnvc_params,
            prior=prior_pccnvc,
        )
        self.means_pccnvc = self.dist_pccnvc.mean[:, :, 1]

        self.dist_pccovc, self.counts_pccovc = self.get_dist_params(
            *pccovc_params,
            prior=prior_pccovc,
        )
        self.means_pccovc = self.dist_pccovc.mean[:, :, 1]

        self.dist_pvc, self.counts_pvc = self.get_dist_params(
            *pvc_params,
            prior=prior_pvc,
        )
        self.means_pvc = self.dist_pvc.mean[:, :, 1]

        return self.means_pccnvc, self.means_pccovc, self.means_pvc

    def get_dist_params(
        self,
        *change_vectors,
        prior=(1, 1),
        offset=2,
    ):
        stacked_vectors = torch.stack(change_vectors, dim=-1)
        # # pad = torch.zeros((change_vectors[0].shape[0], offset, len(change_vectors)))
        # pad = stacked_vectors[:, :offset]
        # pad[:, :, 0] = stacked_vectors[:, :offset, 0]
        # pad[:, :, 1] = stacked_vectors[:, :offset, 1]

        counts = torch.cumsum(
            torch.cat(
                [
                    stacked_vectors[:, :offset],
                    stacked_vectors,
                ],
                dim=1,
            ),
            dim=1,
        )
        return Dirichlet(prior + counts), counts
