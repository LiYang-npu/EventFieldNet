from ._rng import install_cpu_generator_state
from field_primitives.model.selector import Round1FieldScoreHeads
from field_core.selector import span_mean
from dataclasses import replace, dataclass
from field_core.selector import FieldScoreOutput
import torch


@dataclass
class SupportStructureOutput(FieldScoreOutput):
    support_base_raw: torch.Tensor
    support_context_delta: torch.Tensor
    support_dispersion_delta: torch.Tensor
    support_neighbor_count: torch.Tensor
    support_variance_mean: torch.Tensor


class _SupportFieldScoreHeads(Round1FieldScoreHeads):
    def __init__(
        self,
        *args,
        support_context=False,
        support_dispersion=False,
        carrier_mode="tanh",
        region_gain=1.0,
        transition_gain=1.0,
        transition_context_mode="none",
        linear_calibration=False,
        pairwise_interactions=False,
        length_calibration=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if carrier_mode != "tanh":
            raise ValueError(carrier_mode)
        if transition_context_mode != "none":
            raise ValueError("Round31 fixes transition_context_mode=none")
        self.linear_calibration = bool(linear_calibration)
        self.pairwise_interactions = bool(pairwise_interactions)
        if self.linear_calibration:
            self.linear_weights = torch.nn.Parameter(torch.zeros(4))
        if self.pairwise_interactions:
            self.pair_weights = torch.nn.Parameter(torch.zeros(6))
        self.length_calibration = bool(length_calibration)
        if self.length_calibration:
            self.length_bias_gain = torch.nn.Parameter(torch.zeros(1))
        if region_gain not in (1.0, 2.0) or transition_gain not in (1.0, 2.0):
            raise ValueError("fixed field gains must be1 or2")
        self.region_gain = float(region_gain)
        self.transition_gain = float(transition_gain)
        self.support_context = bool(support_context)
        self.support_dispersion = bool(support_dispersion)
        for enabled, name, seed in (
            (self.support_context, "context_branch", 1102),
            (self.support_dispersion, "dispersion_branch", 1104),
        ):
            if enabled:
                with torch.random.fork_rng(devices=[]):
                    install_cpu_generator_state(seed)
                    branch = torch.nn.Sequential(
                        torch.nn.Linear(self.hidden_dim, max(4, self.hidden_dim // 2)),
                        torch.nn.GELU(),
                        torch.nn.Linear(max(4, self.hidden_dim // 2), 1),
                    )
                    torch.nn.init.zeros_(branch[-1].weight)
                    torch.nn.init.zeros_(branch[-1].bias)
                setattr(self, name, branch)
        self.carrier_mode = carrier_mode
        self.transition_context_mode = transition_context_mode

    def transition_readouts(self, state, valid, video_padding_mask=None, context=True):
        roles = self._state_role_updates(state)
        token_valid = (
            ~video_padding_mask.bool()
            if video_padding_mask is not None
            else valid.any(-1) | valid.any(-2)
        )
        values = []
        for name, axis in [("transition_start", 2), ("transition_end", 1)]:
            source = getattr(state, name + "_input", None)
            if (
                not isinstance(source, torch.Tensor)
                or source.shape[:2] != roles.shape[:2]
            ):
                source = roles[:, :, 2, :]
            head = getattr(self, name)
            h = head.encode(source)
            if context:
                masked = h.masked_fill(~token_valid[..., None], 0)
                mean = masked.sum(1) / token_valid.sum(1).clamp_min(1)[:, None]
                h = h.unsqueeze(axis) + span_mean(masked) - mean[:, None, None, :]
                raw = head.readout(h)
            else:
                raw = head.readout(h).unsqueeze(axis).expand_as(valid)
            values.append(raw.masked_fill(~valid, 0))
        return values

    def raw_from_state(self, state, valid, carrier, video_padding_mask=None):
        field = super().raw_from_state(
            state, valid, carrier, video_padding_mask=video_padding_mask
        )
        if self.transition_context_mode == "none":
            return self.add_support_structure(field, state, video_padding_mask)
        rs, re = self.transition_readouts(
            state, valid, video_padding_mask, context=True
        )
        ts = torch.tanh(rs.float()).masked_fill(~valid, 0)
        te = torch.tanh(re.float()).masked_fill(~valid, 0)
        score = field.carrier + field.evidence + field.support + 0.5 * (ts + te)
        return replace(
            field,
            raw_transition_start=rs,
            raw_transition_end=re,
            transition_start=ts,
            transition_end=te,
            score=score.masked_fill(~valid, 0),
        )

    def residuals(
        self, field, overrides=None, disable_linear=False, disable_pair=False
    ):
        overrides = {} if overrides is None else overrides
        values = [
            overrides.get(name, getattr(field, name)).float()
            for name in ("evidence", "support", "transition_start", "transition_end")
        ]
        x = torch.stack(values, -1)
        zero = torch.zeros_like(field.score)
        linear = (
            0.5 * torch.tanh((x * self.linear_weights).sum(-1) / 4)
            if self.linear_calibration and not disable_linear
            else zero
        )
        products = torch.stack(
            [
                values[i] * values[j]
                for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
            ],
            -1,
        )
        pair = (
            0.5 * torch.tanh((products * self.pair_weights).sum(-1) / 6)
            if self.pairwise_interactions and not disable_pair
            else zero
        )
        return linear.masked_fill(~field.valid, 0), pair.masked_fill(~field.valid, 0)

    def length_residual(self, field):
        """round36 H3: zero-init learnable duration-conditioned score bias."""
        length = field.valid.shape[-1]
        idx = torch.arange(length, device=field.score.device, dtype=field.score.dtype)
        duration_ratio = (
            (idx[None, :] - idx[:, None]) / float(max(1, length - 1))
        ).clamp(0.0, 1.0)
        bias = self.length_bias_gain * duration_ratio[None]
        return bias.expand_as(field.score).masked_fill(~field.valid, 0)

    def compose_score(
        self, field, overrides=None, disable_linear=False, disable_pair=False
    ):
        overrides = {} if overrides is None else overrides
        get = lambda name: overrides.get(name, getattr(field, name)).float()
        linear, pair = self.residuals(field, overrides, disable_linear, disable_pair)
        a, b = self.region_gain, self.transition_gain
        if a == 1.0 and b == 1.0:
            return (
                get("carrier")
                + get("evidence")
                + get("support")
                + 0.5 * (get("transition_start") + get("transition_end"))
                + linear
                + pair
            ).masked_fill(~field.valid, 0)
        return (
            4.0
            / (1.0 + 2.0 * a + b)
            * (
                get("carrier")
                + a * (get("evidence") + get("support"))
                + b * 0.5 * (get("transition_start") + get("transition_end"))
            )
        ).masked_fill(~field.valid, 0)

    def apply_residual(self, field):
        if self.region_gain != 1.0 or self.transition_gain != 1.0:
            return replace(field, score=self.compose_score(field))
        if (
            not self.linear_calibration
            and not self.pairwise_interactions
            and not self.length_calibration
        ):
            return field
        linear, pair = self.residuals(field)
        length = (
            self.length_residual(field)
            if self.length_calibration
            else torch.zeros_like(field.score)
        )
        return replace(
            field,
            score=(field.score + linear + pair + length).masked_fill(~field.valid, 0),
        )

    @staticmethod
    def support_statistics(h, token_valid):
        # Prefix sums count only valid tokens, including noncontiguous masks.
        h = h.float().masked_fill(~token_valid[..., None], 0)
        count = token_valid.float()[..., None]
        prefix = lambda x: torch.nn.functional.pad(x.cumsum(1), (0, 0, 1, 0))
        ph, pc, p2 = prefix(h), prefix(count), prefix(h.square())
        length = h.shape[1]
        index = torch.arange(length, device=h.device)
        start, end = index[:, None], index[None, :] + 1
        n = pc[:, end] - pc[:, start]
        mean = (ph[:, end] - ph[:, start]) / n.clamp_min(1)
        variance = (
            (p2[:, end] - p2[:, start]) / n.clamp_min(1) - mean.square()
        ).clamp_min(0)
        left, right = (start - 2).clamp_min(0), (end + 2).clamp_max(length)
        outside_n = pc[:, start] - pc[:, left] + pc[:, right] - pc[:, end]
        outside = (
            ph[:, start] - ph[:, left] + ph[:, right] - ph[:, end]
        ) / outside_n.clamp_min(1)
        contrast = (mean - outside).masked_fill(outside_n.eq(0), 0)
        return contrast, variance, outside_n.squeeze(-1)

    def add_support_structure(self, field, state, video_padding_mask):
        roles = self._state_role_updates(state)
        mask = (
            ~video_padding_mask.bool()
            if video_padding_mask is not None
            else field.valid.any(-1) | field.valid.any(-2)
        )
        contrast, variance, neighbors = self.support_statistics(
            self.support.encode(roles[:, :, 1, :]), mask
        )
        zero = torch.zeros_like(field.raw_support)
        context_delta = (
            self.context_branch(contrast)
            .squeeze(-1)
            .masked_fill(neighbors.eq(0) | ~field.valid, 0)
            if self.support_context
            else zero
        )
        dispersion_delta = (
            self.dispersion_branch(variance).squeeze(-1).masked_fill(~field.valid, 0)
            if self.support_dispersion
            else zero
        )
        raw = (field.raw_support + context_delta + dispersion_delta).masked_fill(
            ~field.valid, 0
        )
        support = torch.tanh(raw.float()).masked_fill(~field.valid, 0)
        # Identical arithmetic to the original mean-path composition at initialization.
        score = (field.score + (support - field.support)).masked_fill(~field.valid, 0)
        values = dict(field.__dict__, raw_support=raw, support=support, score=score)
        return SupportStructureOutput(
            **values,
            support_base_raw=field.raw_support,
            support_context_delta=context_delta,
            support_dispersion_delta=dispersion_delta,
            support_neighbor_count=neighbors.masked_fill(~field.valid, 0),
            support_variance_mean=variance.mean(-1).masked_fill(~field.valid, 0),
        )

    def score_without_support_branch(self, field, branch):
        if branch not in ("context", "dispersion"):
            raise ValueError(branch)
        delta = getattr(field, "support_" + branch + "_delta")
        support = torch.tanh((field.raw_support - delta).float()).masked_fill(
            ~field.valid, 0
        )
        return self.compose_score(field, {"support": support})


from field_core.selector import centered_bounded


@dataclass
class EvidenceStructureOutput(SupportStructureOutput):
    evidence_base_raw: torch.Tensor
    evidence_pool_delta: torch.Tensor
    evidence_context_delta: torch.Tensor
    evidence_neighbor_count: torch.Tensor


class _LegacyFieldScoreHeads(_SupportFieldScoreHeads):
    def __init__(self, *args, evidence_pool=False, evidence_context=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.evidence_pool = bool(evidence_pool)
        self.evidence_context = bool(evidence_context)
        for enabled, name, seed in (
            (evidence_pool, "evidence_pool_branch", 1402),
            (evidence_context, "evidence_context_branch", 1404),
        ):
            if enabled:
                with torch.random.fork_rng(devices=[]):
                    install_cpu_generator_state(seed)
                    branch = torch.nn.Sequential(
                        torch.nn.Linear(self.hidden_dim, max(4, self.hidden_dim // 2)),
                        torch.nn.GELU(),
                        torch.nn.Linear(max(4, self.hidden_dim // 2), 1),
                    )
                    torch.nn.init.zeros_(branch[-1].weight)
                    torch.nn.init.zeros_(branch[-1].bias)
                setattr(self, name, branch)

    def raw_from_state(self, state, valid, carrier, video_padding_mask=None):
        field = super().raw_from_state(state, valid, carrier, video_padding_mask)
        roles = self._state_role_updates(state)
        mask = (
            ~video_padding_mask.bool()
            if video_padding_mask is not None
            else valid.any(-1) | valid.any(-2)
        )
        h = self.evidence.encode(roles[:, :, 0, :])
        # New residual pooling counts valid tokens; original E pooling is untouched.
        masked = h.float().masked_fill(~mask[..., None], 0)
        summed = span_mean(masked)
        count_fraction = span_mean(mask.float()[..., None])
        pooled = summed / count_fraction.clamp_min(1.0e-12)
        contrast, _, neighbors = self.support_statistics(h, mask)
        zero = torch.zeros_like(field.raw_evidence)
        pool = (
            self.evidence_pool_branch(pooled).squeeze(-1).masked_fill(~valid, 0)
            if self.evidence_pool
            else zero
        )
        context = (
            self.evidence_context_branch(contrast)
            .squeeze(-1)
            .masked_fill(~valid | neighbors.eq(0), 0)
            if self.evidence_context
            else zero
        )
        raw = field.raw_evidence + pool + context
        evidence = centered_bounded(raw, valid)
        score = (field.score + (evidence - field.evidence)).masked_fill(~valid, 0)
        return EvidenceStructureOutput(
            **dict(field.__dict__, raw_evidence=raw, evidence=evidence, score=score),
            evidence_base_raw=field.raw_evidence,
            evidence_pool_delta=pool,
            evidence_context_delta=context,
            evidence_neighbor_count=neighbors.masked_fill(~valid, 0),
        )

    def score_without_evidence_branch(self, field, branch):
        if branch not in ("pool", "context"):
            raise ValueError(branch)
        raw = field.raw_evidence - getattr(field, "evidence_" + branch + "_delta")
        evidence = centered_bounded(raw, field.valid)
        return self.compose_score(field, {"evidence": evidence})


from .interaction import RawInteraction
from .primitives import OneStepLatentComposition, edge_span_mean
from .e_readout_scale import scale_e_readout_input
from .candidate_primitives import IndependentSProjectionEncoder, log_mean_exp_span


class Round31FieldScoreHeads(_LegacyFieldScoreHeads):
    """Round31's deployed E/S/T fields.

    The legacy path is kept at the tensor-operation level so ``m0`` has the
    Round21/l0 numerical contract. A/B/C modules are constructed only when
    enabled, which also keeps the control model's parameter set unchanged.
    """

    def __init__(
        self,
        *args,
        learned_e: bool = True,
        learned_t: bool = True,
        projected_cosine_attention: bool = False,
        local_top_quarter_e: bool = False,
        difference_before_interaction_t: bool = False,
        a_local_evidence: bool = False,
        e_readout_scale: str = "raw",
        e_span_lme: bool = False,
        independent_s_projection: bool = False,
        support_length_center: bool = False,
        b_edge_support: bool = False,
        c_latent_composition: bool = False,
        support_deployment: str = "in_score",
        support_input: str = "rms",
        support_pair_mode: str = "plain",
        support_readout: str = "edge_mean",
        support_candidate_chunk_size: int = 128,
        rank_stop_s: bool = False,
        aux_detach_input: bool = False,
        interaction_depth: int = 1,
        interaction_feedforward: bool = False,
        interaction_dropout: float = 0.0,
        edge_head_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.learned_e = bool(learned_e)
        self.learned_t = bool(learned_t)
        self.projected_cosine_attention = bool(projected_cosine_attention)
        # Legacy factors are fixed off in Round31; attributes remain for
        # contract/probe compatibility.
        self.local_top_quarter_e = bool(local_top_quarter_e)
        self.difference_before_interaction_t = bool(difference_before_interaction_t)
        self.a_local_evidence = bool(a_local_evidence)
        if e_readout_scale not in ("raw", "fixed_init", "query_rms", "token_rms"):
            raise ValueError(
                "e_readout_scale must be raw, fixed_init, query_rms, or token_rms"
            )
        self.e_readout_scale = str(e_readout_scale)
        self.e_span_lme = bool(e_span_lme)
        self.independent_s_projection = bool(independent_s_projection)
        self.support_length_center = bool(support_length_center)
        if self.independent_s_projection and not self.learned_e:
            raise ValueError("independent_s_projection requires learned_e")
        self.b_edge_support = bool(b_edge_support)
        self.c_latent_composition = bool(c_latent_composition)
        if support_deployment not in ("in_score", "zero"):
            raise ValueError("support_deployment must be in_score or zero")
        if support_input not in ("raw", "rms"):
            raise ValueError("support_input must be raw or rms")
        if support_pair_mode not in ("plain", "joint_residual"):
            raise ValueError("support_pair_mode must be plain or joint_residual")
        self.support_deployment = str(support_deployment)
        self.support_input = str(support_input)
        self.support_pair_mode = str(support_pair_mode)
        if support_readout != "edge_mean":
            raise ValueError("Round31 fixes support_readout=edge_mean")
        if int(support_candidate_chunk_size) != 128:
            raise ValueError("Round31 fixes support_candidate_chunk_size=128")
        self.support_readout = str(support_readout)
        self.support_candidate_chunk_size = int(support_candidate_chunk_size)
        self.rank_stop_s = bool(rank_stop_s)
        self.aux_detach_input = bool(aux_detach_input)
        self.interaction_depth = int(interaction_depth)
        self.interaction_feedforward = bool(interaction_feedforward)
        self.interaction_dropout = float(interaction_dropout)
        self.edge_head_dropout = float(edge_head_dropout)

        if self.learned_e:
            self.e_interaction = RawInteraction(
                1802,
                projected_cosine_attention=projected_cosine_attention,
                depth=self.interaction_depth,
                feedforward=self.interaction_feedforward,
                dropout=self.interaction_dropout,
            )
            if self.independent_s_projection:
                self.s_projection = IndependentSProjectionEncoder(self.e_interaction)
        if self.learned_t:
            self.t_interaction = RawInteraction(
                1804,
                transition=True,
                projected_cosine_attention=projected_cosine_attention,
                depth=self.interaction_depth,
                feedforward=self.interaction_feedforward,
                dropout=self.interaction_dropout,
            )
        if self.b_edge_support:
            # CPU-only initialization inside a CPU fork leaves the caller's
            # CPU and CUDA RNG streams untouched.
            with torch.random.fork_rng(devices=[]):
                install_cpu_generator_state(2206)
                self.edge_head = torch.nn.Sequential(
                    torch.nn.Linear(256, 32),
                    torch.nn.GELU(),
                    torch.nn.Dropout(self.edge_head_dropout),
                    torch.nn.Linear(32, 1),
                )
                torch.nn.init.zeros_(self.edge_head[-1].weight)
                torch.nn.init.zeros_(self.edge_head[-1].bias)
        if self.c_latent_composition:
            self.composition = OneStepLatentComposition(64)

    @staticmethod
    def _token_valid(valid, video_padding_mask):
        if video_padding_mask is not None:
            return ~video_padding_mask.bool()
        return valid.any(-1) | valid.any(-2)

    def _prepare_support_input(self, h):
        """Prepare the endpoint input for the S edge head and return RMS telemetry."""
        with torch.autocast(device_type=h.device.type, enabled=False):
            hf = h.float()
            before = torch.sqrt(hf.square().mean(dim=-1) + 1.0e-6)
            if self.support_input == "raw":
                return hf, before, before
            normalized = hf / before.unsqueeze(-1)
            after = torch.sqrt(normalized.square().mean(dim=-1) + 1.0e-6)
            return normalized, before, after

    @staticmethod
    def _edge_pair_features(left, right):
        return torch.cat((left, right, right - left, left * right), dim=-1)

    def _edge_head_raw(self, left, right):
        pair = self._edge_pair_features(left.float(), right.float())
        return self.edge_head(pair).squeeze(-1).float()

    def _edge_raw_components_from_prepared_pairs(
        self, left, right, *, include_references=False
    ):
        """Return g(a,b) and the configured edge composition in FP32.

        left and right are already passed through the shared S input helper.
        Joint residual mode evaluates all four references with the same head
        and subtracts additive endpoint terms before the sole tanh. Endpoint
        zeros therefore live in post-RMS feature space and no detached or
        surrogate path is introduced.
        """
        if left.shape != right.shape or left.shape[-1] != 64:
            raise ValueError("edge endpoints must have identical [...,64] shape")
        # Keep the plain operation graph identical to the Round23 path.
        # In particular, repeated endpoint slices preserve the established
        # gradient accumulation order used by the exact control anchors.
        if self.support_pair_mode == "plain" and not include_references:
            pair = torch.cat((left, right, right - left, left * right), dim=-1)
            g_ab = self.edge_head(pair).squeeze(-1)
            return {
                "raw": g_ab,
                "g_ab": g_ab,
                "g_a0": None,
                "g_0b": None,
                "g_00": None,
            }
        left, right = left.float(), right.float()
        g_ab = self._edge_head_raw(left, right)
        zero_left = torch.zeros_like(left)
        zero_right = torch.zeros_like(right)
        g_a0 = self._edge_head_raw(left, zero_right)
        g_0b = self._edge_head_raw(zero_left, right)
        g_00 = self._edge_head_raw(zero_left, zero_right)
        return {
            "raw": (
                g_ab if self.support_pair_mode == "plain" else g_ab - g_a0 - g_0b + g_00
            ),
            "g_ab": g_ab,
            "g_a0": g_a0,
            "g_0b": g_0b,
            "g_00": g_00,
        }

    def edge_raw_components_from_pairs(self, left, right, *, include_references=False):
        """Return raw edge references for probes and donor counterfactuals."""
        if not self.b_edge_support:
            zero = left.new_zeros(left.shape[:-1], dtype=torch.float32)
            return {
                "raw": zero,
                "g_ab": zero,
                "g_a0": None,
                "g_0b": None,
                "g_00": None,
            }
        if left.shape != right.shape or left.shape[-1] != 64:
            raise ValueError("edge endpoints must have identical [...,64] shape")
        with torch.autocast(device_type=left.device.type, enabled=False):
            left, _, _ = self._prepare_support_input(left)
            right, _, _ = self._prepare_support_input(right)
            if self.support_pair_mode == "plain" and not include_references:
                pair = torch.cat((left, right, right - left, left * right), dim=-1)
                raw = self.edge_head(pair).squeeze(-1)
                return {
                    "raw": raw,
                    "g_ab": raw,
                    "g_a0": None,
                    "g_0b": None,
                    "g_00": None,
                }
            return self._edge_raw_components_from_prepared_pairs(
                left, right, include_references=include_references
            )

    def edge_score_from_pairs(self, left, right):
        """Bounded edge values for arbitrary endpoint rows.

        Donor supervision calls this endpoint-sized function and never builds
        a B*L*L*L tensor. Both endpoints use the same S input helper.
        """
        if self.support_readout != "edge_mean":
            if not self.b_edge_support:
                return left.new_zeros(left.shape[:-1], dtype=torch.float32)
            if left.shape != right.shape or left.shape[-1] != 64:
                raise ValueError(
                    "support pair endpoints must have identical [...,64] shape"
                )
            with torch.autocast(device_type=left.device.type, enabled=False):
                left, _, _ = self._prepare_support_input(left)
                right, _, _ = self._prepare_support_input(right)
                raw = self._support_readout_raw_pair(left, right)
                return torch.tanh(raw.float())
        if self.support_pair_mode == "plain":
            if not self.b_edge_support:
                return left.new_zeros(left.shape[:-1], dtype=torch.float32)
            if left.shape != right.shape or left.shape[-1] != 64:
                raise ValueError("edge endpoints must have identical [...,64] shape")
            with torch.autocast(device_type=left.device.type, enabled=False):
                left, _, _ = self._prepare_support_input(left)
                right, _, _ = self._prepare_support_input(right)
                pair = torch.cat((left, right, right - left, left * right), dim=-1)
                raw = self.edge_head(pair).squeeze(-1)
                return torch.tanh(raw)
        components = self.edge_raw_components_from_pairs(left, right)
        return torch.tanh(components["raw"].float())

    def support_score_from_pairs(self, left, right):
        """Unified clean/donor length-two S dispatcher used by loss code."""
        return self.edge_score_from_pairs(left, right)

    def _e_span(self, token_values, token_valid, valid):
        """Reduce local E tokens on the deployed candidate grid."""
        if token_values.ndim == 3 and token_values.shape[-1] == 1:
            token_values = token_values.squeeze(-1)
        if token_values.ndim != 2 or token_valid.shape != token_values.shape:
            raise ValueError("local E tokens and token-valid mask must be [B,L]")
        valid = valid.bool()
        if self.e_span_lme:
            pooled, span_valid = log_mean_exp_span(token_values, token_valid)
            if pooled.shape != valid.shape:
                raise ValueError(
                    "local E reduction shape does not match candidate grid"
                )
            return pooled.masked_fill(~(span_valid & valid), 0)
        # Preserve the established candidate-mean arithmetic for the control
        # path; only the selected reduction changes in LME variants.
        return span_mean(token_values[..., None]).squeeze(-1).masked_fill(~valid, 0)

    def _edge_field(self, h, token_valid, valid):
        bsz, length, _ = h.shape
        support_h, rms_before, rms_after = self._prepare_support_input(h)
        if length <= 1:
            empty = h.new_zeros((bsz, 0), dtype=torch.float32)
            empty_parts = {
                name: empty for name in ("raw", "g_ab", "g_a0", "g_0b", "g_00")
            }
            return (
                empty,
                empty,
                h.new_zeros((bsz, length, length), dtype=torch.float32),
                h.new_zeros((bsz, length, length), dtype=torch.bool),
                rms_before,
                rms_after,
                empty_parts,
                h.new_zeros((bsz, length, length), dtype=torch.float32),
                support_h,
            )
        with torch.autocast(device_type=h.device.type, enabled=False):
            hf = support_h
            if self.support_pair_mode == "plain":
                # This is intentionally the original Round23 expression.
                pair = torch.cat(
                    (
                        hf[:, :-1],
                        hf[:, 1:],
                        hf[:, 1:] - hf[:, :-1],
                        hf[:, :-1] * hf[:, 1:],
                    ),
                    dim=-1,
                )
                raw = self.edge_head(pair).squeeze(-1)
                components = {
                    "raw": raw,
                    "g_ab": raw,
                    "g_a0": None,
                    "g_0b": None,
                    "g_00": None,
                }
                bounded = torch.tanh(raw)
            else:
                components = self._edge_raw_components_from_prepared_pairs(
                    hf[:, :-1], hf[:, 1:]
                )
                raw = components["raw"]
                bounded = torch.tanh(raw.float())
        edge_span, primitive_valid = edge_span_mean(bounded, token_valid)
        raw_span, _ = edge_span_mean(raw, token_valid)
        span_valid = primitive_valid & valid.bool()
        return (
            bounded,
            raw,
            edge_span.masked_fill(~span_valid, 0),
            span_valid,
            rms_before,
            rms_after,
            components,
            raw_span,
            support_h,
        )

    def _candidate_raw_from_prepared_pairs(self, support_h, token_prefix, starts, ends):
        """Return raw joint residuals for one candidate chunk.

        support_h is already token-RMS prepared.  Prefix sums are only over
        token vectors; no B*C*L tensor is materialized.
        """
        widths = ends - starts + 1
        safe_widths = widths.clamp_min(2)
        if self.support_readout == "pooled":
            sums = token_prefix[:, ends + 1] - token_prefix[:, starts]
            pooled = sums / safe_widths.to(dtype=support_h.dtype).view(1, -1, 1)
            return self._edge_raw_components_from_prepared_pairs(pooled, pooled)["raw"]

        split = safe_widths // 2
        left_end = starts + split
        right_end = ends + 1
        left_sum = token_prefix[:, left_end] - token_prefix[:, starts]
        right_sum = token_prefix[:, right_end] - token_prefix[:, left_end]
        left = left_sum / split.to(dtype=support_h.dtype).view(1, -1, 1)
        right_count = (safe_widths - split).to(dtype=support_h.dtype)
        right = right_sum / right_count.view(1, -1, 1)
        raw_lr = self._edge_raw_components_from_prepared_pairs(left, right)["raw"]
        if self.support_readout == "ordered_halves":
            return raw_lr
        if self.support_readout == "unordered_halves":
            raw_rl = self._edge_raw_components_from_prepared_pairs(right, left)["raw"]
            return 0.5 * (raw_lr + raw_rl)
        raise AssertionError(f"unhandled support_readout {self.support_readout!r}")

    def _support_readout_raw_pair(self, left, right):
        """Raw candidate dispatcher for a length-two clean or donor pair."""
        if left.shape != right.shape or left.shape[-1] != 64:
            raise ValueError(
                "support pair endpoints must have identical [...,64] shape"
            )
        if self.support_readout == "edge_mean":
            return self._edge_raw_components_from_prepared_pairs(left, right)["raw"]
        pooled = 0.5 * (left + right)
        if self.support_readout == "pooled":
            return self._edge_raw_components_from_prepared_pairs(pooled, pooled)["raw"]
        raw_lr = self._edge_raw_components_from_prepared_pairs(left, right)["raw"]
        if self.support_readout == "ordered_halves":
            return raw_lr
        if self.support_readout == "unordered_halves":
            raw_rl = self._edge_raw_components_from_prepared_pairs(right, left)["raw"]
            return 0.5 * (raw_lr + raw_rl)
        raise AssertionError(f"unhandled support_readout {self.support_readout!r}")

    @staticmethod
    def _round31_full_candidate_grid(pair_values, pair_mask, length, reference):
        """Expand start-major triangular pairs into the full B x L x L grid."""
        if length <= 0:
            raise ValueError("empty support token sequence")
        rows = []
        mask_rows = []
        bsz = reference.shape[0]
        zero_column = reference[:, 0].sum(dim=-1) * 0.0
        offset = 0
        for start in range(length):
            row_length = length - start
            values = pair_values[:, offset : offset + row_length]
            rows.append(
                torch.cat(
                    (
                        zero_column.unsqueeze(1).expand(bsz, start),
                        values,
                    ),
                    dim=1,
                )
            )
            masks = pair_mask[:, offset : offset + row_length]
            mask_rows.append(
                torch.cat(
                    (
                        torch.zeros(
                            (bsz, start), dtype=torch.bool, device=reference.device
                        ),
                        masks,
                    ),
                    dim=1,
                )
            )
            offset += row_length
        return torch.stack(rows, dim=1), torch.stack(mask_rows, dim=1)

    def _candidate_readout_field(self, h, token_valid, valid):
        """Build pooled/half candidate S while preserving field membership."""
        bsz, length, _ = h.shape
        support_h, rms_before, rms_after = self._prepare_support_input(
            h.masked_fill(~token_valid[..., None], 0)
        )
        if length <= 0:
            raise ValueError("empty support token sequence")
        safe_h = support_h.float().masked_fill(~token_valid[..., None], 0)
        with torch.autocast(device_type=h.device.type, enabled=False):
            if length >= 2:
                edge_components = self._edge_raw_components_from_prepared_pairs(
                    safe_h[:, :-1], safe_h[:, 1:]
                )
                edge_raw = edge_components["raw"]
                edge_bounded = torch.tanh(edge_raw.float())
                edge_valid = token_valid[:, :-1] & token_valid[:, 1:]
                edge_raw = edge_raw.masked_fill(~edge_valid, 0)
                edge_bounded = edge_bounded.masked_fill(~edge_valid, 0)
            else:
                edge_raw = safe_h.new_zeros((bsz, 0), dtype=torch.float32)
                edge_bounded = edge_raw
                edge_valid = token_valid.new_zeros((bsz, 0))
                edge_components = {
                    name: edge_raw for name in ("raw", "g_ab", "g_a0", "g_0b", "g_00")
                }

            token_prefix = torch.cat(
                (
                    safe_h.new_zeros((bsz, 1, 64)),
                    safe_h.cumsum(dim=1),
                ),
                dim=1,
            )
            starts_list = []
            ends_list = []
            for start in range(length):
                for end in range(start, length):
                    starts_list.append(start)
                    ends_list.append(end)
            starts = torch.tensor(starts_list, device=h.device, dtype=torch.long)
            ends = torch.tensor(ends_list, device=h.device, dtype=torch.long)
            widths = ends - starts + 1
            valid_prefix = torch.cat(
                (
                    token_valid.new_zeros((bsz, 1), dtype=torch.long),
                    token_valid.to(dtype=torch.long).cumsum(dim=1),
                ),
                dim=1,
            )
            valid_counts = valid_prefix[:, ends + 1] - valid_prefix[:, starts]
            all_tokens_valid = valid_counts == widths.view(1, -1)
            membership_valid = valid[:, starts, ends].bool()
            pair_valid = all_tokens_valid & membership_valid

            value_chunks = []
            raw_chunks = []
            mask_chunks = []
            chunk = int(self.support_candidate_chunk_size)
            for offset in range(0, starts.numel(), chunk):
                sl = slice(offset, min(offset + chunk, starts.numel()))
                chunk_starts, chunk_ends = starts[sl], ends[sl]
                raw = self._candidate_raw_from_prepared_pairs(
                    safe_h, token_prefix, chunk_starts, chunk_ends
                )
                bounded = torch.tanh(raw.float())
                mask = pair_valid[:, sl]
                nonneutral = mask & (widths[sl] > 1).unsqueeze(0)
                value_chunks.append(
                    torch.where(nonneutral, bounded, torch.zeros_like(bounded))
                )
                raw_chunks.append(torch.where(nonneutral, raw, torch.zeros_like(raw)))
                mask_chunks.append(mask)
            pair_values = torch.cat(value_chunks, dim=1)
            pair_raw = torch.cat(raw_chunks, dim=1)
            pair_mask = torch.cat(mask_chunks, dim=1)
            support, span_valid = self._round31_full_candidate_grid(
                pair_values, pair_mask, length, safe_h
            )
            raw_span, _ = self._round31_full_candidate_grid(
                pair_raw, pair_mask, length, safe_h
            )
            return (
                edge_bounded,
                edge_raw,
                support.masked_fill(~valid.bool(), 0),
                span_valid,
                rms_before,
                rms_after,
                edge_components,
                raw_span.masked_fill(~valid.bool(), 0),
                edge_valid,
                support_h,
            )

    def raw_from_state(self, state, valid, carrier, video_padding_mask=None):
        # Exact inherited Round21 base/readout path before Round31 replaces a
        # deployed field.
        field = super().raw_from_state(
            state, valid, carrier, video_padding_mask=video_padding_mask
        )
        raw_inputs = getattr(state, "round31_raw_inputs", None)
        if raw_inputs is None:
            raise RuntimeError("actual forward raw inputs missing")
        v, t, pad, tpad = raw_inputs
        # Preserve the exact tensors used by this field forward for the
        # production route probe. These are metadata references only; no
        # detached copy or additional arithmetic is introduced here.
        field.round31_raw_inputs = raw_inputs
        token_valid = self._token_valid(valid, video_padding_mask)
        old_e = field.evidence
        old_support = field.support
        old_ts = field.transition_start
        old_te = field.transition_end
        field.round31_legacy_evidence = old_e
        field.round31_legacy_raw_evidence = field.raw_evidence
        field.round31_legacy_raw_ts = field.raw_transition_start
        field.round31_legacy_raw_te = field.raw_transition_end

        if self.learned_e:
            z_e = self.e_interaction.encode(v, t, pad, tpad)
            # Keep the original z for S/T and expose it unchanged.  Only the
            # E readout input passes through the configured parameter-free
            # operator; wrong-query forwards take this same path.
            e_readout_input, e_readout_telemetry = scale_e_readout_input(
                z_e, token_valid, self.e_readout_scale
            )
            raw_local = self.e_interaction.readout(e_readout_input).squeeze(-1)
        else:
            z_e = v.new_zeros((*v.shape[:2], 64), dtype=torch.float32)
            e_readout_input, e_readout_telemetry = scale_e_readout_input(
                z_e, token_valid, self.e_readout_scale
            )
            raw_local = z_e.new_zeros(z_e.shape[:2])
        field.round31_z_e = z_e
        if self.independent_s_projection:
            z_s = self.s_projection.encode(v, t, pad, tpad)
        else:
            z_s = z_e
        field.round31_z_s = z_s
        field.round31_s_projection_mode = (
            "independent" if self.independent_s_projection else "shared_e"
        )
        field.round31_e_span_reduction = (
            "log_mean_exp" if self.e_span_lme else "candidate_mean"
        )
        field.round31_e_span_temperature = 0.2 if self.e_span_lme else None
        field.round31_factor_state = {
            "e_span_lme": self.e_span_lme,
            "independent_s_projection": self.independent_s_projection,
            "s_input_source": "round31_z_s",
            "rank_stop_s": self.rank_stop_s,
            "aux_detach_input": self.aux_detach_input,
        }
        field.round31_e_readout_input = e_readout_input
        field.round31_e_readout_gain = e_readout_telemetry["gain_tensor"]
        field.round31_e_readout_denominator = e_readout_telemetry["denominator"]
        field.round31_e_readout_mode = self.e_readout_scale
        # Keep this per-forward record metadata-only.  Numeric quantiles/RMS
        # are collected by the fixed-panel probe from these graph-connected
        # tensors; doing reductions and .item() here would add GPU sync cost.
        field.round31_e_readout_probe = {
            "mode": self.e_readout_scale,
            "epsilon": e_readout_telemetry["epsilon"],
        }
        field.round31_raw_local_token = raw_local

        edge_bounded = edge_raw = edge_span = edge_valid = None
        if self.b_edge_support:
            if self.support_readout == "edge_mean":
                (
                    edge_bounded,
                    edge_raw,
                    edge_span,
                    edge_valid,
                    support_rms_before,
                    support_rms_after,
                    edge_components,
                    edge_raw_span,
                    support_prepared,
                ) = self._edge_field(z_s, token_valid, valid)
            else:
                (
                    edge_bounded,
                    edge_raw,
                    edge_span,
                    edge_valid,
                    support_rms_before,
                    support_rms_after,
                    edge_components,
                    edge_raw_span,
                    _edge_token_valid,
                    support_prepared,
                ) = self._candidate_readout_field(z_s, token_valid, valid)
            field.round31_edge_score = edge_bounded
            field.round31_edge_raw = edge_raw
            field.round31_edge_valid = token_valid[:, :-1] & token_valid[:, 1:]
            field.round31_edge_span_valid = edge_valid
            field.round31_support_input_rms_before = support_rms_before
            field.round31_support_input_rms_after = support_rms_after
            field.round31_edge_decomposition = edge_components
            field.round31_edge_raw_span = edge_raw_span
            field.round31_support_prepared = support_prepared
            field.round31_support_readout = self.support_readout
            field.round31_support_candidate_chunk_size = (
                self.support_candidate_chunk_size
            )
            field.round31_support_span_valid = edge_valid
            field.round31_support_score = edge_span
            field.round31_support_raw_score = edge_raw_span
            field.round31_support_edge_reference = edge_bounded
        else:
            field.round31_support_prepared = None
            field.round31_support_readout = self.support_readout
            field.round31_support_candidate_chunk_size = (
                self.support_candidate_chunk_size
            )
            field.round31_support_span_valid = None
            field.round31_support_score = None
            field.round31_support_raw_score = None
            field.round31_support_edge_reference = None
            field.round31_edge_score = None
            field.round31_edge_raw = None
            field.round31_edge_valid = None
            field.round31_support_input_rms_before = None
            field.round31_support_input_rms_after = None
            field.round31_edge_decomposition = None
            field.round31_edge_raw_span = None

        # B1 keeps the deployed S value and forward graph unchanged, but gives
        # the auxiliary support loss a second readout whose input is detached.
        # It uses the same edge_mean helper, mask, RMS epsilon, and head; no
        # parent/backbone forward or random operation is introduced.
        if self.aux_detach_input and self.b_edge_support:
            aux = self._edge_field(z_s.detach(), token_valid, valid)
            field.round31_h_aux = z_s.detach()
            field.round31_aux_edge_score = aux[0]
            field.round31_aux_edge_raw = aux[1]
            field.round31_aux_support = aux[2]
            field.round31_aux_support_valid = aux[3]
            field.round31_aux_support_prepared = aux[8]
            field.round31_aux_support_requires_grad = bool(aux[2].requires_grad)
        else:
            field.round31_h_aux = None
            field.round31_aux_edge_score = None
            field.round31_aux_edge_raw = None
            field.round31_aux_support = None
            field.round31_aux_support_valid = None
            field.round31_aux_support_prepared = None
            field.round31_aux_support_requires_grad = False

        composition_info = None
        if self.c_latent_composition:
            h_prime, composition_info = self.composition(
                z_e,
                token_valid,
                support=edge_bounded if self.b_edge_support else None,
            )
        else:
            h_prime = z_e
        # C is fixed off in the Round31 matrix, but apply the E operator to
        # the actual post-composition input as well so every E branch shares
        # one deployed transform if a diagnostic constructs C explicitly.
        if h_prime is z_e:
            e_readout_input_prime, e_readout_telemetry_prime = (
                e_readout_input,
                e_readout_telemetry,
            )
        else:
            e_readout_input_prime, e_readout_telemetry_prime = scale_e_readout_input(
                h_prime, token_valid, self.e_readout_scale
            )
        field.round31_e_readout_input_prime = e_readout_input_prime
        field.round31_e_readout_gain_prime = e_readout_telemetry_prime["gain_tensor"]
        field.round31_e_readout_denominator_prime = e_readout_telemetry_prime[
            "denominator"
        ]
        # The support loss and donor counterfactuals consume the exact S
        # encoder output.  In independent variants this must not alias E.
        field.round31_h = z_s
        field.round31_h_prime = h_prime
        field.round31_composition = composition_info
        if composition_info is not None:
            field.round31_connection = (
                (edge_bounded.float() + 1.0) / 2.0
                if self.b_edge_support
                else torch.ones_like(composition_info["latent_gate"])
            )
        else:
            field.round31_connection = None

        # A off is the legacy candidate-E path with the (possibly composed)
        # readout. A on removes the legacy candidate-E contribution and deploys
        # the mean of bounded local values.
        if self.a_local_evidence:
            raw_local_prime = self.e_interaction.readout(e_readout_input_prime).squeeze(
                -1
            )
            local_bounded = torch.tanh(raw_local_prime.float())
            deployed_e = self._e_span(local_bounded, token_valid, valid)
            raw_e_span = self._e_span(raw_local_prime, token_valid, valid)
            field.round31_e_token = local_bounded
            field.round31_raw_e_token = raw_local_prime
            field.round31_raw_e_span = raw_e_span
            field.round31_deployed_e = deployed_e.masked_fill(~valid, 0)
            field.raw_evidence = raw_e_span.masked_fill(~valid, 0)
            field.evidence = field.round31_deployed_e
        else:
            readout_prime = self.e_interaction.readout(e_readout_input_prime)
            delta = self._e_span(readout_prime, token_valid, valid)
            field.round31_e_token = readout_prime.squeeze(-1)
            field.round31_raw_e_span = delta
            field.round31_deployed_e = field.round31_legacy_raw_evidence + delta
            field.raw_evidence = field.round31_deployed_e
            field.evidence = centered_bounded(field.raw_evidence, valid)
        field.round31_e_no_composition = (
            self._e_span(torch.tanh(raw_local), token_valid, valid)
            if self.a_local_evidence
            else centered_bounded(
                field.round31_legacy_raw_evidence
                + self._e_span(raw_local, token_valid, valid),
                valid,
            )
        ).masked_fill(~valid, 0)

        # Exterior T remains the Round21 path and does not receive C messages.
        if self.learned_t:
            if self.difference_before_interaction_t:
                from torch.nn import functional as F

                vl, vr = RawInteraction.differences(
                    F.normalize(v.float(), dim=-1, eps=1e-6), pad
                )
                left = self.t_interaction.encode_difference(vl, t, pad, tpad)
                right = self.t_interaction.encode_difference(vr, t, pad, tpad)
                z_t = 0.5 * (left + right)
            else:
                z_t = self.t_interaction.encode(v, t, pad, tpad)
                left, right = RawInteraction.differences(z_t, pad)
            field.round31_t_left = left
            field.round31_t_right = right
            ds = self.t_interaction.readout(left).squeeze(-1)[:, :, None]
            ds = ds.expand_as(valid).masked_fill(~valid, 0)
            de = self.t_interaction.end_readout(right).squeeze(-1)[:, None, :]
            de = de.expand_as(valid).masked_fill(~valid, 0)
            field.raw_transition_start = field.round31_legacy_raw_ts + ds
            field.raw_transition_end = field.round31_legacy_raw_te + de
            field.transition_start = torch.tanh(
                field.raw_transition_start.float()
            ).masked_fill(~valid, 0)
            field.transition_end = torch.tanh(
                field.raw_transition_end.float()
            ).masked_fill(~valid, 0)
            field.round31_z_t = z_t
            field.round31_delta_ts = ds
            field.round31_delta_te = de

        # Preserve the inherited arithmetic order for the in-score anchors.
        # First form the post-E/T score that still contains old support. The
        # exact zero score removes that same old support tensor; support itself
        # remains available to the auxiliary edge objective.
        field.score = (
            field.score
            + (field.evidence - old_e)
            + 0.5
            * ((field.transition_start - old_ts) + (field.transition_end - old_te))
        ).masked_fill(~valid, 0)
        score_with_old_support = field.score
        score_without_support = (score_with_old_support - old_support).masked_fill(
            ~valid, 0
        )
        if self.b_edge_support:
            field.raw_support = (
                edge_span_mean(edge_raw, token_valid)[0].masked_fill(~valid, 0)
                if self.support_readout == "edge_mean"
                and not getattr(self, "r48_singleton", False)
                else edge_raw_span
            )
            field.round31_uncentered_support = edge_span
            field.round31_support_centering = None
            if self.support_length_center:
                from .candidate_pair_loss import center_support_by_length

                edge_span, center_detail = center_support_by_length(edge_span, valid)
                field.round31_support_centering = center_detail
            field.support = edge_span
            score_with_support = (
                score_with_old_support + (field.support - old_support)
            ).masked_fill(~valid, 0)
        else:
            score_with_support = score_with_old_support
        field.round31_support_deployment = self.support_deployment
        field.round31_support_input = self.support_input
        field.round31_support_pair_mode = self.support_pair_mode
        field.round31_deployed_support = (
            field.support
            if self.support_deployment == "in_score"
            else torch.zeros_like(field.support)
        )
        field.score = (
            score_with_support
            if self.support_deployment == "in_score"
            else score_without_support
        )
        # A1 changes only the rank-loss autograd graph.  Detaching the new S
        # term preserves the exact deployed value while preventing K/P from
        # reaching edge_head or s_projection.  Pair selection still consumes
        # field.score in losses.py, so hard IDs and masks remain deployment F.
        field.round31_score_with_old_support = score_with_old_support
        field.round31_old_support = old_support
        field.round31_score_without_support = score_without_support
        field.round31_score_with_support = score_with_support
        if self.rank_stop_s and self.support_deployment == "in_score":
            rank_score = (
                score_with_old_support + (field.support.detach() - old_support)
            ).masked_fill(~valid, 0)
            field.round31_rank_score_source = "same_F_arithmetic_with_detached_S"
        else:
            rank_score = field.score
            field.round31_rank_score_source = "deployed_F"
        field.round31_rank_score = rank_score
        field.round31_rank_stop_s = self.rank_stop_s
        field.round31_aux_detach_input = self.aux_detach_input
        if self.c_latent_composition:
            if self.a_local_evidence:
                e0 = self._e_span(torch.tanh(raw_local), token_valid, valid)
            else:
                e0 = centered_bounded(
                    field.round31_legacy_raw_evidence
                    + self._e_span(raw_local, token_valid, valid),
                    valid,
                )
            field.round31_score_without_composition = (
                field.carrier.float()
                + e0.float()
                + field.support.float()
                + 0.5 * (field.transition_start.float() + field.transition_end.float())
            ).masked_fill(~valid, 0)
        else:
            field.round31_score_without_composition = field.score
        _pre_residual_field = field
        field = self.apply_residual(
            field
        )  # round36 fix: wire up linear/pair/length residuals
        if field is not _pre_residual_field:
            # round36 fix: preserve ad-hoc round31_* attributes across replace()
            for _k, _v in _pre_residual_field.__dict__.items():
                if _k not in field.__dict__:
                    setattr(field, _k, _v)
        return field

    def compose_score(
        self,
        field,
        overrides=None,
        disable_linear=False,
        disable_pair=False,
        *,
        support_deployment=None,
    ):
        mode = (
            self.support_deployment
            if support_deployment is None
            else str(support_deployment)
        )
        if mode not in ("in_score", "zero"):
            raise ValueError("support_deployment must be in_score or zero")
        local = {} if overrides is None else dict(overrides)
        # A zero-deployment score must stay S-free even when a field/shuffle
        # override supplies a support tensor.  The only way to ask for an
        # independent in-score recomposition is support_deployment="in_score".
        if mode == "zero":
            local["support"] = torch.zeros_like(field.support)
        return super().compose_score(
            field,
            local,
            disable_linear=disable_linear,
            disable_pair=disable_pair,
        )

    def score_without_interaction(self, field, branch):
        if branch == "e":
            evidence = (
                torch.zeros_like(field.evidence)
                if self.a_local_evidence
                else centered_bounded(field.round31_legacy_raw_evidence, field.valid)
            )
            return self.compose_score(field, {"evidence": evidence})
        if branch == "t":
            return self.compose_score(
                field,
                {
                    "transition_start": torch.tanh(
                        field.round31_legacy_raw_ts.float()
                    ).masked_fill(~field.valid, 0),
                    "transition_end": torch.tanh(
                        field.round31_legacy_raw_te.float()
                    ).masked_fill(~field.valid, 0),
                },
            )
        raise ValueError(branch)
