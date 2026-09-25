# File guide

The public entry points are `train.py` and `inference.py`. Model and loss code is separated by responsibility; checkpoints retain original tensor names.

| File | Responsibility |
|---|---|
| `backbone/__init__.py` | Package initializer and explicit exports; keep only required re-exports. |
| `backbone/conditioned/__init__.py` | Package initializer and explicit exports; keep only required re-exports. |
| `backbone/conditioned/conditioner_helpers.py` | Exact trainability and correlation helpers extracted from unused alternative model. |
| `backbone/conditioned/contracts.py` | Extension output/module interface used by scratch parent. |
| `backbone/conditioned/extensions.py` | Identity extension implementation used by scratch model. |
| `backbone/conditioned/model.py` | Live scratch backbone shell and trainability contract. |
| `backbone/conditioned/official_eval.py` | Compatibility entry point forwarding to the unified MR/HD evaluator. |
| `backbone/conditioned/paired_loader.py` | Independent train sampler and loader-worker RNG streams. |
| `backbone/conditioned/plugin.py` | Fixed scratch extension builder plus required checkpoint-reading helpers. |
| `backbone/conditioned/scratch_plugin.py` | Live scratch initialization and backbone subclass. |
| `backbone/conditioned/trifield_input_conditioner_plugin.py` | Live E/S/T input conditioner and parent model. |
| `backbone/model.py` | Live shared-span head and inherited encoder backbone. |
| `eventfieldnet/__init__.py` | Package initializer and explicit exports; keep only required re-exports. |
| `eventfieldnet/candidate_objective.py` | Candidate-level loss adjustment inherited by final model. |
| `eventfieldnet/candidate_pools.py` | Actual candidate-pool selection, mixed pools and model base. |
| `eventfieldnet/carrier_helpers.py` | Carrier feature/calibration helpers used by current rank path. |
| `eventfieldnet/coverage_loss.py` | S truncation hinge plus neutral expansion objective. |
| `eventfieldnet/dataset.py` | Fixed QV feature loading, GT padding and saliency target preservation. |
| `eventfieldnet/engine.py` | Fixed recipe optimizer, scheduler, epochs, checkpoint selection and RNG resume. |
| `eventfieldnet/evaluation.py` | Frozen-checkpoint MR inference and official VeryGood HD scorer. |
| `eventfieldnet/evaluation_cli.py` | Candidate decoding, metric preparation and checkpoint reconstruction. |
| `eventfieldnet/evaluator.py` | MR metric aliases, subprocess lifecycle and evaluator facade. |
| `eventfieldnet/field_objective.py` | Quality/field objective inherited by final model. |
| `eventfieldnet/gradient_routing.py` | Required local parameter enumeration; avoid deleting based on older class names. |
| `eventfieldnet/hd_evaluator.py` | FP32 subprocess evaluator facade selecting the MR+HD worker. |
| `eventfieldnet/highlight.py` | 448-to-64-to-1 highlight head and per-query balanced top16-negative BCE. |
| `eventfieldnet/joint_model.py` | Final complete E/S/T moment model plus shared/evidence fusion highlight head. |
| `eventfieldnet/joint_training.py` | Joint field/candidate loss integration inherited by final model. |
| `eventfieldnet/length_objective.py` | Duration helper currently imported by coverage/model code; alternatives can be pruned within file later. |
| `eventfieldnet/local_support.py` | Centered local support readout and fixed mixed-KL objective. |
| `eventfieldnet/local_support_helpers.py` | Uniform-null local support tensor operations. |
| `eventfieldnet/model/__init__.py` | Package initializer and explicit exports; keep only required re-exports. |
| `eventfieldnet/model/_rng.py` | Small CPU-only RNG helpers for deterministic local branches. |
| `eventfieldnet/model/adapter.py` | Round31 adapter around the reviewed reusable parent model. |
| `eventfieldnet/model/candidate_pair_loss.py` | Real-grid geometric competition, and GT-free same-length S centering. |
| `eventfieldnet/model/candidate_primitives.py` | Small, CPU-reviewable candidate mechanisms for Round31. |
| `eventfieldnet/model/counterfactual.py` | Round31 geometry-only counterfactual labels and graph-connected objectives. |
| `eventfieldnet/model/e_readout_scale.py` | Parameter-free input scaling used only by the Round31 E readout. |
| `eventfieldnet/model/interaction.py` | Field-model tensor helper: RawInteraction, local_e_mask, interaction_probe. |
| `eventfieldnet/model/local_counterfactual.py` | Round31 local interventions; labels only select existing candidate IDs. |
| `eventfieldnet/model/local_evidence_loss.py` | Candidate ordinal local evidence loss; explicit ratings only. |
| `eventfieldnet/model/losses.py` | Round31 five-term objective. |
| `eventfieldnet/model/mechanism_probe.py` | Field-model tensor helper: mechanism_probe. |
| `eventfieldnet/model/primitives.py` | CPU-reviewable candidate primitives; not imported by any training release. |
| `eventfieldnet/model/query_ordinal.py` | Candidate A loss primitive: anchored query-relative ordinal supervision. |
| `eventfieldnet/model/selector.py` | Field-model tensor helper: SupportStructureOutput, _SupportFieldScoreHeads, EvidenceStructureOutput, _LegacyFieldScoreHeads. |
| `eventfieldnet/model_factory.py` | Fixed complete-field loss composition and rank-only support-gradient routing. |
| `eventfieldnet/optim.py` | AdamW groups with exact trainable-parameter coverage. |
| `eventfieldnet/precision.py` | FP32/no-TF32 precision configuration and receipts. |
| `eventfieldnet/precision_evaluation.py` | Enforce actual no-autocast/no-TF32 inference context. |
| `eventfieldnet/quality_calibration.py` | Actual score/quality calibration model and selector. |
| `eventfieldnet/query_objective.py` | Candidate objective building blocks and fixed query adaptation base. |
| `eventfieldnet/score_calibration.py` | Bounded carrier/field score composition base. |
| `eventfieldnet/support_geometry.py` | Anchor selection, local support geometry and actual model/selector base. |
| `eventfieldnet/support_objective.py` | Support supervision/model initialization inherited by final model. |
| `eventfieldnet/validation.py` | Force strict FP32 evaluation requests without changing other request fields. |
| `external/standalone_eval/__init__.py` | Package initializer and explicit exports; keep only required re-exports. |
| `external/standalone_eval/eval.py` | Pinned official Moment-DETR MR/HD evaluation utility; keep source unchanged. |
| `external/standalone_eval/utils.py` | Pinned official Moment-DETR MR/HD evaluation utility; keep source unchanged. |
| `field_core/__init__.py` | Package initializer and explicit exports; keep only required re-exports. |
| `field_core/adapter.py` | Live foundational field wrapper and batch/metadata helpers. |
| `field_core/losses.py` | Live geometric targets and primitive field loss functions. |
| `field_core/selector.py` | Live primitive E/S/T heads and pooling. |
| `field_primitives/__init__.py` | Package initializer and explicit exports; keep only required re-exports. |
| `field_primitives/model/__init__.py` | Package initializer and explicit exports; keep only required re-exports. |
| `field_primitives/model/adapter.py` | Round1 adapter around the reviewed reusable parent model. |
| `field_primitives/model/losses.py` | Configurable round1 losses built on the V1 geometry contract. |
| `field_primitives/model/selector.py` | Round1 E/S/T heads built on the V1 selector contract. |
| `inference.py` | Public frozen-checkpoint validation CLI; test split not yet exposed. |
| `scripts/event_field_official_eval.py` | Reduced SG-derived get_metrics/tensor conversion API loaded dynamically by MR worker. |
| `sg_components/__init__.py` | Package initializer and explicit exports; keep only required re-exports. |
| `sg_components/dataset/__init__.py` | Package initializer and explicit exports; keep only required re-exports. |
| `sg_components/dataset/collate.py` | SG-derived data/collation operations; preserve feature/time-axis/target handling. |
| `sg_components/dataset/qvhighlights.py` | SG-derived data/collation operations; preserve feature/time-axis/target handling. |
| `sg_components/dataset/utils.py` | SG-derived data/collation operations; preserve feature/time-axis/target handling. |
| `sg_components/metrics/__init__.py` | Package initializer and explicit exports; keep only required re-exports. |
| `sg_components/metrics/highlights/__init__.py` | Package initializer and explicit exports; keep only required re-exports. |
| `sg_components/metrics/highlights/avg_precision.py` | SG-derived highlight metrics; preserve numerical definitions and labels. |
| `sg_components/metrics/highlights/hit1.py` | SG-derived highlight metrics; preserve numerical definitions and labels. |
| `sg_components/metrics/highlights/metrics.py` | SG-derived highlight metrics; preserve numerical definitions and labels. |
| `sg_components/metrics/metrics_collection.py` | SG-derived metric factory; preserve numerical definitions and labels. |
| `sg_components/metrics/moments/__init__.py` | Package initializer and explicit exports; keep only required re-exports. |
| `sg_components/metrics/moments/avg_precision.py` | SG-derived moment metrics; preserve numerical definitions and labels. |
| `sg_components/metrics/moments/metrics.py` | SG-derived moment metrics; preserve numerical definitions and labels. |
| `sg_components/metrics/moments/misc.py` | SG-derived moment metrics; preserve numerical definitions and labels. |
| `sg_components/metrics/moments/utils.py` | SG-derived moment metrics; preserve numerical definitions and labels. |
| `sg_components/model/__init__.py` | Package initializer and explicit exports; keep only required re-exports. |
| `sg_components/model/event_field/__init__.py` | Package initializer and explicit exports; keep only required re-exports. |
| `sg_components/model/event_field/framework.py` | Standalone EventField-Net with structured field routing and 2-D spans. |
| `sg_components/model/event_field/framework_boundary_preserving_v2.py` | Boundary-preserving contextual EventField decoder. |
| `sg_components/model/event_field/framework_context_contrast.py` | Task-specific tri-field decoder with multi-scale boundary context contrast. |
| `sg_components/model/event_field/framework_field_structured.py` | Compact field-structured backbone for standalone EventField-Net. |
| `sg_components/model/event_field/framework_proven.py` | EventField-Net using the validated I5 interaction as its internal stem. |
| `sg_components/model/event_field/framework_taskaware.py` | Task-aware backbone for Evidence, Support, and Transition fields. |
| `sg_components/model/event_field/interaction.py` | Controlled cross-modal interaction variants for continuous Event Field. |
| `sg_components/model/event_field/losses.py` | Losses for the standalone Event Field F0 module. |
| `sg_components/model/event_field/model.py` | Query-conditioned temporal Event Field F0 model. |
| `sg_components/model/event_field/targets.py` | Ground-truth field construction for Event Field F0. |
| `sg_components/utils/__init__.py` | Package initializer and explicit exports; keep only required re-exports. |
| `sg_components/utils/rw_utils.py` | SG-derived tensor/span/serialization utility. |
| `sg_components/utils/span_utils.py` | SG-derived tensor/span/serialization utility. |
| `sg_components/utils/tensor_utils.py` | SG-derived tensor/span/serialization utility. |
| `tests/test_release.py` | Regression checks for masks, gradients, schedules, tasks and model selection. |
| `train.py` | Public scratch/resume training CLI. |
| `training/__init__.py` | Package initializer and explicit exports; keep only required re-exports. |
| `training/config.py` | RunnerConfig/FactoryConfig types used by new public engine and model factories. |
| `training/contracts.py` | PreparedBatch/DataBundle/ParameterGroup/LossResult/OfficialEvalRequest dataclasses. |
| `training/probability.py` | Masked softmax used by live model. |

The seven efficiency changes are in `backbone/model.py`, `eventfieldnet/engine.py`, `eventfieldnet/model/{counterfactual,losses,selector}.py`, `field_core/selector.py` and `field_primitives/model/selector.py`. See `docs/EFFICIENCY.md` and `results/efficiency.json` for validation.
