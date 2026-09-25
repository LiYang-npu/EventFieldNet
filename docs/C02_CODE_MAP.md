# Reading order and paper-to-code map

|Paper component|Source file|Entry point|
|---|---|---|
|C02 model assembly / GT time-axis|eventfieldnet/evidence_supervision/model.py|build_model, gt_weights, CounterfactualMixin.compute_loss|
|Four-corner E equation|eventfieldnet/evidence_supervision/objectives.py|counterfactual_evidence_loss|
|Reciprocal negative pairing|eventfieldnet/evidence_supervision/pairing.py|reciprocal_pairs|
|Shared E-only auxiliary function|eventfieldnet/evidence_supervision/e_only.py|evidence_tokens|
|Fusion HD head and target|eventfieldnet/highlight.py|HighlightHead, highlight_loss|
|MR + HD assembly|eventfieldnet/joint_model.py|EventFieldNet|
|Shared encoder / candidate head|backbone/model.py|model definitions|
|Input field conditioner|backbone/conditioned/trifield_input_conditioner_plugin.py|conditioner definitions|
|Output field scoring|eventfieldnet/model/selector.py|forward, compose_score|
|Raw field interactions|eventfieldnet/model/interaction.py|RawInteraction|
|Local support residual|eventfieldnet/local_support.py|model/selector assembly|
|S auxiliary geometry|eventfieldnet/coverage_loss.py|coverage objective|
|Rank routing / complete model|eventfieldnet/model_factory.py|build_model|
|Weighted loss composition|eventfieldnet/model/losses.py|cf_e_objective hook and loss composition|
|Fixed training recipe|eventfieldnet/recipe.json|model_factory + loss_weights|
|Epoch loop / selectors|eventfieldnet/engine.py|train|

The first four files are the C02 addition. Lower-level historical attribute names remain to preserve state-dictionary keys and numerical operation order. They are compatibility details, not additional paper modules. This package is a readable faithful implementation, not a claim that the inherited backbone has been rewritten from scratch. See FILES.md for the broader inherited dependency map.
