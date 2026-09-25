# Tasks

`task: mr_hd` enables retrieval and the fusion highlight head. `task: mr` omits the highlight head/loss. Both retain C02 E counterfactual loss, S/T losses and all three scoring paths. E requires temporal GT, video identity and query text, not ordinal ratings. HD still requires genuine QV ratings. The supplied loader, annotation duration contracts and evaluator are QV-specific; other datasets need separate adapters.
