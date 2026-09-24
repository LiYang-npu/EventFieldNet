# Cleanup audit

The starting reference contained155 Python files across multiple staged roots. The release consolidates15 byte-identical backbone copies, removes4 historical runtime/deployment bridge modules, removes the unused short/Long KL factory functions and dispatcher, removes34 unused imports and12 unused local bindings, and restricts the public model factory to the V00 recipe. All source/entry files pass Ruff F-series checks. Formatting is normalized without editing the vendored standalone evaluator.

Code is organized into src/configs/scripts/docs/results/tests/external. Historical import names are migrated only in checkpoint factory metadata; tensor keys and mathematical initialization are preserved. Compatibility attribute names remain where changing them risks checkpoint, optimizer or probe behavior. Inherited backend classes are not necessarily redundant merely because their names originate in earlier experiments; they participate in the verified inheritance chain. A few conditional diagnostic/utility branches remain and are not claimed to have universal execution coverage.

FILE_REVIEW.csv inventories every distributed file available at the audit pass, including its role and evidence category. The inventory is a static/dependency review supplemented by integration and parity tests, not proof that every line is bug-free. Validation receipts document exact observed scope. No claim of complete24epoch retraining, universal correctness, fresh-environment certification is made.

Known failed packaging attempts are retained outside the release tree in the project audit directory. The smoke metadata migration bug was fixed and the full smoke passed after the fix. Original experiments, checkpoints and the earlier L07 package were not changed.
