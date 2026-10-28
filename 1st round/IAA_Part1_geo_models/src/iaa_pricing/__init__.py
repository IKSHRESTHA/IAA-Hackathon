"""Technical risk-premium modelling for motor, home buildings and home contents.

Package layout
--------------
config       run configuration (YAML -> typed object) and domain constants
io           typed loading of the competition parquet files
contracts    data-contract checks (schema, domains, keys, referential integrity)
cleaning     deterministic, logged data-quality fixes
claims       claim-level net cost and claim -> policy-row matching
development  reporting-delay (IBNR) completeness, open-claim development, trend
features     line-specific feature engineering incl. as-at claims history
metrics      Tweedie deviance / D^2, Gini, lift and A/E diagnostics
models       GLM benchmark, LightGBM Tweedie, blending
guardrails   submission checks, escalation triggers, run manifest
"""

__version__ = "1.0.0"
