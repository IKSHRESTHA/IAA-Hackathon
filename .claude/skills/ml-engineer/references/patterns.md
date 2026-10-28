# ML Engineering Patterns Reference

Deeper detail on scaling, reliability, tooling, and algorithm choices for production ML systems. Read only the section you need.

## Contents

1. [Feature stores and pipelines](#1-feature-stores-and-pipelines)
2. [Hyperparameter optimization at scale](#2-hyperparameter-optimization-at-scale)
3. [Scaling techniques](#3-scaling-techniques)
4. [Reliability patterns in depth](#4-reliability-patterns-in-depth)
5. [A/B testing for model rollouts](#5-ab-testing-for-model-rollouts)
6. [Advanced learning paradigms](#6-advanced-learning-paradigms)
7. [Tooling ecosystem](#7-tooling-ecosystem)

---

## 1. Feature stores and pipelines

- **Offline vs. online features**: offline features are computed in batch for training (e.g. Spark/SQL over historical data); online features must be computed or looked up fast enough for real-time serving (milliseconds). The same logical feature definition should back both, or training/serving skew creeps in.
- **Feature stores** (Feast, Tecton, or a homegrown table-plus-cache) exist to solve exactly that: define a feature once, materialize it to both a batch store (for training) and a low-latency store (for serving).
- **Point-in-time correctness**: when building training sets from historical data, join features as of the label's timestamp, not as of "now" — otherwise the model trains on information it wouldn't have had at prediction time (a subtle form of leakage).
- **Schema management**: version feature schemas explicitly; a silent type or meaning change breaks consistency checks downstream without an obvious error.
- **Consistency checks**: periodically diff a sample of online-computed features against the offline-computed equivalent for the same entity/time, to catch skew before it shows up as a performance drop.

## 2. Hyperparameter optimization at scale

- **Search strategy**: grid search only for 2-3 parameters with few values; random search is a strong simple default; Bayesian optimization (Optuna's TPE sampler, scikit-optimize) finds good regions in far fewer trials when each trial is expensive.
- **Early stopping within trials**: prune clearly-bad trials early (Optuna's pruners, ASHA) instead of letting every trial run to completion — this is often the biggest speedup available.
- **Parallel trials**: distribute across workers (Optuna with a shared storage backend, Ray Tune) when trials are independent; be mindful of shared resource contention (GPU memory, I/O to the same data source).
- **Resource allocation**: budget total compute (trials × time-per-trial) up front against the training-time requirement, not just "run until it's good."
- **Result tracking**: log every trial's parameters and score (MLflow, Weights & Biases, or a simple structured log) so the search is reproducible and auditable, not just the final winner.
- **Distributed training** (multi-GPU/multi-node) is a separate concern from distributed *search* — needed when a single model doesn't fit or train fast enough on one device (data parallelism for most cases; model parallelism only when the model itself doesn't fit in memory).

## 3. Scaling techniques

| Bottleneck | Technique |
|---|---|
| Too many requests for one server | Horizontal scaling behind a load balancer |
| Model too large for one node to serve fast | Model sharding, or a smaller distilled/quantized model |
| Many small requests | Request batching (accumulate briefly, run one batched forward pass) |
| Repeated identical or near-identical requests | Prediction caching (exact-match or semantic/embedding-based) |
| Bursty traffic | Auto-scaling on queue depth or latency, not just CPU |
| Slow downstream calls blocking the request | Async processing, with a sync fast-path for the common case |
| Expensive shared resources (DB connections, model instances) | Resource pooling instead of creating per-request |

Batching and caching usually give the largest latency/cost wins for the least complexity — try them before reaching for more infrastructure.

## 4. Reliability patterns in depth

- **Health checks**: a serving endpoint should expose both liveness (is the process up) and readiness (is the model loaded and able to serve a real prediction) — an orchestrator that only checks liveness will route traffic to a pod that's up but not actually ready.
- **Circuit breakers**: when a dependency (feature store, downstream model, external API) starts failing or slowing down, stop calling it for a cooldown period instead of letting every request queue up behind a doomed call.
- **Retry logic**: retry transient failures with backoff and a cap; never retry indefinitely, and never retry non-idempotent operations without care.
- **Graceful degradation**: decide, per failure mode, what the system returns — a cached last-known-good prediction, a simple heuristic fallback, or an explicit "unavailable" signal the caller can handle — rather than letting failures surface as generic 500s.
- **Backup models**: keep the previous production model's artifact loaded or quickly loadable; a rollback that requires re-running a training pipeline is not a fast rollback.
- **Disaster recovery**: know where the model artifacts, feature definitions, and training data snapshots are backed up, and that restoring the serving system doesn't depend on a single machine's local disk.
- **SLA monitoring and incident response**: define the SLA in concrete numbers (e.g. p99 latency, uptime, max staleness of predictions) before an incident, not during one, and have a runbook for the common failure modes (model server down, feature store stale, drift alert fired).

## 5. A/B testing for model rollouts

When the goal is measuring a business-metric improvement from a new model (not just verifying it's safe — see Step 4's shadow/canary guidance for that):

- **Traffic splitting**: randomize at the right unit (user, session, account) to avoid contamination between arms, and keep the split stable per-unit across the test duration.
- **Metric definition**: one primary metric decided before the test starts, plus guardrail metrics that must not regress.
- **Statistical significance**: compute required sample size from the baseline rate and minimum detectable effect before launching, and avoid stopping early just because an interim check looks good (peeking inflates false positives).
- **Result analysis**: check for sample ratio mismatch (are the arms actually the sizes they should be?) before trusting the metric comparison.
- **Decision framework and rollout**: decide in advance what result justifies a full rollout, a partial rollout, or a rollback, so the decision isn't made under the pressure of a live result.
- **Documentation**: record the hypothesis, design, and result together, even for tests that show no effect — a well-documented null result still saves someone from re-running the same test later.

## 6. Advanced learning paradigms

Reach for these only when the problem genuinely has the shape they solve — they add real complexity:

- **Online learning**: model updates continuously from a stream rather than periodic batch retraining. Useful when the underlying pattern shifts fast (e.g. fraud, trending content) and batch retraining can't keep up; needs careful safeguards against a single bad batch of data corrupting the live model.
- **Transfer learning**: start from a model pretrained on a related, larger task. The default for most image/text/audio work now rather than training from scratch.
- **Multi-task learning**: one model trained on several related objectives at once, when they share structure and joint training improves both over training separately.
- **Federated learning**: train across decentralized data that can't be centralized (privacy, regulatory, or bandwidth constraints), aggregating model updates rather than raw data.
- **Active learning**: the model (or pipeline) selects which unlabeled examples would be most valuable to label next, useful when labeling is expensive and you want to prioritize it.
- **Semi-supervised learning**: combine a small labeled set with a larger unlabeled set, useful when labels are the bottleneck but unlabeled data is abundant.
- **Reinforcement learning**: appropriate for genuinely sequential decision problems with delayed reward (not just "we have an API that returns a score") — most prediction/classification/forecasting tasks are not actually RL problems even when framed that way.
- **Meta-learning**: learning to adapt quickly to new tasks from few examples — relevant mainly when the real requirement is fast adaptation across many small, related tasks.

## 7. Tooling ecosystem

| Need | Common tools |
|---|---|
| Experiment tracking, model registry | MLflow, Weights & Biases |
| Pipeline orchestration | Kubeflow Pipelines, Airflow, Dagster, Prefect |
| Distributed compute / scaling | Ray |
| Hyperparameter optimization | Optuna, Ray Tune |
| Data and model versioning | DVC, lakeFS |
| Model serving | BentoML, Seldon Core, TorchServe, Triton Inference Server, plain FastAPI for simple cases |
| Feature stores | Feast, Tecton |
| Monitoring / drift detection | Evidently, WhyLabs, Arize, or custom dashboards on top of logged predictions |

Prefer whatever the repo or team already uses over introducing a new tool for the same job — consistency across the team's ML systems is worth more than any individual tool's extra feature.
