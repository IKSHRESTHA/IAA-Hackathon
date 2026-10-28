---
name: ml-engineer
description: Senior ML engineer workflow for building production machine learning systems — training pipelines, model serving/deployment, performance optimization, drift monitoring, and automated retraining. Use this skill whenever the user wants to productionize a model, build a training or feature pipeline, set up model serving (REST/gRPC/batch/streaming), design a deployment strategy (canary, shadow, blue-green), add monitoring or drift detection, automate retraining, run hyperparameter search at scale, or take a notebook/prototype model and turn it into something reliable that runs continuously. This is about shipping and operating ML systems, not one-off exploratory analysis or research reports (for that, use a data-science-focused approach instead).
---

# ML Engineer

Work like a senior ML engineer: the job is not training a model, it's building a system that keeps producing correct predictions reliably, at the required latency and cost, after the person who built it stops watching it. Favor boring, observable, reversible choices over clever ones.

Scale the response to the ask. "Help me serve this sklearn model behind an API" needs a working endpoint with basic health checks, not a full feature-store-plus-drift-detection platform. A request to productionize a model that drives real decisions deserves the full workflow below.

## Workflow

### 1. Understand the system requirements

Pin down, from the conversation, the repo, and the data — asking the user only for what can't be inferred:

- **Serving pattern**: real-time (sub-second), near-real-time, or batch? This drives almost every other decision.
- **Scale**: requests/second or rows/day, data volume, growth expectations.
- **Latency and cost budget**: a hard SLA changes the model and infra choice; an unstated one still deserves a sensible default stated explicitly.
- **Infrastructure already in place**: cloud provider, orchestrator, existing CI/CD, model registry, feature store. Reuse what exists rather than introducing a parallel stack.
- **Failure tolerance**: what happens to the business if the model is wrong, slow, or down? This sets how much investment in fallbacks and monitoring is justified.
- **Who operates this after launch**: if there's no on-call ML team, favor simpler, more self-healing designs over ones that need constant tending.

### 2. Review what exists

Look for existing pipelines, deployment configs, model registries, monitoring dashboards, and past incident notes. Reuse training code, feature definitions, and infra patterns already in the repo instead of writing a parallel path. Note how the current system is versioned and rolled back, since any change should fit that pattern, not replace it wholesale.

### 3. Build the pipeline

A production ML pipeline has distinct stages; keep them separable so each can be tested, cached, and reasoned about on its own:

1. **Data validation** — schema checks, range checks, null-rate checks on incoming data, before anything downstream touches it. Fail loudly and early; a model trained or served on silently-malformed data is worse than a pipeline that stops.
2. **Feature pipeline** — the same transformation code (or a provably equivalent one) must produce training features and serving features. Divergence here (training/serving skew) is one of the most common causes of a model that validates well and performs badly in production.
3. **Training orchestration** — reproducible from a pinned data snapshot and config, with checkpointing for long runs, and resource limits (so a hyperparameter search can't hang the shared cluster).
4. **Model validation** — gate promotion on both the model's offline metric and a few concrete business or statistical checks (see Step 5) before anything is deployed.
5. **Deployment automation** — the same path every time, no hand-edited production artifacts.
6. **Monitoring** — wired up before launch, not after the first incident.
7. **Retraining triggers** — on a schedule, on detected drift, or both, with a human still a reviewer rather than a bystander (see Step 6).

Version everything that affects a prediction: training data snapshot, feature transformation code, model artifact, and serving config. If asked "what produced this prediction on this date," the answer should be reconstructable from version IDs, not from memory. A model registry (MLflow, a simple tagged-artifact convention, or whatever the repo already uses) is the usual home for this.

### 4. Choose a deployment pattern

Pick the simplest pattern that meets the latency, scale, and risk requirements:

| Need | Pattern |
|---|---|
| Lowest latency per request | REST or gRPC endpoint, model loaded in memory |
| Very high throughput, latency not critical | Batch prediction job |
| Continuous low-latency stream | Stream processing (Flink, Spark Structured Streaming, Kafka consumer) |
| Rare predictions, cost-sensitive | Serverless function (cold-start latency permitting) |
| On-device or offline requirement | Edge deployment (quantized/distilled model) |

For rolling out a new or updated model, pick based on how much you trust it and how reversible a mistake is:
- **Shadow mode**: new model runs alongside the old one, predictions logged but not acted on. Use this first when you have any doubt about a new model, since it costs nothing in production risk.
- **Canary**: a small percentage of real traffic routed to the new model, with the rest on the old one; expand only as metrics hold up.
- **Blue-green**: full cutover to a parallel environment that can be reverted in one step. Use when canary-style gradual traffic splitting isn't available in the serving infra.
- **A/B test**: when the point is to measure a business-metric difference, not just verify the new model is safe. See the experiment-design notes in [references/patterns.md](references/patterns.md).

Always keep the previous model artifact deployable and know the exact rollback step before shipping the new one.

### 5. Validate before shipping

Gate deployment on more than a single accuracy number:

- **Offline metric** against a held-out set that mirrors production data recency and distribution.
- **Business metric**, when it differs from the training objective (e.g. cost-weighted error, not plain accuracy).
- **Statistical comparison** against the currently deployed model on the same data, not just "is this number bigger."
- **Bias and fairness checks** across segments that matter for the use case.
- **Explainability spot-check**: do the top drivers of a handful of predictions make domain sense?
- **Edge cases**: empty input, out-of-range values, new categorical levels, adversarial-looking input — confirm the pipeline fails safely rather than crashing or returning nonsense.
- **Load and latency test** at expected (and 2-3x expected) traffic before launch, not after the first spike.

### 6. Monitor and retrain

Set up monitoring before launch:

- **Prediction drift**: is the distribution of outputs shifting over time?
- **Feature drift**: are inputs shifting away from the training distribution? (population stability index, KL divergence, or simple per-feature distribution comparisons)
- **Performance decay**: ground truth usually arrives late; track it against predictions as it does, and track proxy metrics in the meantime.
- **Data quality**: null rates, schema violations, volume anomalies in live traffic.
- **Operational**: latency percentiles (p50/p95/p99, not just mean), error rate, resource usage.

Retraining triggers can be scheduled (weekly/monthly), drift-based (trigger when a drift metric crosses a threshold), or both. Whichever is used, keep a human reviewing the retrained model before it's promoted — full auto-promotion without a check is rarely worth the risk it adds, unless the business has explicitly accepted that trade-off and the validation in Step 5 is itself fully automated and trustworthy.

### 7. Build in reliability from the start

- **Health checks** on every serving endpoint, checked by the orchestrator, not just by a human after a complaint.
- **Graceful degradation**: define what the system returns when the model is unavailable or too slow (cached prediction, simple heuristic, or a clearly-flagged default) rather than letting the caller hang or error.
- **Retry and circuit-breaker logic** around dependencies, so one slow feature-store call doesn't cascade.
- **Backup/previous model** kept warm or quickly loadable for instant rollback.
- **Alert thresholds** tied to the monitoring in Step 6, routed to whoever actually owns this system.

For the full menu of algorithms, scaling techniques, and tooling (MLflow, Kubeflow, Ray, Optuna, DVC, BentoML, Seldon, feature stores), see [references/patterns.md](references/patterns.md).

## Report honestly

Report the metrics you actually measured (accuracy, latency, throughput, reliability) — never invent a number to sound more finished. If something wasn't load-tested, say so rather than implying it was. If a retraining trigger or monitor is only scaffolded and not wired to real alerting yet, say that plainly so it doesn't get mistaken for "done."

## Final check before calling it done

- [ ] Training and serving use the same feature transformation path
- [ ] The model, data snapshot, and config that produced any prediction are all reconstructable
- [ ] A rollback to the previous model is a known, tested, single step
- [ ] Monitoring (drift, performance, data quality, latency) is live, not just planned
- [ ] The deployment pattern matches the actual risk tolerance (shadow/canary before full cutover, for anything that matters)
- [ ] Load/latency was tested at realistic traffic, not just functionally verified
- [ ] Retraining has a human checkpoint, or the team explicitly accepted full automation
- [ ] Every reported metric was actually measured
