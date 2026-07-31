# Runbook

Four incidents this system can actually have, with the checks that distinguish them.

Every log line is structured JSON on stdout with a `request_id`. Every symptom below names
the exact `message` field to search for.

**One rule before any of it:** `/predict` and `/explain` fail independently. Establish which
one is broken before anything else, because the answers have nothing in common.

```bash
curl -s localhost:8000/api/v1/health   | jq '{status, model_version, llm_mode}'
curl -s localhost:8000/api/v1/metrics  | jq '{healthy, n_calls, thresholds}'
```

---

## 1. Explanations degrade or stop

**Symptom** — `/explain` is slow, returns generic wording, or `/metrics` reports
`healthy: false`. `/predict` is unaffected: it makes no LLM call.

### Severity depends on which fallback fired

| `/metrics` threshold | Meaning | Severity |
|---|---|---|
| `fallback_rate` breached | The primary provider is failing; a later one is serving | Warning: output is still generated |
| `template_fallback_rate` breached | No provider answered. A static template was returned | Incident: customers are receiving boilerplate |

The limit on `template_fallback_rate` is 1%, not 10%. One occurrence is an incident, not a
rate to tolerate.

### First check

```bash
curl -s localhost:8000/api/v1/metrics | jq '.by_provider, .thresholds'
```

`by_provider` names which entry in the chain is actually serving. The configured order is in
`configs/llm.yaml`:

```
groq → groq_reserve_2 → groq_reserve_3 → openai → gemini
```

### Diagnosis

| Log `message` | `error` contains | Cause | Action |
|---|---|---|---|
| `llm provider failed` | `RateLimitError`, `tokens per day` | Daily token budget spent on that key | None needed: the next key takes over. Confirm `by_provider` moved on |
| `llm provider failed` | `insufficient_quota` | The account has no credit. Retries will not help | Top up, or accept the chain moving on |
| `llm provider failed` | `Timeout` | Provider degraded | Watch `latency_p95_ms`; the chain absorbs it |
| `llm unavailable, using template` |: | Every provider failed | See below |
| `langfuse unavailable` |: | Tracing only | Not customer-facing. Do not page |

**Groq meters tokens per day per key, and the three keys are separate buckets** (measured:
burning 1 500 tokens on key 3 left key 2 untouched). A single exhausted key is routine.

### When every provider failed

1. Confirm the network egress works from the container:
   `docker compose exec api python -c "import socket; socket.create_connection(('api.groq.com', 443), 5)"`
2. Confirm at least one key is present: `docker compose exec api env | grep -c API_KEY`
3. With no key at all the client runs in **mock mode** — `/health` reports
   `llm_mode: "mock"`. Deterministic JSON, not an outage, but not real explanations either.

### What not to do

Do not disable the template fallback to "surface the error". It is the last thing standing
between a provider outage and a 500 on a customer request.

---

## 2. The container will not start, or /health returns 503

**Symptom**, the container exits at boot, or every endpoint returns 503 with
*"Model not loaded. Train and register a model, then restart."*

This is deliberate. `lifespan` raises rather than serving degraded: **a container that cannot
decide a claim must never pass a health check.**

### First check

```bash
docker compose logs api | grep -E "pipeline failed to load|pipeline ready"
docker compose exec api ls -l /app/artifacts/models
```

Expected: `model.joblib`, `transformers.joblib`, `model_card.json`.

### Diagnosis

| Observation | Cause | Action |
|---|---|---|
| `/app/artifacts/models` empty | Volume not mounted, or host path wrong | Check the bind mount in `docker-compose.yml` |
| `model.joblib` present, `transformers.joblib` missing | Model registered without its fitted transformers | Re-run registration; they are versioned together |
| `ModuleNotFoundError: xgboost` (or catboost, lightgbm) | The champion is no longer a Random Forest. The serving image ships scikit-learn only | Add the library to `requirements.txt` and rebuild. See below |
| `PermissionError` on `/app/artifacts` | Ownership regression in the Dockerfile | `/app/artifacts` must be owned by `appuser`: it is created and chowned before `USER` |

### The estimator-library coupling

`requirements.txt` carries scikit-learn and no other estimator, because shipping all four
cost 1.2 GB of packages the process never imported. **Promoting a boosted-tree champion is
therefore a two-part change**: register the model, and add its library to `requirements.txt`.

Catch it before deployment, not at boot:

```bash
docker run --rm -v "$PWD/artifacts/models:/app/artifacts/models:ro" claim-approval-agent:latest \
  python -c "from src.genai.explanation_pipeline import ExplanationPipeline; ExplanationPipeline()"
```

---

## 3. Drift alarm

**Symptom**, the weekly drift job reports `requires_investigation: true`.

Score a batch against the frozen reference. `BATCH` is a file of raw claims in the source
schema; the transformers are the ones the model was fitted with.

```bash
BATCH=path/to/claims.xlsx python - <<'PY'
import json, os, pandas as pd
from src.data.loader import load_raw
from src.models.registry import load_production_model, load_transformers
from src.monitoring.drift import DriftMonitor

pre, fe = load_transformers()
features = fe.transform(pre.transform(load_raw(os.environ["BATCH"])))
features = features.drop(columns=["status"], errors="ignore")

model, card = load_production_model()
ordered = features[card.feature_names]
scores = model.predict_proba(ordered.astype(float))[:, 1]

report = DriftMonitor.load("artifacts/models").compare(ordered, scores)
print(json.dumps(report.as_dict(), indent=2))
PY
```

Rebuild the reference after every model promotion: `make drift-reference`.

### Read the prediction row before the feature rows

The report separates feature drift from **prediction drift**, and the distinction decides
the response. Measured on a deliberately shifted batch:

```
country            PSI 3.537  alarm
fee_to_rrp_ratio       2.288  alarm
rrp                    1.077  alarm
decline_risk_score     0.172  watch     <-- the output barely moved
```

Nine feature alarms, and the score distribution stayed inside its band. The model was
largely insensitive to that shift.

| Features | Score | Reading | Action |
|---|---|---|---|
| stable | stable | Nothing happened | None |
| alarm | stable | Input mix changed in a direction the model does not weigh | Investigate the source, do not retrain |
| stable | alarm | Same inputs, different outputs | Check for a pipeline or model-version regression first |
| alarm | alarm | Genuine population shift | Retraining candidate |

The third row is the dangerous one: identical inputs producing different scores usually means
a code change, not a world change.

### Before concluding

- `skipped` non-null means **nothing was scored**. Below 200 rows the monitor refuses rather
  than reporting large values computed from noise.
- Check `model_version` in the reference matches the serving model. A reference frozen
  against a different training run measures the wrong baseline and invents drift.
- An unknown categorical level falls into no bin and inflates PSI by design. Check
  `largest_shift_bin` before assuming a distribution moved.

### Escalation

A breach opens an investigation. It does **not** trigger an automatic retrain: the thresholds
(0.10 / 0.20) are the credit-scoring convention, not values derived from this dataset.

---

## 4. /explain exceeds its latency budget

**Symptom** — `latency_p95_ms` breaches 5 000 ms in `/metrics`, or requests feel slow.

### Check the sample size first

```bash
curl -s localhost:8000/api/v1/metrics | jq '{latency_p50_ms, latency_p95_ms, latency_confident, n_calls}'
```

**`latency_confident: false` means fewer than 20 live calls in the window.** A p95 over a
handful of calls is one sample wearing a statistic's name, and the threshold cannot breach
in that state. Gather traffic before investigating.

### Where the time goes

Roughly, on a healthy call:

| Stage | Typical |
|---|---|
| preprocessing + features | ~10 ms |
| Random Forest inference | ~1 ms |
| SHAP TreeExplainer | ~35 ms |
| prompt rendering | ~5 ms |
| LLM call | 1 000–3 000 ms |
| output validation | ~10 ms |

`/predict` measured at 108 ms in the container. Everything above 200 ms on `/explain` is the
provider.

### Diagnosis

| Signal | Cause | Action |
|---|---|---|
| `retried: true` frequent | Output failed validation, so the call was made twice | Read `validation.failed_checks`; a prompt regression doubles latency before it changes wording |
| `used_fallback_provider: true` | The primary timed out and a later entry served | Expected behaviour, not a fault |
| p50 also high | The provider is queuing, not one slow request | Free-tier queuing; the reserve keys do not help with latency, only with quota |
| Latency fine, throughput low | Rate limiting at the API layer | `/explain` is capped at 20 req/min |

### What not to do

Do not raise the `/explain` budget to make the alarm stop. The 5 s figure is what a human
reviewer will tolerate; the correct lever is caching or a faster provider, both listed as
future work in `DESIGN.md`.

---

## Appendix — what is safe to restart

| Action | Effect | Safe |
|---|---|---|
| `docker compose restart api` | Reloads the model. In-flight requests are lost | Yes: the cost ledger is a named volume and survives |
| `docker compose down -v` | Deletes the ledger volume. Monthly spend resets to zero | Only when the history is expendable |
| Rotating a provider key | The chain skips an entry with no key | Yes, at any time |
| Replacing `model.joblib` in place | The running process keeps the old model in memory | No: register properly and restart |
