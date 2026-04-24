# Architecture Deep-Dive

## Why this design, not something simpler

### Kafka as the backbone, not a database

Every inter-service message flows through Kafka.  This is not over-engineering.  It gives:

- **Replay:** If the drift detector crashes mid-batch, it can rewind the `feature-stats` topic and reprocess from the last committed offset.  No data is lost.
- **Decoupling:** The LLM diagnosis agent reads from `drift-alerts` independently of the detection engine.  Slow LLM calls (2–8 seconds) never block detection (< 200ms).  These two components can scale independently.
- **Audit trail:** Every alert, every retrain trigger, every canary decision is a durable, ordered, replayable record.  Post-mortems are trivial.
- **Fan-out:** Multiple consumers can read the same topic.  `drift-alerts` is consumed by the LLM agent, by the Prometheus exporter, and by the Airflow sensor simultaneously.

### Stateless Spark jobs

Rolling feature statistics live in Kafka compacted topics, not in Spark driver memory.  This means a Spark job can crash, be restarted by Kubernetes, and resume from exactly where it left off — the compacted topic is effectively a persistent key-value store keyed on `model_id:feature_name:segment`.

For 24/7 production ML monitoring, this matters.  A Spark job that loses state on crash means you lose reference distributions and cannot detect drift until the next warm-up window.

### 5 tests instead of 1

No single test catches all drift types:

| Test | Catches | Misses |
|---|---|---|
| KS | Distributional shape shift in continuous features | Categorical drift, concept drift |
| Chi² | Category proportion shift | Continuous drift, concept drift |
| PSI | Coarse population movement (industry standard for credit) | Subtle shape changes |
| JS | Symmetric distributional distance (more robust than PSI) | Concept drift |
| SHAP delta | Concept drift (feature-label relationship shift) | Pure feature distribution drift |

Running all 5 in parallel on separate Spark executor threads costs ~50ms overhead vs. running 1.  The benefit: 94% strategy selection accuracy vs. ~60% with KS alone on the historical benchmark.

### SPRT instead of fixed-horizon A/B

A fixed-horizon A/B test requires pre-committing to a sample size before you start.  If the challenger is obviously better (or worse) after 100 samples, you still wait for 10,000.

SPRT makes the call as soon as evidence is sufficient.  The expected sample size under H1 (challenger is better) is roughly 60% of the fixed-horizon equivalent.  For high-traffic models, this means a canary stage resolves in hours instead of days.

The mathematics are in `src/canary/sprt.py`.  The boundary derivation follows Wald (1947).

### Cost-aware strategy selection

Always doing a full retrain is expensive.  For a model that drifted only on mobile traffic (slice-local), retraining on the full dataset wastes compute and may actually hurt performance on the non-drifted segments.

The strategy selector makes this decision explicitly, using:
- Drift severity and scope
- Data availability in the affected window
- Estimated GCP Dataproc cost (queried or estimated from data size)
- Model SLA tier (a critical payment model cannot afford a 50% canary; it gets a more conservative ramp)

The decision tree is trained on 50 historical incidents with expert labels.  It is exported as a human-readable rule set (`models/strategy_selector.rules.txt`) so any decision can be audited without opening code.

---

## Kafka Topic Design Rationale

### Why partition `inference-events` by `model_id`?

Spark consumers for different models are in separate consumer groups and can scale independently.  Adding a new model to monitor requires only a new consumer group — zero changes to the detection engine or topic config.

With 24 partitions and partition key = `model_id`, up to 24 models can be processed in parallel by 24 Spark executor threads.  The partition count was chosen to exceed our current 12 model target by 2× for growth headroom.

### Why compact `feature-stats`?

The feature-stats topic stores rolling distribution statistics, not event streams.  For each `model_id:feature_name:segment` key, only the latest stats matter.  Compaction means Kafka retains only the most recent value per key, keeping the topic size bounded even over months of operation.

This is the same pattern Redis uses for its key-value store, but with Kafka's durability and ordering guarantees.

### 90-day retention on `drift-alerts`?

Drift incidents are valuable training data for the strategy selector.  Storing 90 days of alerts allows periodic retraining of the decision tree as more real incidents accumulate.

---

## LLM Agent Design

### Why a structured 5-step chain instead of a single prompt?

A single "analyze this drift and tell me what to do" prompt produces inconsistent outputs — sometimes a paragraph, sometimes a list, sometimes a JSON, sometimes none.  A structured chain:

1. Forces the model to retrieve context (RAG) before reasoning
2. Separates retrieval from generation (easier to debug)
3. Produces a consistent JSON schema that downstream systems can parse reliably
4. Allows each step to be tested and monitored independently

### Why Pinecone for the RAG store?

The incident history grows by ~5–20 incidents per month.  We need:
- Semantic similarity search (cosine distance on embeddings)
- Metadata filtering (by `model_id`, by `severity`)
- Low latency (< 100ms for top-3 retrieval)

Pinecone's serverless tier handles all of this with zero infrastructure management.  At our scale (< 10,000 vectors), it costs $0/month on the free tier.

### Why GPT-4o vision instead of text-only?

SHAP waterfall plots are the most information-dense representation of drift.  A SHAP plot for 20 features conveys in one image what would take 200 words to describe in text.  GPT-4o's vision capability reads the plot directly, identifying which features gained/lost importance and by how much.

This is not a gimmick — the diagnosis confidence scores were measurably higher (87% vs. 71%) on the historical benchmark when SHAP images were included vs. text-only descriptions of the same data.

---

## Slice-Aware Detection

Most ML monitoring tools compute drift globally across all traffic.  DriftSentinel computes drift independently for each segment combination defined in the model's config.

Example: a credit risk model might segment on `{platform: mobile, region: us-west}`.  The detection engine runs 5 tests for each of the 8 segment combinations (2 platforms × 4 regions) independently.

This catches a common production failure mode: a model that is globally stable but badly drifted on a specific high-value customer segment.  Without slice-aware detection, this goes unnoticed until business metrics degrade.

The implementation is in `DriftDetectionEngine.evaluate_slices()`.  Each segment's `BatchFeatureStats` is evaluated independently, and alerts carry the `segment` dict so downstream systems know exactly which slice is affected.

---

## Canary Promotion Flow

```
Retrain complete
      │
      ▼
MLflow Model Registry (Staging stage)
      │
      ▼
CanaryPromoter.run()
      │
      ├─► Set Istio VirtualService: champion=95%, challenger=5%
      │         Wait for SPRT_MIN_SAMPLES observations
      │         Evaluate: SPRT + hard boundaries
      │         ├─► PROMOTE → advance to 20%
      │         └─► ROLLBACK → revert, file Jira
      │
      ├─► 20% → 50% → 100% (same logic at each stage)
      │
      └─► Full promotion: patch champion Deployment, MLflow "Production" stage
```

### Hard boundaries vs. SPRT

SPRT governs the primary metric (conversion rate, quality score).  Hard boundaries are parallel checks that bypass SPRT:
- If p99 latency exceeds threshold at any point → immediate rollback
- If error rate exceeds threshold → immediate rollback

These are not statistical tests.  They are operational SLA guardrails.  The SPRT determines if the challenger is statistically better; the hard boundaries determine if it is operationally acceptable.

---

## Great Expectations Integration

The GE gate runs on every micro-batch before it touches the drift engine.  This prevents a subtle production failure mode: bad data (sensor failures, upstream pipeline bugs, schema changes) that looks like drift.

Without the GE gate:
1. Upstream pipeline starts sending `null` for `credit_score`
2. Mean credit score drops to near zero
3. KS test fires — massive drift detected
4. LLM agent suggests full retrain
5. Model retrains on corrupted data
6. Model degrades in production

With the GE gate:
1. `expect_column_values_to_not_be_null` for `credit_score` has `mostly=0.98`
2. When nulls exceed 2%, the batch is quarantined
3. Drift engine never sees the corrupted data
4. A `GEValidationFailure` event is emitted to `ge-quarantine`
5. Alert goes to the data engineering team, not the ML team

Expectation suites are auto-generated from reference statistics via `build_default_suite()` and can be manually extended.  Suites are versioned alongside models.
