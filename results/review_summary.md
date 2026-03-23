# Strategy Draft Review Summary

**Generated:** 2026-03-23T01:29:00+00:00
**Source:** /Users/k/Downloads/cody/hypex capital/rooroo/results/strategy_drafts/draft_vec_rsi_regime_core.yaml
**Drafts reviewed:** 1

## Verdict Summary

| Verdict | Count |
|---------|-------|
| PASS | 0 |
| REVISE | 1 |
| REJECT | 0 |
| Export Eligible | 0 |

## Individual Reviews

### draft_vec_rsi_regime_core — REVISE (confidence: 67)

| Criterion | Score | Severity | Reason |
|-----------|-------|----------|--------|
| C1_edge_plausibility | 95 | pass | Thesis contains specific causal reasoning. |
| C2_overfitting_risk | 80 | pass | 6 filters within acceptable range. |
| C3_sample_adequacy | 68 | pass | Estimated 36 annual opportunities. |
| C4_regime_dependency | 80 | pass | Single regime (mixed) but cross-regime validation planned. |
| C5_exit_calibration | 10 | fail | take_profit_rr=0.0 below 1.5 |
| C6_risk_concentration | 40 | warn | risk_per_trade=2.0% exceeds 1.5% |
| C7_execution_realism | 50 | warn | No volume filter in entry conditions. |
| C8_invalidation_quality | 80 | pass | 4 invalidation signals defined. |

**Revision Instructions:**
- Tighten stop-loss to <=15% and ensure reward-to-risk >= 1.5.
- Reduce risk_per_trade to <=1.5% and max_positions to <=10.
- Add volume filter (e.g., avg_volume > 500000) for execution realism.

