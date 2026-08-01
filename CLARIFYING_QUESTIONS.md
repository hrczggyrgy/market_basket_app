# Clarifying Questions for Market Basket App Transformation

## Data Additions (Priority Order)

| Additional Field/Table | Unlocks | Do you have or plan to acquire? |
|------------------------|---------|----------------------------------|
| Product master: category, brand, pack size, unit, hierarchy | Trustworthy CDTs, category scorecards, price ladders | |
| Cost and funding | Margin, true promo ROI, GMROI | |
| Store/channel | Local assortment, store clusters, online vs store behavior | |
| Inventory and availability | Stockout-aware demand, turns, availability loss | |
| Promo calendar and mechanic | Reliable causal promotion evaluation | |
| Competitor price | Real price index and externally meaningful KVI decisions | |
| Planogram/space | Space productivity and actual range constraints | |
| Digital/shelf traffic | Conversion and lost-opportunity diagnosis | |

## Team Workflow

1. **Action recommendation owners**: Who will be the "owners" in the action recommendations table? Single category manager, multiple category managers by category, or a central PM?

2. **Approval workflow**: What's the expected review/approval flow? (e.g., Analyst → Category Manager → Director)

3. **Decision tracking**: Do you need audit trail with `decision_date`, `status` (pending/approved/rejected/deferred), and `actual_outcome` for learning loop?

## Export & Reporting

4. **Export formats needed**:
   - CSV/JSON (already supported)
   - PowerPoint-ready slides?
   - PDF executive summaries?
   - Scheduled email reports?

5. **Dashboard vs. Worklist**: Is the primary consumption a dashboard (exploratory) or a prioritized worklist (operational)?

## Timeline & Scope

6. **Timeline**: Is the 10-week phased approach acceptable, or do you need a faster MVP (e.g., Category Health + Action Center in 4 weeks)?

7. **Migration strategy**: 
   - Keep existing tabs alongside new workspaces (gradual migration)?
   - Full cutover to workspace model?
   - Hybrid: workspaces as primary, legacy tabs under "Advanced"?

## Technical Decisions

8. **Taxonomy confidence threshold**: What confidence % = "high confidence" for category-level decisions? (Suggested: ≥0.8)

9. **Bootstrap iterations for Demand Transference CI**: 500 (faster) vs 1000 (more precise)?

10. **Recommendation ID format**:
    - UUID v4 (e.g., `550e8400-e29b-41d4-a716-446655440000`)
    - Human-readable (e.g., `CAT-20260801-001`, `AST-20260801-042`)

11. **Model registry storage**:
    - SQLite file (local, queryable)
    - JSONL (simple, append-only)
    - PostgreSQL/other DB (if deployed centrally)

12. **Testing framework**: Currently uses pytest. Add `hypothesis` for property-based tests on malformed transactions?

13. **CI/CD**: GitHub Actions already configured? Should validation benchmarks run on every PR?

## UX Preferences

14. **Default mode on load**: Business / Guided / Expert?

15. **Color scheme for RAG**: Current (Green/Amber/Red) or custom?

16. **Drill-through depth**: How many levels? (Category → SKU → Detail → Scenario is 4 levels)

---

## Decision Log

| Date | Question | Decision | By |
|------|----------|----------|-----|
|      |          |          |     |

*Add decisions here as they're made*
