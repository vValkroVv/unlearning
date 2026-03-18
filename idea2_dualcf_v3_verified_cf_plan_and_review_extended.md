# Idea2 for DualCF: verified multi-alternate counterfactual supervision

_Expanded research note + implementation review for the current DualCF v1/v2/v3 line._

## 1. Scope and updated thesis claim

This note treats **Idea2** as the natural continuation of your current line:

- **Idea1**: difficulty- and attribution-aware routing decides **how hard** and **how cautiously** to unlearn each forget sample.
- **DualCF v2/v3**: you already implemented the routed core, broader artifact plumbing, multi-bank proxy retain maps, belief-bank generation, localized negative masking, and an optional SAM variant.
- **Idea2**: improve the **quality of the forget-side positive supervision** itself by replacing one weak alternate answer with a **verified pool of relation-preserving, answer-type-matched, shared-fact-safe candidates**, then train on the best one.

A concise thesis-level claim is:

> **DualCF performance is now limited less by routing capacity than by the quality of forget-side positive supervision. Verified multi-alternate counterfactual supervision should improve broad utility and truthfulness by making the positive branch relation-faithful, answer-type-safe, and shared-fact-preserving, while leaving the routed unlearning core intact.**

This framing is the most defensible one for your current results. Your runs already show that DualCF is strong on **holdout/locality**, but weaker on **broad utility**, with the largest deficits in **truthfulness/generalization** rather than ARC-style basic recall. That means the next improvement should target **target quality**, not simply stronger negative pressure.

## 2. Why Idea2 is the right next step after Idea1 and v2/v3

### 2.1 What Idea1 already established

Your original Idea1 note is fundamentally a **controlled unlearning** proposal:

- counterfactual supervision tells the model **where to move**,
- difficulty tells the method **how hard the target is to erase**,
- attribution tells it **how risky the update is for retained knowledge**.

That makes Idea1 a policy for allocating unlearning regimes across forget samples, not just another forget loss.

### 2.2 Where the current bottleneck now sits

Once routing works reasonably well, the next failure mode is usually **counterfactual target quality**:

- if the alternate is too noisy, the model learns the wrong positive target,
- if the alternate breaks relation structure, it hurts truthfulness and utility,
- if the alternate rewrites shared/public facts, it damages locality and broad capability,
- if the alternate is only surface-different from the gold answer, the method may look good on lexical suppression but not on semantic forgetting.

That is exactly why Idea2 is complementary to Idea1 rather than a replacement for it.

### 2.3 Literature motivation

Idea2 fits very cleanly into the current unlearning literature:

- **AltPO** argues that negative-only forget objectives often produce nonsensical or inconsistent outputs, and that adding positive supervision on the forget set improves forget-side response quality and overall behavior.
- **DUSK** argues that unlearning should remove **unique target content** while preserving **shared/public facts**; this is almost a direct formalization of what your counterfactual candidates should and should not change.
- **GUARD** shows that adaptive, attribution-aware allocation of unlearning power improves retention, which means Idea2 should be added **on top of**, not instead of, the retention-aware routing you already built.
- **Which Retain Set Matters?** shows that syntactic neighbors are especially fragile and especially useful as retain regularizers, which strongly supports your syntax/semantic/utility-anchor proxy bank design.

So the clean scientific story is:

- Idea1 = better routing policy,
- Idea2 = better positive supervision inside that routed policy.

## 3. Problem setup

For forget sample \(i\), define:

- prompt: \(x_i\)
- original answer / target memory: \(y_i^{orig}\)
- candidate alternate pool: \(\mathcal{C}_i = \{c_{i1}, c_{i2}, \dots, c_{iK}\}\)
- difficulty score: \(d_i\)
- attribution / retention-risk score: \(a_i\)

Idea1 already proposes a routed loss of the form:

\[
L_i
=
\lambda_i^{cf} CE(y_i^{cf}\mid x_i)
+
\lambda_i^{neg} L_{neg}(y_i^{orig}\mid x_i)
+
\lambda_i^{ret} L_{ret}.
\]

Idea2 changes **how** \(y_i^{cf}\) is chosen.

Instead of using one fixed alternate, define a verifier score for each candidate:

\[
q_{ij}
=
 w_{type} s_{type}(c_{ij})
+
 w_{short} s_{short}(c_{ij})
+
 w_{bank} s_{bank}(c_{ij})
+
 w_{judge} s_{judge}(c_{ij})
+
 w_{rel} s_{rel}(c_{ij})
+
 w_{shared} s_{shared}(c_{ij})
-
 w_{overlap} s_{overlap}(c_{ij}).
\]

Then choose either:

### Hard selection

\[
y_i^{cf,*} = \arg\max_{c_{ij} \in \mathcal{C}_i} q_{ij}
\]

and train with:

\[
L_i^{Idea2}
=
\lambda_i^{cf} CE(y_i^{cf,*}\mid x_i)
+
\lambda_i^{neg} L_{NPO}(x_i, y_i^{orig})
+
\lambda_i^{ret} L_{ret}.
\]

### Soft mixture (optional later ablation)

Define a normalized weight over candidates,

\[
p_{ij} = \frac{\exp(q_{ij}/T_q)}{\sum_k \exp(q_{ik}/T_q)}
\]

and optimize:

\[
L_i^{cf-mix} = \sum_j p_{ij} \cdot CE(c_{ij}\mid x_i).
\]

For the first serious campaign, **hard selection is better** because it is easier to interpret and closer to your current artifact contract.

## 4. What “verified” means in Idea2

“Verified” should not mean only “different from gold.” It should mean the candidate satisfies all of the following:

1. **Same relation as the original question**
2. **Same answer type / granularity**
3. **Short-answer style compatible with your evaluator**
4. **Not the gold answer and not a trivial paraphrase/subspan of gold**
5. **Safe with respect to shared/public facts**
6. **Compatible with local candidate-bank evidence when available**

This is the crucial difference between:

- a merely **incorrect** answer,
- and a **usable counterfactual target** for unlearning.

## 5. Why Idea2 should help your exact metrics

### 5.1 Utility and TruthfulQA

If the positive branch is poorly controlled, the model can learn to produce fluent but structurally wrong answers. That tends to hurt:

- truthfulness,
- factual calibration,
- broader multiple-choice/general evals,
- sometimes even retain-side locality.

That matches your current pattern more than a generic collapse story.

### 5.2 Forget cosine similarity

If the alternate is only lexically different but not semantically well-separated, the model may suppress benchmark text overlap without really moving away in semantic space. Idea2 will not solve all of your cosine gap by itself, but **better alternates should reduce the amount of “surface-only” forgetting**.

### 5.3 Stability

Better counterfactual targets usually reduce gradient noise on the forget branch. Since Idea1 is about controlled unlearning, that matters not only for final metrics but for:

- smoother trajectories,
- lower best-final gap,
- less checkpoint brittleness,
- lower seed variance.

## 6. Data requirements on DUET and RWKU

## 6.1 DUET

DUET is the cleaner place to test Idea2 because the relation structure is more controlled. For each forget sample, you want:

\[
(x_i, y_i^{orig}, \mathcal{C}_i, d_i, a_i)
\]

where the candidate pool \(\mathcal{C}_i\) is built from:

- candidate-bank neighbors from the same property / relation,
- external generator outputs,
- narrow answer-type-aware repair rules.

The most informative DUET analysis is **not only merged**. You should still report:

- rare vs popular,
- optionally stage-aware slices when relevant,
- matched-forget utility and locality.

Idea2 should be strongest on the slices where the current alternate quality is likely to be the main bottleneck rather than raw forgetting capacity.

## 6.2 RWKU

RWKU is the better realism stress test. There the alternates must be more conservative because the benchmark is closer to incomplete-information unlearning:

- you do not get a true forget corpus in the same way,
- you do not get a perfect retain set,
- adversarial/locality behavior matters more.

So on RWKU the main goal of Idea2 is not “invent more diverse wrong answers.” It is:

- preserve relation,
- preserve shared/public fact structure,
- only change the target answer span,
- avoid long explanatory alternates.

That should help utility/locality more than raw forget aggression.

## 7. The intended pipeline

## Phase A — generate 4 candidates, not 1

For each forget sample, generate a sidecar row such as:

```json
{
  "index": 17,
  "alternates": ["1997", "1998", "2001", "1993"],
  "scores": [0.94, 0.88, 0.76, 0.63],
  "relation_scores": [1.0, 1.0, 0.8, 0.6],
  "shared_fact_scores": [1.0, 1.0, 0.9, 0.2],
  "answer_type": "year",
  "generator": "gpt-5.4",
  "prompt_version": "cf_v3_idea2"
}
```

This sidecar should be generated **before** cleaning, difficulty, attribution, and calibration.

### DUET generation policy

- same relation,
- same answer type,
- short answer only,
- no explanation,
- moderate diversity allowed,
- candidate-bank aware when possible.

### RWKU generation policy

- change only the target answer,
- preserve shared/public facts,
- same answer type,
- short answer only,
- stronger penalty on fabricated explanations,
- more conservative than DUET.

## Phase B — verify and rerank locally

The reranker should not trust the first external candidate. Instead it should combine:

- answer-type match,
- shortness,
- candidate-bank consistency,
- lexical overlap penalty,
- external quality score,
- relation-preservation bonus,
- shared-fact-preservation bonus.

The current repo is already close to this, but still incomplete.

## Phase C — repair only failed rows

A good repair ladder is:

1. existing alternate,
2. external sidecar candidates,
3. candidate-bank neighbors,
4. narrow numeric/date/ordinal fallback,
5. second-pass generator call only for the remaining invalid rows.

Do not regenerate all rows if only a small tail is bad.

## 8. Relation to Idea1: how the two ideas compose

Idea2 should **not** change the routed core.

The intended composition is:

- Idea1 decides how hard/cautious to unlearn each sample.
- Idea2 decides what the positive target should be for that sample.

In other words:

- difficulty and attribution decide the **magnitude and risk policy**,
- verified multi-alternate supervision decides the **quality of the destination**.

That makes the combined method much cleaner scientifically than mixing many optimizer-level changes at once.

## 9. Minimal formal ablation tree

To isolate the value of Idea2, the minimum ablation tree should be:

1. **Current v3 artifact flow** (single best current alternate)
2. **4 candidates + current reranker**
3. **4 candidates + relation/shared-fact-aware reranker**
4. **4 candidates + relation/shared-fact-aware reranker + dataset-specific prompts**
5. Optional later: **top-2 soft mixture** instead of argmax single target

To keep the story clean, do **not** introduce strong new optimizer changes at the same time.

## 10. Expected empirical behavior

## 10.1 On DUET

Expected improvements:

- utility should improve more than raw forgetting,
- TruthfulQA-like behavior should improve first,
- holdout/locality should stay strong,
- forget ROUGE should stay similar or improve modestly,
- trajectory should become smoother.

Idea2 gains may be clearer on split-aware analysis than on a fully merged summary.

## 10.2 On RWKU

Expected improvements:

- locality / neighbor preservation,
- broad utility retention,
- robustness of artifact validity,
- cleaner forget-side behavior.

More cautious expectation:

- big gains on semantic forget metrics may still require your belief-suppression / localized-negative branch in addition to Idea2.

## 11. Evaluation protocol

Because this is still part of a controlled-unlearning claim, evaluation should not stop at endpoint metrics.

### Primary task metrics

- `forget_qa_rouge`
- `holdout_qa_rouge`
- `forget_qa_cos_sim`
- `holdout_qa_cos_sim`
- `utility_avg`
- `truthfulqa_bin_200_acc`
- `mmlu_pro_400_acc`
- `arc_200_acc`
- `winogrande_200_acc`

### Trajectory metrics

- time-to-threshold
- `U@F_tau`
- utility AUC
- max drawdown
- best-final gap
- seed variance

### Artifact-quality metrics (new for Idea2)

These are important because Idea2 is directly about counterfactual quality.

Per artifact build, log:

- valid-row rate,
- exact-match rate,
- gold-substring rejection rate,
- average candidate count,
- repair-source counts,
- sidecar coverage,
- relation-safe coverage,
- shared-fact-safe coverage,
- fraction repaired by fallback rules.

Without this, it will be hard to explain why one artifact regime helped more than another.

## 12. Current implementation review against the updated repo dump

## 12.1 What is genuinely in place

Your current repo dump is **substantially more mature than early v2**. The following are real and aligned with the v3 plan:

- multi-candidate sidecars and ranked picking exist,
- candidate-bank fusion exists,
- answer-type-aware fallback exists,
- proxy retain maps now include syntax, semantic, and utility anchors,
- belief-bank generation exists,
- `DualCFv3` and `DualCFSAM` configs exist,
- localized negative masking exists,
- launchers expose belief/local-neg/SAM knobs,
- validation notes in `dual_cf_integration_diff.md` are much stronger than before.

So this is **not** a broken prototype. It is a real v3 scaffold.

## 12.2 Where the current Idea2 review comments overstate the state of the code

The most important finding from the actual repo-to-text dump is that **several “fixed in the current patch” comments in the older Idea2 note are ahead of the code**.

### A. Route-disable semantics are still not neutralized in code

Current `DualCF._routing_state()` still does:

```python
if self.disable_difficulty_route:
    difficulty = torch.zeros_like(difficulty)
if self.disable_attribution_route:
    attribution = torch.zeros_like(attribution)

difficulty_gate = torch.sigmoid((difficulty - self.tau_d) / self.temp_d)
risk_gate = torch.sigmoid((attribution - self.tau_a) / self.temp_a)
```

That means the ablations are still not clean:

- difficulty-off is not the same as `difficulty_gate = 1`,
- attribution-off is not the same as `risk_gate = 0`.

So the route-disable bug is **still present in the repo dump**.

### B. Duplicate-candidate bias is still present

The sidecar path still seeds:

```python
primary_candidates = external_candidates[:1] or external_candidates
```

and then `select_best_alternate()` appends both `primary_candidates` and `external_candidates` into the same pool.

That duplicates the first external candidate.

The same pattern is still present in the vLLM primary path, where `response_candidates` are passed in both places.

So duplicate bias is **still present in the repo dump**.

### C. Prompt-family coverage is still narrow in the actual generator code

The HF generator still only exposes:

- `default`
- `strict_short`

I do **not** see `duet_relation_safe` or `rwku_shared_fact_safe` in the actual generator prompt-family table from the repo-to-text dump.

Also, the vLLM generator does not appear to take a `prompt_family` argument at all. It has a single generic system prompt and a small structured schema with:

- `alternate`
- `same_relation`
- `answer_type`

That means dataset-specific prompting is still not actually implemented in the dump you gave me.

### D. The reranker is still surface-form-aware, not Idea2-aware

Current `score_counterfactual_candidate()` still effectively uses:

- type match,
- shortness,
- bank membership,
- external score,
- overlap penalty.

It literally drops `question` with:

```python
del question  # Relation-level checks remain an offline heuristic for now.
```

So the reranker still does **not** explicitly score:

- relation preservation,
- shared-fact preservation.

This is the biggest remaining scientific gap for Idea2.

### E. The fallback policy is still more permissive than the older note says

`build_answer_type_fallback_candidates()` still includes:

```python
if answer_type == "short_span":
    ...
    candidates.append(f"not {text}")
```

and those fallback candidates still enter the ordinary repair pool. So the `not {text}` fallback has **not** yet been fully demoted to a last-resort, low-confidence repair path.

### F. Sidecar semantics are still too shallow for full Idea2

The structured vLLM schema currently exposes:

- `alternate`
- `same_relation`
- `answer_type`

That is useful, but it is still weaker than the richer Idea2 sidecar we want, which should support per-candidate arrays such as:

- `scores`
- `relation_scores`
- `shared_fact_scores`
- potentially candidate provenance or judge notes.

So the external sidecar is still underpowered for full Idea2.

## 12.3 What is still correctly claimed in `dual_cf_integration_diff.md`

Your updated `dual_cf_integration_diff.md` is mostly aligned with the repo dump on the major v3 facts:

- trainer-side contract is unchanged,
- multi-bank proxy maps exist,
- belief-bank support exists,
- localized negative masking exists,
- `DualCFSAM` exists,
- validation status still explicitly says GPU-backed attribution/generation and real short train/eval validation were not completed for this patch set.

That is a good and honest integration note.

The mismatch is mainly between:

- the **code dump**, and
- the more optimistic “fixed now” comments inside the older Idea2 review note.

## 13. Fix order before GPU debugging

1. **Fix route-disable semantics in `src/trainer/unlearn/dual_cf.py`**
2. **Remove duplicate candidate bias in both sidecar and vLLM primary paths**
3. **Add DUET- and RWKU-specific prompt families to both HF and vLLM generation paths**
4. **Extend sidecar schema with per-candidate relation/shared-fact metadata**
5. **Use those signals in `score_counterfactual_candidate()`**
6. **Demote `not {text}` to genuine last-resort repair only**
7. **Only then run the small GPU smoke and short-train validation ladder**

## 14. Tiny validation ladder before the H100 campaign

For each of:

- DUET rare,
- DUET popular,
- RWKU,

run:

1. 16-row artifact build
2. strict artifact validation
3. artifact-quality report
4. 1-step train smoke
5. short train + eval
6. only then the real sweep

The validation output should explicitly include:

- invalid counts,
- repair-source counts,
- duplicate-candidate detection,
- sidecar coverage,
- relation/shared-fact metadata coverage.

## 15. Final recommendation

The clean next thesis step is:

1. keep the current routed DualCF v3 core,
2. correct the remaining implementation mismatches,
3. implement Idea2 as **verified multi-alternate counterfactual supervision**,
4. run a controlled ablation against current v3,
5. add heavier belief / SAM combinations only after the positive branch is fixed.

That gives the cleanest scientific progression:

- **Idea1**: routing policy for controlled unlearning,
- **v2/v3**: routed implementation with broader artifact plumbing,
- **Idea2**: better positive supervision for utility-, truth-, and locality-preserving forgetting.

## References

- AltPO — Alternate Preference Optimization for Unlearning Factual Knowledge in Large Language Models
- DUSK — Do Not Unlearn Shared Knowledge
- GUARD — Guided Unlearning and Retention via Data Attribution for Large Language Models
- Which Retain Set Matters? — A Case Study on Entity Unlearning
- your `difficulty_attribution_counterfactual_unlearning_idea1_extended.md`
- your `dual_cf_integration_diff.md`
