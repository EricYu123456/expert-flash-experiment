# Round 2 Local-Affine Compiler — Gemini Critique of the Prospective `PG-19 train16` Calibration Refresh

- **Scope & Hygiene**: Source/theory review only (`same_experimental_round: 2`). Read [stage2_root_review_notes.md](file:///home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/round02-local-affine/stage2_root_review_notes.md) in full and incorporated all qualifications below. Zero access to fitted `.npz`, activations, targets, `SCORE`, `GGUF` tensors, prompt/book payloads, or `TEST`; zero code edits, downloads, or execution.

---

## 0. Explicit Acceptance of Root's Stage-2 Qualifications

All eight qualifications in [stage2_root_review_notes.md](file:///home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/round02-local-affine/stage2_root_review_notes.md#L9-L16) are accepted without reservation:
1. Unmeasured `0.5%–2.5%` exposure estimates and linear `50×–100×` $\Delta\text{NLL}$ extrapolations are retracted; hit counts, true $\alpha$-mass (with actual unnormalized/normalized denominators), compute-weighted work (`DeepSeek` $I=2816$ shared branch has $2\times$ a routed $I=1408$ expert's coefficients), and nonlinear cross-layer interactions must be measured directly.
2. Teacher-forced tokens agree at the first modified layer (`OLMoE L0`, `DeepSeek L1`) only under identical upstream deterministic execution, and do **not** suppress hidden/KV propagation through downstream layers.
3. `q = 512` was one of two tested factor ranks (`q128`, `q512`), **not** a proved minimum rank (`q256` is unmeasured); `6.80 GB` on `OLMoE` is a known capacity floor, not a full deployment-fit proof, and smaller host budgets are at most secondary sensitivity cases.
4. Low guard coverage (`< 50%`) is **never** a standalone failure rule (partial coverage can still save work, and joint $K$-expert DAGs cannot assume independent $q^K$ fallbacks). Nor is there any `"iff"` theorem for publication value.

---

## 1. Critique of the Prospective One-Shot `PG-19 train16` Refresh Recipe

### 1.1 Rebuilding `C32` Tree / Centers / `P128` vs Keeping Old Synthetic Coordinates
- **Verdict — Rebuilding once on `combined FIT` is strictly fairer and scientifically right**:
  - Keeping old `tree / meanX_leaf / P128` learned solely on synthetic `pos 0..135` prompts would lock `P128` to synthetic short-context variance directions, asymmetrically penalizing `r=128`-projected families (`C_local` and the 6 `PE_RRR` operators) relative to full-input `G_RRR` (`input_rank = 2048`, 2 operators) and leaving `PG-19` `pos 136..799` states uncentered or imbalanced across leaves.
  - Rebuilding once with the **exact frozen Stage-0 unsupervised algorithm** treats `C_local` and `D_factor` symmetrically (~`30k` rows $\implies \sim 900+$ rows/leaf conditionally) and requires no hyperparameter search.
  - *Scale-invariance check (`[Proven from Stage-0 spec]`)*: Because Stage-0 normalized weighted loss and covariance by $\sum w_n$ and scaled penalties by $\operatorname{tr}(\hat{\Sigma})/r$, carrying the original dimensionless $(\eta, \lambda)$ from `~16k` to `~30k` rows on rebuilt coordinates is sample-size and coordinate-scale well-posed (with `1e-10` relative eigen-cutoff protecting $\eta=0 / \lambda=0$).

### 1.2 Teacher/Student Trajectory Union, Target Budget, and One Important Wording Fix
- **Target identity (`100%` coherent)**: Original `120 FIT` targets already evaluated the exact mathematical `canonical_expert` (`row-W8 / whole-A8 / full-I SiLU`) operator used by `B_pilotcanonical`. Reusing cached targets on the `120 FIT` rows and querying all 8 canonical experts once per new `PG-19 train16` row gives `C_local` and `D_factor` the **exact same input multiset and identical canonical target values**.
- **Causal correction on `"duplicate C/D prefix states"`**:
  - At the **first** pilot layer (`OLMoE L0`, `DeepSeek L1`), `C_old` and `D_old` prefix states on `0..767` are indeed bitwise identical to `B_old` (under identical batching/threads).
  - At the **second** pilot layer (`OLMoE L8`, `DeepSeek L13`), whenever `L0`/`L1` hits a pilot expert during the 768 prefix tokens, `C_old` and `D_old` hidden/KV states **differ** from `B_old`—they are **not** bitwise duplicates!
  - **Assessment**: Using **only `B_old`'s 768 prefix states + `B_old/C_old/D_old` up-to-32 decode states** (`max 13,824` new rows/layer) is **still the right finite design**—it avoids `3×` exact duplication at `L0/L1`, avoids `3×` heavily collinear same-token prefix reweighting and `3×` query cost at `L8/L13`, and already samples perturbed student KV trajectories via `C_old/D_old` decode steps (while acknowledging one-step off-policy shift between `old` collection models and `refreshed` deployed models). Simply **amend the protocol rationale** from *"exclude duplicate C/D prefix states"* to *"include one canonical `B` prefix trajectory per book plus `B/C/D` decode rollouts to avoid `3×` identical (`L0/L1`) or same-token collinear (`L8/L13`) prefix weighting."*

### 1.3 Necessary Pre-Payload Closures (4 Tight Items Only — Zero Extra Machinery)
Before reading any `PG-19 train16` book bytes, lock these four specification details:
1. **Exact decode-step & EOG boundary (`ctx = 1024` supports `768 + 32 = 800`)**: Pin whether logits at `pos = 767` pick greedy `g_0` and the loop runs `step = 0..31`, breaking **before** `llama_decode` whenever `g_step == EOG` (never feeding EOG into KV, yielding `0..32` decode input states per arm/book).
2. **`PG-19 train16` window & `< 768` token fail-closed rule**: Apply the identical UTF-8 / `CRLF->LF` / post-midpoint-`LF` suffix and `add_special=true, parse_special=false` native tokenization as `validation16`; if any pinned `train16` book suffix has `< 768` tokens on either tokenizer, abort fail-closed as an acquisition/window blocker before fitting (no loss-driven or silent book substitution).
3. **Preserve router `IDs` alongside `X` during `train16` capture**: Primary `C_local` is unweighted (`w=1`), **but** verify whether any of the 8 carried `VALID`-selected `PE6 / G2` recipes in `D_factor` use natural-route indicators (e.g., `1 vs 1/4` route weights or natural-route RMS/bias subsets). Saving `IDs[t, 0..K-1]` on every new row guarantees exact recipe replay for both `C` and `D`.
4. **Deterministic union row order & `FIT`-only post-refit sanity gate**: Freeze exact concatenation order (`120 FIT` in Stage-0 order, then `train16` in frozen hash order: `B prefix [0..767]`, `B decode`, `C decode`, `D decode`) so PCA/median ties and `F32` bias reductions are deterministic; verify finite weights/scales and log leaf occupancies (`synthetic` vs `PG-19` row counts) **before** releasing `PG-19 validation16`.

---

## 2. Informativeness of a `> 2%` Failure & Whether `guard85` Is Required

1. **Could a `2%` paired-book NLL failure (`lower simultaneous 95% band > ln 1.02`) still be uninformative for this concrete compiler?**
   - **No—once refreshed on `PG-19 train16` (`pos 0..799`), a valid `lower band > ln(1.02)` rejection is highly informative**:
     - Both major excuses from the unrefreshed build—**domain shift** (`synthetic` vs `PG-19`) and **positional extrapolation** (`pos 0..135` vs scored `pos 256..511`, now strictly interior to `[0..767]`)—are eliminated, leaf sample counts roughly double (`~30k` total), and the simultaneous 95% paired-book bootstrap lower bound automatically prevents a single outlier book (e.g., an isolated router tie-flip) from triggering false rejection (high single-book variance widens $s_j$, yielding `inconclusive`, not failure).
     - The only remaining caveats—that $(\eta, \lambda)$ and `C32/r128` were carried without a new PG-19 hyperparameter grid and without iterative on-policy refitting—are **inherent properties of a practical finite post-hoc compiler**, properly scoped in the bounded failure statement.
2. **Is a predeclared `guard85` arm necessary for a responsible stop decision?**
   - **No—it is NOT strictly necessary anymore**: Prior to the refresh, `guard85` was needed to protect against `pos 0..135 -> 256..511` OOD linear blowups. **With in-domain `PG-19 train16` (`pos 0..799`) calibration in place, rejecting the concrete unguarded `C32/r128` configuration if its simultaneous lower band exceeds `ln(1.02)`—while explicitly leaving guarded/other-rank/retrained variants unresolved—is completely honest and scientifically responsible.**
   - **Broad Scientific Recommendation**: Keep the Stage-2 quality screen lean and decisive—freeze the four tiny closures in §1.3, execute the single `PG-19 train16` refresh, and evaluate the **unchanged 4-arm `PG-19 validation16` protocol (`source`, `B_pilot`, `C_refreshed`, `D_refreshed`)** with the two primary `(Deep C-B, OLM C-B)` simultaneous bands. *(Only if Root actively plans to pursue a guarded variant inside Round 2 would adding one **secondary descriptive** `FIT-85%-distance` fallback diagnostic—scored by NLL + full fallback work, never failing on coverage alone and excluded from primary `max-|t|` multiplicity—save a follow-up pass; it is NOT required to validate stopping on the adopted unguarded compiler.)* If either model's primary lower band exceeds `ln(1.02)`, close Round 2 cleanly as a bounded deployment-quality failure and pivot fresh to Round 3; if non-rejected and competitive against `D_refreshed`, proceed to all-routed quality and inclusive `8e9 B / 8 GiB` host/AiF resource evaluation.
