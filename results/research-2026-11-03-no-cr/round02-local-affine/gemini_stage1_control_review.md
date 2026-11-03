# Round 2 Local-Affine Compiler — Gemini Stage-1 Falsification & Control Design Review

- **Status**: Conceptual & mathematical control-design review only (`2026-11-03 UTC`, same Round 2 candidate `layer-shared-input-partition-local-affine-expert-compiler`).
- **Data Hygiene**: Frozen `rev003` `FIT`/`VALID` summary reviewed at aggregate range level only; **`SCORE` (40 prompts) and `TEST` (100 prompts) remain strictly sealed**. Zero model/prompt/activation/target reads, zero code edits, zero terminal execution.
- **Epistemic Labels**: Every claim below explicitly separates **`[Proven]`** (exact linear algebra / IEEE-754 / architectural byte arithmetic), **`[Cited Prior]`** (primary literature), and **`[Unmeasured]`** (empirical hypotheses requiring test).

---

## 0. Mandatory Self-Correction of Earlier Unadopted Hadamard Note

In Block 4 §4B (Option A2, **not adopted** in frozen `rev003`), I overstated the variance- and row-`amax`-equalization properties of a single fixed deterministic signed Walsh-Hadamard transform $Q_r = \frac{1}{\sqrt{r}} H_r D_{\text{sign}}$. That optional claim is **retracted and restricted** as follows (`[Proven]`):

1. **No deterministic diagonal-variance equalization for general conditional covariances**: For $Q_r = \frac{1}{\sqrt{r}} H_r D_s$ ($s \in \{\pm 1\}^r$),
   $$(Q_r \Sigma Q_r^T)_{jj} = \frac{1}{r}\operatorname{tr}(\Sigma) + \frac{1}{r}\sum_{k \neq m} H_{jk} H_{jm} s_k s_m \Sigma_{km}.$$
   The off-diagonal sum vanishes **only** when $\Sigma$ is strictly diagonal or **in expectation** over random i.i.d. Rademacher signs $\mathbb{E}_s[s_k s_m] = 0$. For one fixed sign draw and a non-diagonal conditional covariance—such as within-leaf demeaned or route-weighted pooled covariance even in global PCA coordinates—a Hadamard transform can **worsen** coordinate dispersion: e.g., for $\Sigma = \bigl[\begin{smallmatrix}1 & 0.5 \\ 0.5 & 1\end{smallmatrix}\bigr]$ and $H_2 = \frac{1}{\sqrt{2}}\bigl[\begin{smallmatrix}1 & 1 \\ 1 & -1\end{smallmatrix}\bigr]$, $H_2 \Sigma H_2^T = \bigl[\begin{smallmatrix}1.5 & 0 \\ 0 & 0.5\end{smallmatrix}\bigr]$.
2. **No deterministic per-row `amax` improvement**: Preserving $\|w\|_2$ does not bound $\|w Q_r^T\|_\infty \le \|w\|_\infty$ deterministically; e.g., $w = [1, 1]$ maps to $[\sqrt{2}, 0]$, increasing `amax` from $1$ to $\sqrt{2}$ (and up to $\sqrt{r}$ for worst-case rows aligned with Hadamard columns).
3. **Empirically moot in `rev003`**: Frozen `rev003` `VALID` `FP32` vs `W8A8` diagnostics differ by only `~1e-4`, confirming quantization outlier rotation is not the bottleneck (`"just fix quantization"` is empirically unsupported).

---

## 1. What the `rev003` `VALID` Numbers Do—and Do NOT—Identify

Observed `VALID` normalized SSE/energy across the 8 pilot operators (`DeepSeek` L1/L13 × 2 experts, `OLMoE` L0/L8 × 2 experts; `C=32, r=128, normal PCA`, unweighted 120 `FIT` prompts, unguarded):
- **Local `C32, r128` bank ($A_{e,c}\delta + b_{e,c}$)**: **`0.0147 – 0.1667`**
- **Pooled `R1(r128)` ($A_e \delta + b_{e,c}$, same leaf free biases)**: **`0.1408 – 0.5407`**
- **Global full-rank `R2(H=2048)` ($A_e^{\text{full}}(x - \bar{x}_c) + b_{e,c}$, identity coordinates)**: **`0.0191 – 0.1749`** (`16×` active coefficients per expert vs `r=128`)
- **Learning curve (`80 -> 120` `FIT` prompts)**: All 8 operators still improve.

### Critical Structural Observation (`The R2(H2048) Identification Clue`)
Look jointly at **`Local(C32, r128)` (`0.0147 – 0.1667`)** and **`R2(H2048)` (`0.0191 – 0.1749`)**:
- `R2(H2048)` uses **ONE single global slope matrix $A_e^{\text{full}} \in \mathbb{R}^{2048 \times 2048}$ per expert** ($C_{\text{slope}} = 1$, zero slope variation across the 32 leaves; only the intercept $b_{e,c}$ varies by leaf), yet achieves nearly the same `VALID` error range as the 32-leaf local bank!
- **`[Proven Algebraic Consequence]`**: Beating `R1(r128)` does **not** prove that an expert's conditional output subspace rotates across leaves, nor even that conditional slopes are required once input subspace sharing is relaxed! Specifically, **four distinct mechanisms** can generate `Local(C32, r128) ≈ R2(H2048) ≪ R1(r128)`, and **three of the four falsify the necessity of a conditional full-output Flash bank**:

| Hypothesis | Underlying Mathematical Mechanism | Requires $C=32$ Full $H\times r$ Flash Bank? | Falsifying Control |
| :--- | :--- | :---: | :--- |
| **H1: Estimator / Grid Capacity** | Two-stage `(eta, lambda)` iterated-ridge spectral filter $1 - \frac{\eta\lambda}{(d+\eta)(d+\lambda)}$ has $O((\eta/d)^2)$ passband bias vs single-stage ridge $1 - \frac{\eta}{d+\eta}$ (`C=1` local $\neq$ single ridge). | **No** | **`C1-2Stage`**: Give pooled `R1`/`R2` the exact same two-stage `(eta, lambda)` regularizer. |
| **H2: Expert-Specific Global Input Subspace (`Manifold Aliasing`)** | Suppose $F_e(x) - b_{e,c} \approx U_{e,q} V_{e,q}^T(x - \bar{x}_c)$ globally ($C_{\text{slope}}=1, q \approx 128$), where expert $e$'s `Gate`/`Up` subspace $V_{e,q}$ has large energy in $\ker(P_{128})$. Globally $\operatorname{Cov}(\delta, x_\perp)=0$ so `R1(r128)` fails, whereas on curved manifold leaves $x - \bar{x}_c \approx M_c \delta$, `Local` sets $A_{e,c} = U_{e,q}(V_{e,q}^T M_c)$ merely to reconstruct a **global** rank-$q$ map! | **No** (`~1.22 GiB` Host-resident total for `DeepSeek`!) | **`G-RRR`**: Per-expert global reduced-rank regression $b_{e,c} + U_{e,q} V_{e,q}^T(x - \bar{x}_c)$ ($C_{\text{slope}}=1, q \in \{128, 256\}$). |
| **H3: Fixed Output Subspace + Conditional Host Latent** | Every SwiGLU output lies in $\operatorname{col}(D_e)$; if $A_{e,1},\dots,A_{e,32}$ share one rank-$q$ output basis $U_{e,q}$ (or group/layer-shared $U_{g,q}$) and only rotate $q \times r$ latent maps $B_{e,c}$ on Host (`MoLAE`-like). | **No** (`~1.26–1.66 GiB` Host-resident total for `DeepSeek`!) | **`PE-RRR` & `Shared-U RRR`** ($b_{e,c} + U_{e,q} B_{e,c}\delta$ and $U_{g,q}\sum \alpha_e h_e$), plus `MoLAE-Nonlinear` $\phi_e(\delta, c)$. |
| **H4: True Rotating Output Subspaces / Flash-Capacity Tradeoff** | Across contexts $c$, $A_{e,c}\delta$ genuinely rotates across output subspaces whose union has high effective rank $\gg q$, so no Host-fittable shared/per-expert $U_q$ matches fidelity at equal active work. | **Potentially Yes** (if full-model quality & `W4`/cache controls also lose) | Survives **all** controls above + Oracle Fixed-$U_q$ bound + Stage-1C full-model `W4`/cache DAG. |

---

## 2. Exact Mathematical Derivations & Traps for Common-Output Controls

### 2.1 Optimal Common-Output Fitting vs Unweighted & Plug-In Coefficient SVD
Fix an expert $e$ (or expert group) with within-leaf demeaned dequantized `A8` design $V_{e,c} \in \mathbb{R}^{N_c \times r}$, centered canonical teacher target $\tilde{Y}_{e,c} \in \mathbb{R}^{N_c \times H}$, sample weights $W_{e,c}$, Stage-0 fitted slopes $\hat{A}_{e,c}(\eta, \lambda) \in \mathbb{R}^{H \times r}$, and FIT predictions $\hat{Y}_{e,c} = V_{e,c} \hat{A}_{e,c}^T$.

1. **Trap T1 — Why Unweighted Coefficient SVD $[\hat{A}_{e,1}\dots\hat{A}_{e,C}]$ Is a Weak Strawman (`[Proven]`)**:
   - Maximizing $\operatorname{tr}(U^T (\sum_c \hat{A}_{e,c} \hat{A}_{e,c}^T) U)$ treats all $r$ PCA coordinates as unit-variance ($I_r$) and all $C$ leaves as equally occupied. Because $\sigma_1^2 \gg \sigma_{128}^2$ in `normal PCA`, noisy slopes on tiny-variance tail PCs and rarely routed leaves hijack singular vectors of $U$. Report unweighted SVD only as a diagnostic check, **never** as the primary common-output control.
2. **Trap T2 — Plug-In Surrogate Covariance ($M_{\text{plug}}$) vs Exact Teacher-Target RRR ($M_{\text{target-exact}}$) under Ridge Shrinkage (`[Proven]`)**:
   - Because scalar-isotropic ridge ($\eta, \lambda$) acts identically across all $H$ output coordinates, for any fixed orthonormal $U \in \mathbb{R}^{H \times q}$ ($U^T U = I_q$), refitting latent matrices $B_{e,c}$ to projected targets $\tilde{Y}_{e,c} U$ with the same $(\eta, \lambda)$ yields the exact closed form $\hat{B}_{e,c}(U; \eta, \lambda) = U^T \hat{A}_{e,c}(\eta, \lambda)$ in real arithmetic.
   - However, when $\eta > 0$ or $\lambda > 0$, ridge residuals are **not** $W$-orthogonal to predictions ($\tilde{Y}_{e,c}^T W_{e,c} \hat{Y}_{e,c} \neq \hat{Y}_{e,c}^T W_{e,c} \hat{Y}_{e,c}$). Expanding the **true unpenalized FIT teacher squared error** of $\hat{Y}_{e,c} U U^T$ gives (`[Proven Identity]`):
     $$\sum_{c=1}^C \|\tilde{Y}_{e,c} - \hat{Y}_{e,c} U U^T\|_{W_{e,c}, F}^2 = \sum_{c=1}^C \|\tilde{Y}_{e,c}\|_{W_{e,c}, F}^2 - \operatorname{tr}\!\Bigl(U^T \underbrace{\sum_{c=1}^C \bigl(\tilde{Y}_{e,c}^T W_{e,c} \hat{Y}_{e,c} + \hat{Y}_{e,c}^T W_{e,c} \tilde{Y}_{e,c} - \hat{Y}_{e,c}^T W_{e,c} \hat{Y}_{e,c}\bigr)}_{M_{\text{target-exact}, e}\text{ (symmetric } H\times H\text{)}} U\Bigr).$$
   - Notice: $M_{\text{plug}, e} = \sum_c \hat{Y}_{e,c}^T W_{e,c} \hat{Y}_{e,c} = \sum_c \omega_{e,c} \hat{A}_{e,c} S_{e,c} \hat{A}_{e,c}^T$ coincides with $M_{\text{target-exact}, e}$ **if and only if** $\eta = \lambda = 0$ (unregularized OLS, classical `Izenman 1975` RRR). Under finite ridge shrinkage, $M_{\text{target-exact}, e}$ directly accounts for shrinkage bias against actual teacher targets (while remaining a single closed-form $H \times H$ eigendecomposition!). Both $M_{\text{plug}}$ and $M_{\text{target-exact}}$ can be evaluated on `FIT` with `VALID` selecting the winner cleanly.
3. **Oracle Fixed-$U$ Subspace Lower Bound (`[Proven Pointwise Bound]`, `sources.md` line 123)**:
   - For any `FIT`-frozen orthonormal $U_q \in \mathbb{R}^{H \times q}$ and leaf intercepts $b_{e,c}$, **no** latent function $h(x) \in \mathbb{R}^q$—even an oracle unbounded nonlinear network—can achieve squared error lower than:
     $$\mathcal{E}_{\text{oracle}}(U_q) = \frac{\sum_{x} \bigl\|(I_H - U_q U_q^T)(y^{\text{canon}}(x) - b(x))\bigr\|_2^2}{\sum_{x} \|y^{\text{canon}}(x)\|_2^2}.$$
   - Computing $\mathcal{E}_{\text{oracle}}(U_q)$ on `VALID` separates **insufficient output subspace rank $q$** (where even an oracle nonlinear Host MLP $\phi_e(\delta, c)$ must fail!) from **insufficient linear latent $B_{e,c}\delta$** (where $\mathcal{E}_{\text{oracle}}(U_q)$ is tiny, motivating a small nonlinear Host latent control only if linear $B_{e,c}\delta$ leaves a large gap above $\mathcal{E}_{\text{oracle}}$).

### 2.2 Trap T3 — Latent Gauge Degeneracy & Second-`A8` Starvation in Lowered `U B_{e,c}`
In real arithmetic, $(U_q D)(D^{-1} B_{e,c}) = U_q B_{e,c}$ for any invertible diagonal $D \in \mathbb{R}^{q \times q}$. **In deployed `W8/A8` hardware, it is NOT invariant (`[Proven Non-Equivalence]`)**:
- Deployed execution computes $h(x) = \text{deqW8}(B_{e,c})\,v(x) \in \mathbb{R}^q$, **re-quantizes $h(x)$ via whole-vector dynamic `A8`** ($\tilde{h}(x) = s_h(x)\,q_h(x)$, $s_h(x) = \|h(x)\|_\infty / 127$), and then applies $\text{deqW8}(U_q)\,\tilde{h}(x) + b_{e,c}$.
- If $U_q$ is kept unit-norm orthonormal ($U_q^T U_q = I_q$), all output singular-value decay ($\sigma_1 \gg \sigma_q$) concentrates in $h_1(x) \gg h_q(x)$; whole-vector `A8` sets $s_h(x) = |h_1(x)|/127$ and **crushes tail latent coordinates $h_j$ ($j \gg 1$) toward zero**, creating an artificial quantization collapse in `U B_{e,c}`!
- **Mandatory Lowering Hygiene for All Factored Controls (`G-RRR`, `PE-RRR`, `Shared-U RRR`)**:
  1. **Latent `rms-floor` balancing on `FIT`**: Compute per-coordinate RMS $\rho_j = \sqrt{\text{mean}_{\text{FIT}}(h_j^2)}$, define $d_j = \max(\rho_j, \rho_1 / 16)$ (or balanced split $\sqrt{d_j}$), absorb $U' = U_q \operatorname{diag}(d)$, $B'_{e,c} = \operatorname{diag}(d)^{-1} B_{e,c}$ so dynamic `A8` utilizes full 8-bit range across all $q$ latent channels while monitoring row-`amax` of $U'$.
  2. **Post-`W8(U')` least-squares & `F32` bias recalibration**: Quantize $U' \to \hat{U}'^{\text{deq-W8}}$ first, project `FIT` teacher residuals onto $\hat{U}'^{\text{deq-W8}}$ via ridge-stabilized pseudo-inverse so $B'_{e,c}$ compensates `W8(U')` rounding, quantize $B'_{e,c} \to \hat{B}'^{\text{deq-W8}}_{e,c}$, run the **exact two-stage scaled-integer `W8/A8` pass** on `FIT`, and calibrate `F32` leaf biases $b_{e,c}$ from `weighted_mean_FIT(target - actual_two_stage_pred)`. Always report `FP32` factored predictions alongside `W8/A8` to verify zero artificial lowering degradation.

### 2.3 Shared-$U$ Degeneracies, Route-$\alpha$ Cross-Errors, and Pre-Lift Aggregation
For routed mixture output $\hat{y}_{\text{mix}}(x) = \sum_{e \in S(x)} \alpha_e(x)\,\hat{F}_e(x)$ with expert errors $\varepsilon_e(x) = F_e^{\text{canon}}(x) - \hat{F}_e(x)$:
$$\|\varepsilon_{\text{mix}}(x)\|_2^2 = \sum_{e \in S(x)} \alpha_e(x)^2 \|\varepsilon_e(x)\|_2^2 + 2\sum_{\substack{e < e' \\ e, e' \in S(x)}} \alpha_e(x)\alpha_{e'}(x)\langle \varepsilon_e(x), \varepsilon_{e'}(x)\rangle.$$

Three exact traps arise when sharing $U_{g,q} \in \mathbb{R}^{H \times q}$ across experts (`G=1` layer-shared $U_{l,q}$ or `G=4` group-shared $U_{g,q}$):
1. **Trap T4 — Coherent Orthogonal Residual Alignment**: Under a shared $U_l$, every active expert drops $\ker(U_l^T) = \operatorname{range}(I_H - U_l U_l^T)$. If co-routed experts share positively correlated components in $\ker(U_l^T)$, the off-diagonal sum $2\sum_{e<e'}\alpha_e\alpha_{e'}\langle \varepsilon_{\perp,e}, \varepsilon_{\perp,e'}\rangle$ amplifies layer error up to $K\times$ relative to separable expert SSE! Conversely, fitting $U_l$ directly to the **FIT centered mixture target** $r_{\text{mix}}(x) = \sum_{e \in S(x)} \alpha_e(x)(F_e^{\text{canon}}(x) - b_{e,c(x)})$ (or convex combination $(1-\beta)\Sigma_{\text{mix}} + \beta \sum_e \Sigma_{\alpha\text{-routed}, e}$, $\beta \in \{0, 0.5, 1\}$ selected on `VALID`) captures co-routed coherent directions directly.
2. **Trap T5 — Active-Work Unfairness if `Layer-Shared U_l` Is Restricted to $q=128$ Only**:
   - With `G=1` (`Layer-Shared` $U_{l,q}$) and **Path A (`Aggregate-Before-Lift`)**, Host forms $h_{\text{agg}}(x) = \sum_{e \in S(x)} \alpha_e(x) h_e(x) \in \mathbb{R}^q$ and executes **ONLY ONE** $H \times q$ output lift for the entire layer instead of $K=6$ (`DeepSeek`) or $K=8$ (`OLMoE`) lifts!
   - At $q = 512$, **one** $2048 \times 512$ `W8` lift is `64` 16 KiB pages (**1 ideal 64-plane wave**, or `1.05M` MACs), which is **strictly less active work** than $K=6$ local maps at $r=128$ (`96` pages, $\ge 2$ waves, `1.57M` MACs) or $K=8$ (`128` pages, `2` waves, `2.10M` MACs)! Limiting `Layer-Shared U_l` to $q=128$ (`16` pages total vs `96–128` pages!) would handicap the shared-lift baseline by $6\times\text{–}8\times$ in active lift work. Hence `Layer-Shared U_l` **must** be allowed $q \in \{128, 512\}$ (and `G=4` group-shared $q \in \{128, 256\}$).
3. **Trap T6 — Path A (`Aggregate-Before-Lift`) vs Path B (`Lift-Each`) Non-Equivalence**:
   - Because $Q_{\text{A8}}(\sum_e \alpha_e h_e) \neq \sum_e \alpha_e Q_{\text{A8}}(h_e)$ and `F32` rounding orders differ, Path A is **not** bit-equivalent to Path B: Path A saves $(K-1)$ large lifts and $(K-1)$ `A8` quantizers, at the cost of quantizing small-$\alpha_e$ expert latents under the combined vector's `amax`. Stage-1 must evaluate the **exact ordered Path A implementation** (frozen accumulation order by increasing expert ID, single ties-away `A8`, single `W8` $U_g$ lift, plus $\sum_e \alpha_e b_{e,c}$ recalibrated on `FIT` mixture residuals).

---

## 3. Joint Host DRAM + Flash Capacity & Hardware Reality Check

### 3.1 Why Host-Resident Factored Controls Cannot Be Forced onto Flash (`[Proven Byte Arithmetic]`)
From architectural dimensions alone (`DeepSeek-V2-Lite`: 15.7B total, 26 MoE layers × 64 experts × $3 \times 2048 \times 1408 = 14.395\text{B}$ routed FFN params $\implies \mathbf{\sim 1.30\text{B}}$ non-routed-FFN params including `449.84 decimal MB` / `429.00 MiB` for all 26 native shared experts; `OLMoE-1B-7B`: 6.92B total, 16 layers × 64 experts × $3 \times 2048 \times 1024 = 6.442\text{B}$ routed FFN params $\implies \mathbf{\sim 0.48\text{B}}$ non-routed-FFN params):

| Full-Model Routed Representation (`W8` maps + `F32` $C=32$ biases/scales/`P_l`/tree) | `DeepSeek` (26 MoE L, $K=6$) Routed Footprint | `OLMoE` (16 MoE L, $K=8$) Routed Footprint | Fits in `8,000,000,000 B` / `8 GiB` Host **with** Non-FFN + KV? | Steady-State Routed NAND Sensing |
| :--- | :---: | :---: | :---: | :---: |
| **Candidate Local Bank $A_{e,c}$ (`C32, r128`)** | `13.84 GiB` (`14.86 GB`) | `8.52 GiB` (`9.15 GB`) | **NO** (Exceeds both `8e9 B` & `8 GiB`!) | $K$ maps (`96` / `128` pages/layer) |
| **`G-RRR` Global Slope ($U_{e,128} V_{e,128}^T + b_{e,c}$, $C_{\text{slope}}=1$)** | **`~1.25 GiB`** (`1.34 GB`) | **`~0.77 GiB`** (`0.83 GB`) | **YES (`~2.7–3.0 GB` total w/ non-FFN+KV!)** | **`0` pages** (100% Host DRAM!) |
| **`PE-RRR` Per-Expert $U_{e,128} B_{e,c}\delta + b_{e,c}$** | **`~1.67 GiB`** (`1.79 GB`) | **`~1.03 GiB`** (`1.11 GB`) | **YES (`~3.1–3.4 GB` total w/ non-FFN+KV!)** | **`0` pages** (or 1 Flash map/exp) |
| **`Layer-Shared Lift` $U_{l,128} B_{e,c}\delta + b_{e,c}$ (`G=1, q=128`)** | **`~1.26 GiB`** (`1.35 GB`) | **`~0.77 GiB`** (`0.83 GB`) | **YES (`~2.7–3.0 GB` total w/ non-FFN+KV!)** | **`0` pages** (`360k` Host MACs/L!) |
| **`Layer-Shared Lift` $U_{l,512} B_{e,c}\delta + b_{e,c}$ (`G=1, q=512`)** | **`~3.79 GiB`** (`4.07 GB`) | **`~2.33 GiB`** (`2.50 GB`) | **YES (`~5.3–5.6 GB` total w/ non-FFN+KV!)** | **`0` pages** (or `1` 64-page wave/L) |

- **Crucial Systems Consequence**: If **any** of `G-RRR`, `PE-RRR`, or `Layer/Group-Shared Lift` preserves teacher/model fidelity at these ranks, **it fits entirely in the `8 GB` Host DRAM alongside non-FFN weights and KV cache** (requiring zero steady-state routed NAND reads, with cold SSD-to-DRAM preload/demand-paging timed honestly at `8.0 GB/s` PCIe + ordinary `28 us` LSB reads)! Pretending `1.26–3.79 GiB` representations must reside on Flash to manufacture a story is strictly forbidden.
- **Native Shared-Expert Symmetry (`429 MiB` Fact on `DeepSeek`)**: All 26 `DeepSeek` shared experts total **`429.00 MiB` (`449.84 decimal MB`)** in `W8` with `100%` token hit rate. If left on Flash, `DeepSeek` keeps its 2-phase `GU -> Down` shared critical path even when all 6 routed maps are 1-phase; if pinned in Host DRAM (`429 MiB`), **both** baselines and candidate remove the Flash shared barrier equally. Both regimes must be modeled symmetrically in Stage-1C DAG schedules.
- **AiF Interface Honesty**: Ordinary `be-enc133` LSB read is `28 us / 16 KiB` (`CR-read` disabled); `6.4 GB/s` is ECCLITE/CR internal target, **not** proven PE compute rate; `r=128` (`g4`, 4 chips × 4 planes = 16 pages, 512 rows/chip) requires an explicit firmware/command contract for short-vector dispatch, FIFO output streaming (`2048 B/chip` for 4-byte dots), per-command input posts (no free $K$-context PB persistence), and complete-row scale handling.

---

## 4. Additional Primary Prior Art Beyond Already-Named Papers

Beyond `Flash-Jacobian`, `MoLAE`, `ITC-MoE`, `Sub-MoE`, `MoE-SVD`, `SLBF`, `Swift-SVD`, `Izenman (1975)`, `Evgeniou & Pontil (2004)`, `SPICE`, `MiLo`, `Half the Nonlinearity`, `TARDIS`, `LAWS`, `PREP`, `MoLE`, `PDDP`, `M5`, `FlashMoE`, `KVNAND`, and `AiF`, the Stage-1 design and any surviving narrative must explicitly account for:

1. **Closed-Form Activation-Whitened SVD & Shared-Base MoE Decomposition (`[Cited Prior]`)**:
   - **ASVD** (Yuan et al., *arXiv:2312.05821*) and **SVD-LLM** (Wang et al., ICLR 2025, *arXiv:2403.07378*): Establish closed-form truncation of $W S^{1/2}$ under empirical activation covariance $S = \mathbb{E}[x x^T]$ as exact data-aware rank reduction—precluding novelty claims for covariance-weighted factor projection (`M_plug`).
   - **D2-MoE** (Wang et al., ICLR 2025) and **RS-MoE** (2025): Post-training decomposition of MoE experts into shared base weights + SVD delta / ridge-calibrated low-rank + sparse components without retraining.
2. **Binary-Tree Conditional FFN Execution & Affine Spline Theory (`[Cited Prior]`)**:
   - **Fast Feedforward Networks (`FFFs`)** (Belcak & Wattenhofer, NeurIPS 2023, *arXiv:2308.14711*) and **UltraFastBERT** (2023): Traverse a binary hyperplane tree ($O(\log C)$ inner products on $x$) to execute a single leaf linear/FFN block conditionally per token—direct neural precedent for tree-routed leaf operators (trained end-to-end rather than post-hoc compiled per MoE expert for Flash).
   - **Affine-Spline Deep Network Representation** (Montúfar et al., NeurIPS 2014; Balestriero & Baraniuk, ICML 2018): Classical piecewise-affine partition theory of gated/rectifier networks.
3. **Contextual Sparsity Predictors + Flash Bundling + Host Caching (`[Cited Prior]`)**:
   - **DejaVu** (Liu et al., ICML 2023), **LLM in a Flash** (Alizadeh et al., ACL 2024), and **PowerInfer / PowerInfer-2** (Song et al., SOSP 2024; Xue et al., 2024): Pre-read low-rank contextual predictors + Host hot-weight DRAM caching + contiguous Flash bundling without `CR-read`.
4. **Iterated Tikhonov & Multi-Group Subspace Regression (`[Cited Prior]`)**:
   - **Iterated Tikhonov Regularization** (Twomey 1963; Hanke & Groetsch 1998) for the two-stage `(eta, lambda)` spectral filter; **Ando & Zhang** (JMLR 2005), **Cook, Li & Chiaromonte** (*Statistica Sinica* 2010), and **Reinsel, Velu & Chen** (*Multivariate Reduced-Rank Regression*, 2nd ed., 2022) for multi-task/multi-group shared-subspace estimation.

---

## 5. Compact, Decisive Stage-1 Falsification Funnel & Pre-Frozen Decision Rules

To avoid both strawman baselines and unbounded parameter grids, execute Stage-1 as a **3-step sequential short-circuiting funnel** within **Round 2** (`100 TEST` prompts stay **100% sealed** until Gate 1C protocol freeze!):

### Step 0: Unseal Frozen `rev003` Stage-0 `SCORE` First (Zero Retroactive Changes)
- Evaluate already-frozen `rev003` on the 40 `SCORE` prompts exactly as specified.
- **Outcome rules (already frozen)**:
  - Dominated/tied on `SCORE` by `{R0, R1, R2}` at $\le$ active/host/retained cost with simultaneous `max-|t|` support $\implies$ **Valid Same-Round Bounded Failure** (stop Round 2; 8 opportunities remain).
  - Undefined studentization / `CI crossing 0` / sparse hits $\implies$ **Inconclusive (remain in Round 2)**.
  - Credible non-dominated local point $\implies$ **Mark `40 SCORE` as opened Stage-0 development data** (`100 TEST` remains untouched held-out!) and immediately run **Gate 1A**.

### Gate 1A: 8-Operator Zero-New-Query Subspace Falsification (Minutes on CPU, 0 New Teacher Passes)
Using **only** the already-captured `120 FIT` / `40 VALID` states & targets of the 8 pilot operators (hyperparameters selected strictly on `VALID`, evaluated on `VALID` + `40 SCORE` with simultaneous paired prompt-bootstrap `max-|t|` bands; no arbitrary post-hoc `%` cutoff):
1. **`C1-2Stage`**: Global pooled slope at $r \in \{128, 512, 2048\}$ with identical two-stage `(eta, lambda)` iterated-ridge capability.
2. **`G-RRR` ($C_{\text{slope}}=1$, $q \in \{128, 256\}$)**: Per-expert global reduced-rank slope $b_{e,c} + U_{e,q} V_{e,q}^T(x - \bar{x}_c)$ on full $x - \bar{x}_c$, lowered with `FIT` latent `rms-floor` balancing, `W8(U)` pseudo-inverse refit of `W8(V)`, and two-stage `F32` leaf bias recalibration. *(Also log $C_{\text{bias}}=1$ vs $32$ diagnostic.)*
3. **`PE-RRR` ($C_{\text{slope}}=32$, $q \in \{128, 256\}$)**: Per-expert common-output conditional-latent $b_{e,c} + U_{e,q} B_{e,c}\delta$ fitted via `VALID`-chosen `{M_target-exact, M_plug}` (with unweighted SVD logged as diagnostic only), lowered with the exact same `W8/A8` recalibration hygiene, **plus the Oracle Fixed-$U_{e,q}$ Floor $\mathcal{E}_{\text{oracle}}(U_{e,q})$**.

- **Gate 1A Decision Logic**:
  - **Valid Same-Round Failure (Falsification of Conditional Full-Output Bank)**: If `C1-2Stage(r128)`, `G-RRR(q=128/256)`, or `PE-RRR(q=128/256)` dominates or matches `Local(C32, r128)` (simultaneous band supports control $\le$ candidate, or exact tie) **and** learning curves (`80 -> 120` prompts) + $\mathcal{E}_{\text{oracle}}(U_{e,q})$ show no unresolved sample-starvation crossover, then a `13.00 / 8.00 GiB` Flash bank is **dominated on both error and total/Host-fittable storage (`1.25–1.67 GiB` Host-resident $\implies 0$ steady-state routed Flash reads!)**. *(If a global/shared-lift engineering variant later yields $\ge 10\%$ whole-model speedup without novelty, log under the campaign's separate engineering label, never as candidate publication success.)*
  - **Inconclusive (Stay in Round 2)**: `CI crosses 0`, or `Local(C32, r128)` trails at 80 prompts and converges/crosses sharply at 120 prompts while $\mathcal{E}_{\text{oracle}}(U_{e,q}) > \mathcal{E}_{\text{Local}}(120)$ (indicating 120 truncated-128-token `FIT` prompts underfit `Local`; resolve within the existing non-`TEST` development budget—e.g. `40 -> 80 -> 120` curve audit, retaining $>128$ tokens per `FIT` prompt, or refitting on pooled non-`TEST` development splits under cross-validation—while keeping `100 TEST` strictly sealed).
  - **Advance to Gate 1B**: `Local(C32, r128)` achieves a statistically supported error reduction over **all** of `C1-2Stage`, `G-RRR(q=128/256)`, and `PE-RRR(q=128/256)` at matched/superior active-work/Host-budget frontier (corroborated by $\mathcal{E}_{\text{oracle}}(U_{e,128}) > \mathcal{E}_{\text{Local}}$).

### Gate 1B: 4-Pilot-Layer All-64-Expert Mixture & Group/Layer-Shared Lift Screen (`256` Experts Streaming)
Stream sufficient statistics across all 64 experts of the 4 pilot layers (`DeepSeek` L1, L13; `OLMoE` L0, L8) and score **exact routed 6-/8-expert layer mixture error** $\|\sum_{e \in S(x)} \alpha_e \varepsilon_e(x)\|_2^2 / \|y_{\text{mix}}^{\text{canon}}(x)\|_2^2$:
1. **`Shared-U RRR` (`MoLAE-Linear`)**: Layer-shared ($G=1, q \in \{128, 512\}$) and cosine-clustered group-shared ($G=4, q \in \{128, 256\}$) output lifts fitted with mixture-aware RRR, evaluated under **exact Path A (`Aggregate-Before-Lift`)** `W8/A8` execution.
2. **`MoLAE-Nonlinear`**: Small Host 2-layer SwiGLU bottleneck $\phi_e(\delta, c) \in \mathbb{R}^q$ ($q=128$, fixed `FIT` step budget, `VALID` early-stop) feeding `PE-RRR` $U_{e,128}$ and `Layer-Shared` $U_{l,512}$ (testing whether cheap Host nonlinearity replaces $C=32$ local linear pieces).
3. **Host-Split Projected-`GU` + Original Flash `Down` (`AiF` split)** and **SPICE-Style Base + Low-Rank Residual** (`DeepSeek`: native shared expert base; `OLMoE`: top-1 routed expert base).

- **Gate 1B Decision Logic**:
  - **Valid Same-Round Failure**: Any Host-fittable (`\le 8e9 B / 8 GiB` inclusive of non-FFN+KV) or equal-active-wave control above dominates `Local(C32, r128)` on layer-mixture error with simultaneous prompt-bootstrap support.
  - **Inconclusive**: Ambiguous bands / off-route calibration mismatch unresolved.
  - **Advance to Gate 1C (Full-Model Expansion)**: `Local(C32, r128)` remains non-dominated on the layer-mixture error vs active-Flash-work + Host-DRAM Pareto frontier across both `DeepSeek` and `OLMoE`.

### Gate 1C: Pre-Frozen All-Layer Free-Running Quality, Guarded Fallback & Hardware Schedule on Sealed `100 TEST`
Freeze the complete Stage-1C protocol **before opening `100 TEST`**, compile all routed layers (`26` `DeepSeek`, `16` `OLMoE`), and evaluate **both unguarded (`100%`) and `VALID`-quantile guarded (`50%, 70%, 85%` layer-synchronized `||x - meanX_leaf||_2` + residual-risk guard with exact 2-phase `3HI` original fallback)** on:
- **Quality metrics**: True closed-loop **free-running generation** (measuring actual downstream trajectory shift and achieved fallback rates per layer; if off-policy drift degrades deeper layers, perform one `FIT`-only on-policy state refresh before ever opening `TEST`!) + held-out `100 TEST` teacher-forced NLL/PPL and task accuracy.
- **Mandatory Full-System Baselines (under identical `8,000,000,000 B` and `8 GiB` Host budgets & symmetric Flash/Host shared-expert placement)**:
  1. Canonical `W8A8` & native source `GGUF`;
  2. Calibrated **`W4A8` original experts** (per-row `W4` and groupwise `g128`/`Q4_0`, paying `GU -> Down` barrier and disclosed nibble-unpack / partial-sum payload costs);
  3. **Reduced top-$K$** original `W8A8` experts ($K' \in \{2, 3, 4\}$);
  4. **Equal-Host-DRAM Hot-Expert / Hot-Map Cache** (`LRU` and `FIT`-frequency static+LRU pinning, timing both cold startup load over `8 GB/s` PCIe + ordinary NAND reads and warm decode).
- **Final Round 2 Adjudication**: Candidate achieves a valid positive research outcome **only** if, on sealed `100 TEST`, it traceably Pareto-dominates **all** controls above on `(free-running & held-out NLL quality) × (modeled native wave/channel DAG + measured Host CPU latency) × (8e9 B / 8 GiB Host + 1 TB/3 ordinary-LSB Flash capacity)` across **both** shared-expert (`DeepSeek`) and non-shared-expert (`OLMoE`) models. Dominance by any control on `TEST` with statistical support is a **Valid Same-Round Failure**; unresolved CI or broken engineering plumbing remains **Inconclusive** within Round 2.
