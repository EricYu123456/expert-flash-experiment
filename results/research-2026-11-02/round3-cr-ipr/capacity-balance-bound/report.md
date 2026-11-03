For owner-homogeneous full pages and every top-K set, universal full-wave utilization forces all experts to have the same plane-count profile. If W=(K L+S)/P is integral and K ceil(L/P)<=W, the minimum possible maximum resident footprint is W+(E-K)ceil(L/P). This is an elementary global page-count certificate, not native scheduling sufficiency.

Exhaustive checks: 123 parameter cases and 36,983 layout pairs, including nonidentical expert profiles. Every route and capacity threshold agrees.

G4 (L=6,S=12,E=64,K=6,P=4): W=12, minimum maximum footprint128pages; one192-WL block suffices for counts. DeepSeek (L=22,S=44): W=44, minimum392pages on a plane, exceeding two192-WL blocks. Thus no such one-copy universal44-full-wave layout fits two blocks per plane, although the mean resident footprint363pages does.

Elementary counting lemma, not a new general theorem or a publication breakthrough.
Sufficiency is only for global page counts: it proves no row-stream, same-block epoch, finite-PB, ECC or native timing feasibility.
No guarantee for nonuniform expert sizes, mixed-owner pages, copies, inactive overfetch, variable shard widths, or changing the selected set.
A TRAIN-only layout need not serve all top-K sets equally and can trade a small route-specific imbalance for lower resident capacity.
A lower bound on resident blocks is not a bound on actually touched blocks, cold starts, or latency.
