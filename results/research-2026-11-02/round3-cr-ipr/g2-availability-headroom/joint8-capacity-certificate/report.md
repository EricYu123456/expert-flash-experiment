# Joint eight-pair CBC: cap33 is impossible

**Within the fixed24-page full-copy contract, the minimum possible maximum residentjobs/group is34.** A33-job cap is impossible by exact Hall counting; a known34-job construction passes all255Hall constraints. This is a standard CBC capacity certificate, not a newtheorem or fullFFN result.

LetA,B be the nonempty supports of the two fixed sharedhalf objects. For everygroup subsetT, universal top6routed plusbothshared matching requires min(nE(T),6)+s(T)≤|T|.

First,A∪B must cover all8groups. Ifg is outside their union, T=allgroups exceptg contains both shared supports, hence nE(T)≤5. Groupg would have to store at least59routedexperts, contradicting33.

If a sharedsupport has sizea≤5, let r=8−a andh=6−a. Sum Hall over all size6supersets of that support. An expert havingd supportgroups outside it is counted C(r−d,h−d)times. Exact scalar inequalities give the following lower bounds on routedcopies outside the sharedsupport; eachoutsidegroup already stores the other shared object, leaving at most32expertcopies.

| Shared support size | Outsidegroups | Required expertcopies, integer LB | Capacity |
|---:|---:|---:|---:|
| 1 | 7 | 300 | 224 |
| 2 | 6 | 251 | 192 |
| 3 | 5 | 206 | 160 |
| 4 | 4 | 162 | 128 |
| 5 | 3 | 113 | 96 |

Thus both sharedsupports have size≥6. Summing the56size5Hall inequalities gives ΣE C(8−dE,5−dE)≤280. For every nonempty support,6d+C(8−d,5−d)≥28, so the64experts require at least252copies. A33-job cap allows264copies total; sharedcopies require at least12. Equality is forced: both supports have size6, expertcopies=252, and every size5subset contains exactly5expert supports.

Take one shared6support. Its6five-subsets therefore have30expert-containment incidences. Hall on the6support allows at most5expert supports inside it, and every nonempty expert support is contained in at most5of itsfive-subsets. There can be at most25incidences, contradicting30. This proves33impossible even for fractional support-count LPvariables.

Solver cross-check:24shared-size symmetry representatives use LP relaxations;3representatives also use directintegerMILP. Time limit is2seconds/call. Infeasible statuses: 27/27; unknown statuses: 0. Exact arithmetic, rather than solver status, is the certificate.

The34-job witness retains64distinct four-subsets after deleting3complementpairs from all70four-subsets; eachgroup stores32experts andbothsharedobjects. All255Hall inequalities pass. Down uses204WL/chip. With GU186WL, total390WL exceeds the384WL two-block budget by6WL; the hoped-for33-job/198WL composition is unavailable under this contract. The known witness has303MiB/layer with onecopyGU99, and390MiB with GU186. Its total payload is not claimed globally minimal.

No TRAIN, prospective data/outcomes, heldout orfullFFN was accessed. This does not reject partialcopy/variable-row layouts or other ABIs beyond the fixed full24pagecopy model.
