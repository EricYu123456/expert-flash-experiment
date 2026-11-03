# g2 Down availability: capacity and kernel check

All252completeDowncopybands andbothchipshards were expanded and checked:508private/sharedkernels pass sealedfixed0123/strict500 timing. Eachprivatejob has768completeC512rows; eachsharedjob has384completeC1024rows. No job crossesblocks.

| Quantity | Verified result |
|---|---:|
| Private g2 / shared g4 kernel | 89.940µs |
| Private owner occupation, input1 | 90.510µs |
| Shared owner occupation, input1 | 90.734µs |
| Copy instances / resident experts pergroup | 252 / 42 |
| PrivateDownWL/chip + GUreservation | 252 +128 =380 |
| Actualweights with99MiBonecopyGU | 289.5MiB/layer |
| ConservativeGUreservation +Down | 318.5MiB/layer |
| Commoncapacitybudget | 384MiB/layer |
| SyntheticpackedW8_A8exactrows | 23040 |

Eachprivatechip places10whole6WLjobs inblock0WL128..187 and32 inblock1WL0..191. Thefourremainingblock0WL areunused. SharedDown usesblock0WL128..133 onchips12..15. Everyjob hasonecommand andonecoldstart. GUreservationsarenotfictitiousdataorreadpages.

Capacity1 is not universal. The explicit route[40, 41, 42, 43, 44, 45] avoids group0 and sees onlyfivegroups. Everygroupmisses22experts,so suchacounterexampleexists foreachgroup. Universalcapacity1wouldrequireeachgroupmissatmost5experts, at least354totalcopies, beyond252.

Capacity2 is universal byHall: anynonempty subsetofatmostsixselectedexperts hasatleastthreeneighborgroups,so twiceitsneighborcount isatleastthesubsetsize. This permitsatmosttwoseparateprivatejobs pergroup, executedserially. It doesnotgrant extra PE inputcontexts.

The six-wavecopy boundary is1940ns pass /1941ns fail underthedeclaredwholepagecache-onlymodel. The89940nskernel accountsfor28000nsstartup,500nscopy,and61440nsECC; arrayworkendsat76500ns. It is not inferredbymax(array,ECC).

The phase/groupGUreplica186WL maps do not fit this privateg2layout:186+252=438>384locally. Aggregateweightbytes do not remove thatperchipoverflow. ThiswitnessusesGU<=128WL andmustbecomparedwiththecorrespondingfeasibleGUmenus.

A perfectsixgroupmatching wouldreduceprivatechipkernelwork fromtwog4jobs118.44µs tooneg2job89.94µs, saving28.5µs. This is kernelheadroom only; actualTRAINmatchingandfullFFNreplayareseparateroot/numericwork. The construction is astandardreplication/shardingbaseline, not newCBC.
