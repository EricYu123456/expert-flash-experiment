# Equal-byte GU replica controls

The two-phase and standard g8 whole-GU constructions both add58completeGUcopies (87MiB/layer), use186WL/plane forGU, and admit every top6route. The ordinary g8 control prevents a phase-specific novelty claim.

| Kernel | Source/capability | waves | copy ns | latency µs |
|---|---|---:|---:|---:|
| phase replicas + split10/prefix1 | fixed0123, IPR, conditional pre-ECC drop | 13 | 500 | 155.140 |
| g8 whole-GU replicas + split10/prefix1 | fixed0123, IPR, conditional pre-ECC drop | 13 | 500 | 155.140 |
| g8 whole-GU replicas + whole dummy after10usefulwaves | fixed0123, everywave SAME-WL, conditional pre-ECC drop | 13 | 500 | 155.680 |
| either construction, plain | fixed0123, no drop | 12 | 100 | 150.980 |

Plain12waves atcopy500 fails the finite two-weight-latch deadline. No cold or idle rescue is inserted. The sealed-compatible split primary is checked directly by the unchanged pipeline; the whole-dummy same-WL control uses a small explicit full-plane recurrence, with every ECCcompletion and latch hazard bridged against the sealed simulator on432nonemptycases.

There are16availabilitycountclasses representing all74,974,368top6sets. Eachside has3exclusiveexperts; dualexperts fillremainingcapacity3. Fourexclusiveexperts ononeside would give acounterexample, so58dualexperts is minimal within this two-full-copy/capacity3 contract, not a newgeneral CBC theorem.

Capacity jointly includes triple routed Down and dedicated sharedg4 Down: GU186MiB + privateDown144MiB + sharedDown1.5MiB=331.5MiB/layer, below common384MiB. Privatechips use GUblock0andDownblock1; sharedchips fill GU0..185 andDown186..191 inblock0. EveryDownjob remainswhole384rows, onecommand/onecoldstart; 28blocks are nonempty.

The source ledgers preserve source page/byte identity. Phase replicas use64GUrows/chip across16chips; groupreplicas use128rows/chip across8chips. Shared remains128rows/chip across16chips. Complete owners consume their full6/12pages without insertinganotherowner. No quantization, dotproduct, PEselector, extra inputcontext orFIFO is changed.

Pre-ECC drop remains conditional. Skip-WL CR is the user premise; same-WL whole-dummy g8 does not need IPR, while split controls do. Firstpage28us is a conservative accounting convention, not measured availability. These are conditional kernel/compiler witnesses; fullFFN results and external independent audit live separately.
