New sharedg4 geometry has substantial standard-replication headroom; the old fixedsharedg16 bound does not apply.

With all64completeDown experts resident in each of3routedg4groups, everytop6 can use2/2/2. The nativekernel work is2×59.220=118.440us on routedchips versus89.940us on dedicatedsharedchips. FixedcurrentGU calendars plus minimumcommand/fullinput/lastONFI ownercost give optimisticfullFFN lowerbounds280.510/285.422us forinput1/4. Relative to the completednewsharedg4 means318.719906/323.713948us, remainingoptimisticheadroom is11.9886%/11.8289%. These arebounds, notachievedspeedups; actualfullFFN is anotheragent's standardcontrol.

Triplesadd96MiB/layer (twoextra completeDowncopies/expert), fitwithin2blocks/chip, and use28chipblocks in the fixedcandidate layout. RoutedDownoccupies192WL/plane inblock1; GUusesblock0max128WL. Sharedg4 needs6WL atblock0offset128. Uniquephysicalpagecoordinates andthreecompletecopies perlogicalband are checked. Onecopynewgeometry uses20blocks, so no equalactualallocation or capacityefficiency advantage is claimed.

The cheaperoverflow oracle gives everyexpert a completebackup in thesharedgroup:48MiB/layer and24chipblocks; it does not optimize a smallerstaticcopy inventory. Its bestchip-calendar means are301.956385/307.251312us. Forcing routedcap2 is not always latencyoptimal: (3,3,0) needs twospills andregresses, while (4,1,1) prefersonespill/threejobs overtwospills/twojobs. Exactenumeration andhandcases are retained.

This is standardwholeexpert replication/declustering, not a newrouting or CBCresult. The scopefixescurrentGU/nativejobgeometry, completeprivateinputs, row/scales semantics and thedeclaredconditionalhardwarecontract; otherresources/FIFO/selection remain relaxed. No newactualfullFFN, heldout orqualityrun was launched.
