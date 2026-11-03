Stop: unlimited fast private-Down band copies leave less than5% optimistic full-FFN headroom for the fixedcurrentGU + g4 Down jobs + sharedg16 configuration. No copies or new full-FFN replay were implemented.

The frozen TRAIN archive supplies512ordered top6routes ×3layers. Actual g4_reused_TRAIN_labels menus give24routed384-row bands/event. Maximum routed bands on onechip:2 in1202events,3 in328,4 in6; exactmean2.221354166667. Existing candidate versus equallyreoptimizedRR is only0.4016%input1 /0.3942%input4.

The finitePB pipeline gives59,220ns for a12page/3wave fourplane band; a plane2/3only copy needs6waves and82,120ns. Sharedg16 Down costs43,860ns. Everyjob still uploads the fullprivateC512vector, has onechip owner, and preserves384complete outputrows and their fixedbytes/scales.

With unlimited fastcopies, fixedGU ownerrelease calendars and minimumcommand/input/lastONFI owner time, the exactchip-calendar assignment bound is324,164ns forinput1 and329,540ns forinput4. Against currentmeans338,596.218750ns /344,970.645833ns, optimistic headroom is4.262369% /4.473032%. Every oneof1536currentcosts is independently explained by its frozenGU calendar plus nativeDown ownerload and an8ns finalPCIe packet. Otherresources/priority/FIFO/selector costs are relaxed in the oracle.

The common two-block budget can store all64experts' completeDown matrices in all4disjoint4-chip groups. This standardwholeexpert-copy control matches the independent-band oracle at chip-calendar level. Threeextra completeDown copies cost150,994,944usefulbytes, the same extra budget as onecompleteGU+DownFFNcopy ofall64routedexperts. No novelreplica/declustering primitive or allocatedcapacity advantage is claimed. One-block capacities are also disclosed: candidatehas only14/14/73/73freepages perchipplane versusRR43/43/44/44; the morelimited plane2/3copies are slower and cannot improve the unlimitedfast-copy bound.

The oldsame-allocated wholeGU replica witness and priorbounded replica no-gos are retained as context; their service regimes differ from this CRmodel. Adjacent sameexpert-band coalescing is outside the fixed384-row joboracle, as is a differentGU release calendar.

A simple equalGU split is separately rejected: two6wave nativejobs cost179,880ns/chip in compute alone. Adding actualfixedDown maxloads and sharedg16 gives meanexclusive-chip lowerbound355288.593750ns, above the achieved338,596.218750ns before any freshinput/command/output costs. HeterogeneousGUcuts or differentGU/Downsupports remain an untested headroom direction; no newexperiment is launched.

This is TRAINonly conditionalsoftware accounting, not hardwaremeasurement, taskquality, numericalABI evidence or a globaloptimum over changedjobgeometry. Reproduce: `python /home/ericyu/expert_flash_experiment/scripts/research_20261102_cr_ipr_down_replica_headroom.py`. Fullcommands, SHA256inputs, perqueryarrays and localassignment witnesses are retained in report.json/query_oracle_arrays.npz.
