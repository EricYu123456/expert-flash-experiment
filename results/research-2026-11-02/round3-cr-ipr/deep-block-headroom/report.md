# Deep two-block common-input GU headroom

ExistingW8/A8TRAIN26layers ×512inputs. Geometry andNAND-onlycold-costscreening; notnativecompletionlatency. Allmatchedcontrols have2blocks,1452programmedpages/chip,176selectedpages/query/chip,onecopyandnooverfetch.

| Control | Meanwaves | P99waves | Meanactiveblocks | NAND-onlymeanus | NAND+ECCnecessaryLBmeanus |
|---|---:|---:|---:|---:|---:|
| flat_global_RR | 45.282527 | 47.0 | 1.992939 | 475.711291 | 480.421774 |
| whole32_RRphase_shared0 | 45.272461 | 47.0 | 1.992563 | 475.606776 | 480.389832 |
| whole32_uniform4phase_shared0 | 45.772837 | 47.0 | 1.992563 | 480.460419 | 482.691283 |
| striped11_global_phase_continuation | 46.547551 | 48.0 | 2.000000 | 488.111245 | 488.594410 |
| striped11_uniform4phase | 46.479492 | 48.0 | 2.000000 | 487.451074 | 488.047852 |
| flat_RR_TRAIN_frequency_sharedfirst | 45.248648 | 47.0 | 1.843525 | 472.648385 | 480.258489 |
| whole32_TRAIN_frequency_shared0 | 45.248648 | 47.0 | 1.843525 | 472.648385 | 480.258489 |
| striped11_TRAIN_rare_capacity_repair | 44.394832 | 62.0 | 2.000000 | 467.229868 | 481.839817 |
| striped11_TRAIN_phase_coordinate | 45.452524 | 47.0 | 2.000000 | 477.489483 | 481.237764 |
| generic_TRAIN_block_plane_onepage_moves | 44.404898 | 62.0 | 1.993765 | 467.213409 | 481.595228 |
| legacy_compensation_3blocks_reference | 53.681115 | 58.0 | 2.997897 | 575.568322 | 575.568322 |

Eachselectedpage contains8completeC2048rows, so theexplicitfixed0123pageconsumeorderpreservesfulldots withrow/expert scatter. Aplane thatrunsoutstaysidleuntilthenextblock; allactiveplanesstartcoldinthenewblock. No idleplane is silentlykeptCRhot.

Global44waves alone isonlyworkLB. Universal44foralltop6 wouldrequireminimalmaxresident392pages/plane, exceeding384pagesavailablein2blocks. Thisis anecessarycapacityargument, notnativeexecution/PBtheorem ornewalgorithm.

Frequencyordering,hotblockplacement,waterfillcapacityrepair,phasecoordinatefit andonepage8binmoves arestandardstrongcontrols. AllTRAINfitsreuseobservedroutes. The3blockcompensator is anextra-capacityreference, notcapacitymatched.

FinitePB/ECCpipeline,command/input/outputcosts andfullFFN areoutside thisscreen. The6.4GB/s bound is necessaryonly andcanmasksmallNANDwave improvements. No measuredhardwareperformance ornewlayoutalgorithm claim.
