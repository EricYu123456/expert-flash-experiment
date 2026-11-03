# Wide common-input GU and compact private-input Down

TRAINonly:32prompts ×16decodeinputs ×3MoElayers. Same13Downmenus were screenedfor everyGUfamily on64fixedindices/layer, thenwinnersfrozenbeforefull512TRAINreplay. Thisis aconditionalservice-modelresult, nothardwaremeasurement.

| GU / selectedDown | copies | priority | meanus | P99us | validqueries | peakFIFO |
|---|---:|---|---:|---:|---:|---:|
| candidate / g4_reused_TRAIN_labels | 1 | GF | 338.596219 | 385.962000 | 1536/1536 | 128B |
| candidate / g4_reused_TRAIN_labels | 1 | DF | 338.596219 | 385.962000 | 1536/1536 | 128B |
| candidate / g4_reused_TRAIN_labels | 4 | GF | 344.970646 | 393.178000 | 1536/1536 | 128B |
| candidate / g4_reused_TRAIN_labels | 4 | DF | 344.970646 | 393.178000 | 1536/1536 | 128B |
| RR / g4_reused_TRAIN_labels | 1 | GF | 339.961401 | 395.392000 | 1536/1536 | 128B |
| RR / g4_reused_TRAIN_labels | 1 | DF | 339.961401 | 395.392000 | 1536/1536 | 128B |
| RR / g4_reused_TRAIN_labels | 4 | GF | 346.335828 | 402.232800 | 1536/1536 | 128B |
| RR / g4_reused_TRAIN_labels | 4 | DF | 346.335828 | 402.232800 | 1536/1536 | 128B |
| g8_fixed / g2_reused_TRAIN_labels | 1 | GF | 348.256142 | 465.672500 | 1536/1536 | 128B |
| g8_fixed / g2_reused_TRAIN_labels | 1 | DF | 348.256142 | 465.672500 | 1536/1536 | 128B |
| g8_fixed / g2_reused_TRAIN_labels | 4 | GF | 353.963040 | 471.922900 | 1536/1536 | 128B |
| g8_fixed / g2_reused_TRAIN_labels | 4 | DF | 353.963040 | 471.922900 | 1536/1536 | 128B |

Candidate+g4 meanimproves20.7205% overthesameGUwithg16Down atinput1/GF. RR+g4 similarlyimproves20.6545% overitsg16Downcontrol. CandidateoverequallyreoptimizedRR isonly0.4016%input1 and0.3942%input4;378events areslower and533equal. Genericcompilerexactlytiescandidate.

TheprivateDownC512vectorstayscompleteon everyjob. g2/g4 full0123waves arelegalunderstrict500nsPB model;g1twelve-waveburstisinfeasible. Coldstarts, commandcount andfullinputcopiesdecline; usefulcoefficients/readbytes andoutputsareunchanged. Resourceworkcounters areattributionevidence, notanadditivelatencydecomposition.

Alloptimizedtracesrequire128B underdeclaredpagecompletionburstgranularity;64B isinvalid. Everyselectedfullmodelconfigurationhas onecopy andfitsone192WLblock/chip in theadditionaladdresscertificate. Thecomparisons retained thesame2blockcapacitybudgetduringselection.

Themajorresultis operand-awareasymmetricsharding underCR, notanewsharedlayoutalgorithm. OldTRAINlearnedgroups areessentialstrongcontrols andcarrynoalgorithmnoveltyclaim. Noheldout,measurednativeperformanceorpublicationreadinessclaim.
