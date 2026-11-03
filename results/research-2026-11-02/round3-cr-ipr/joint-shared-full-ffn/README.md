# Joint shared／routed Down 的完整 FFN 服務模型

**此處的完整FFN是有因果依賴的服務時間重播，不是放置後的實體晶片或模型推論測量。** 所有係數為固定W8/A8參考；GU→完整host activation/A8→私有完整Down input→加權combine的因果與輸出drain保留。

八個2-chip groups，各resident32E加兩個shared halves。每E存於四個groups；標準CBC匹配六E後，shared兩個完整列區段放到兩個空閒groups。每selected chip只有一個Down nativejob；R768C512與R384C1024皆24pages／6waves。無私有輸入融合、columnpartials、新MACcontext或buffer。

Payload303MiB/layer，其中GU99、routedDown192、sharedDown12。原g4triples244.5MiB、sparseg2為289.5MiB。共同容量上限32chip-blocks，新布局實際使用32，不能把usefulpayload與blockreservation混為一談。

| GU | array | input copies | priority | mean µs | event P99 µs | three-layer P99 µs |
|---|---|---:|---|---:|---:|---:|
| candidate | cr | 1 | GF | 254.712000 | 254.712 | 764.136 |
| candidate | cr | 1 | DF | 254.712000 | 254.712 | 764.136 |
| candidate | cr | 4 | GF | 258.856000 | 258.856 | 776.568 |
| candidate | cr | 4 | DF | 258.856000 | 258.856 | 776.568 |
| RR | cr | 1 | GF | 256.793448 | 271.868 | 790.366 |
| RR | cr | 1 | DF | 256.793448 | 271.868 | 790.366 |
| RR | cr | 4 | GF | 261.739448 | 276.780 | 805.102 |
| RR | cr | 4 | DF | 261.739448 | 276.780 | 805.102 |
| g8_fixed | cr | 1 | GF | 282.957484 | 329.868 | 930.164 |
| g8_fixed | cr | 4 | GF | 287.608266 | 333.708 | 942.756 |
| g8_fixed | ordinary | 1 | GF | 579.681781 | 712.828 | 1969.484 |
| g8_fixed | ordinary | 4 | GF | 584.332562 | 716.668 | 1982.076 |

每列1536既有TRAIN events，32prompts／512對齊inputs／3layers。全部PB與128B FIFO有效。Conditional candidate與RR使用原共同GU first-feasible policy、source-permutation／preECCdrop條件。Nativeg8為完整selected四plane、same-WL、固定0123；不需IPR、dummy、drop，PB壓力明確付coldreset。

Candidate主配置比g4三副本快9.202%，但增加58.5MiB；比sparseg2主配置TRAIN全相等，增加13.5MiB主要換到all-top6保證。RR對sparseg2有34個小幅退步（input1平均+0.265625ns；input4 +34.265625ns），不能宣稱所有GU routeconstant。

[同303MiB固定shared ablation](../joint-shared-fixed-capacity-control/report.md)隔離派送因素，candidate平均3.507%、eventP9925.297%、三層對齊P9918.330%改善；這不等於所有可學固定布局的平均最優比較。

第一批16新prompt是在本設計提出前已看過的診斷；[diagnostic-reuse/report.json](diagnostic-reuse/report.json)如實標重用。第二批16prompts在新配置與全部16個comparison profiles凍結後才執行：[新診斷完整報告](../second-prospective-routes/README.md)。未做新placed-logit／任務品質或原GGUF等價主張。

[完整TRAIN結果](TRAIN512/report.json)、[全部陣列](TRAIN512/service_arrays.npz)、[凍結協定](protocol.frozen.json)、[544jobs／Hall與地址驗證](../eight-pair-cbc-degree4/README.md)、[saved trace獨立重放](../joint-shared-full-ffn-independent/README.md)。

Fixed common-release Down-only stage可達89.94µs的byte下界；該證明不包括GU、host／command／input／output，不能加到任意GU下界變成fullFFN最優。八pair固定24pagefullcopy格式中maxresidentjobs最少34，亦不代表303MiB是任意rowlayout的總storage最小。

CBC／matching／replication均是既有元件：[一手文獻與貢獻界線](../joint-shared-prior-notes.md)。目前只完成手工G4instantiation和第二shapekernel負例，尚未建立可直接投稿的可重用後端貢獻。
