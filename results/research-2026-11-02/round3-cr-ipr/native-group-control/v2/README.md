# 固定 source、完整 page-row group controls

GU採既有g8 whole-expert groups，所有wave都讀四個selected planes的相同WL，固定0123。沒有inactive頁、IPR、dummy、可程式source或ECC前drop。PB壓力明確終止CR run並計入新的cold28µs；不新增buffers或input contexts。

v1在第一個event前因unsupported enum拒絕，未得到效能結果；v2改為reject所有unselected pages。更正紀錄與v1來源保存在[erratum](../erratum.json)。

| array | Down | input copies | mean µs | event P99 µs | g2平均改善 |
|---|---|---:|---:|---:|
| cr | g4triples | 1 | 308.431521 | 355.966 | 0.000% |
| cr | g2availability | 1 | 282.957219 | 329.868 | 8.259% |
| cr | g4triples | 4 | 313.048302 | 359.806 | 0.000% |
| cr | g2availability | 4 | 287.574000 | 333.708 | 8.137% |
| ordinary | g4triples | 1 | 586.895818 | 720.666 | 0.000% |
| ordinary | g2availability | 1 | 579.681516 | 712.828 | 1.229% |
| ordinary | g4triples | 4 | 591.512599 | 724.506 | 0.000% |
| ordinary | g2availability | 4 | 584.298297 | 716.668 | 1.220% |

每列1536既有TRAIN layer-events，FIFO128全部通過；g2相對同arraymode、同inputcopies的g4triples沒有配對退步。g4payload244.5MiB，g2為289.5MiB，不是等payload優勢。

移除CR後平均收益只剩約1.2%，不足以主張對所有in-flash平台都有大幅收益。這個ordinary模型仍是特定28µs／PB／ECC服務模型，不能代表各平台的真實read/compute primitive。Coldreset仍需保留同一native GEMV的input及列內accumulator，該ABI尚待實體確認。

[凍結協定](protocol.frozen.json)、[機器結果](report.json)、[完整陣列](service_arrays.npz)。
