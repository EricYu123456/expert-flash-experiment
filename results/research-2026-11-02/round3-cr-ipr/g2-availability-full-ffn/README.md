# 兩-chip Down 分片與稀疏副本：完整 FFN

這是固定 W8/A8、條件式 NAND/PB/ECC/PE 模型，非實體晶片或完整模型推論測量。固定既有 GU，改以六個兩-chip routed groups 及四-chip shared 專區；每 routed group 放42個 experts，總252份完整 Down copies。

每 routed native job 為 R768×C512；shared 每 chip 為 R384×C1024。兩者都24pages、六個完整 same-WL waves，固定0123，copy500模型為89.940µs。GU仍沿用明列的 source/drop 條件。

| GU | input copies | priority | mean µs | event P99 µs | 相對同 GU g4 triples 平均改善 | fallback |
|---|---:|---|---:|---:|---:|---:|
| candidate | 1 | GF | 254.712000 | 254.712 | 9.202% | 0 /1536 |
| candidate | 1 | DF | 254.712000 | 254.712 | 9.202% | 0 /1536 |
| candidate | 4 | GF | 258.856000 | 258.856 | 9.313% | 0 /1536 |
| candidate | 4 | DF | 258.856000 | 258.856 | 9.313% | 0 /1536 |
| RR | 1 | GF | 256.793182 | 271.868 | 8.903% | 0 /1536 |
| RR | 1 | DF | 256.793182 | 271.868 | 8.903% | 0 /1536 |
| RR | 4 | GF | 261.705182 | 276.780 | 8.751% | 0 /1536 |
| RR | 4 | DF | 261.705182 | 276.780 | 8.751% | 0 /1536 |

每一列為32個既有TRAIN prompts ×16decode ×3layers；512個對齊 inputs，並非1536個獨立樣本。所有 profiles 的 FIFO peak 均128B，全部配對都比同 GU 的較低容量 g4 triples 更快。

**容量成本：289.5MiB/layer，較 g4 triples 的244.5MiB增加45MiB。** 相對原單份 Down 增加141MiB。高 GU plane128WL加 Down252WL=380/384；每個 private job 完整位於單一 block。186WL GU副本不能套到這個已封存布局。

這是標準 replicated declustering／CBC matching 加上 native shard geometry，不是新CBC或solver。TRAIN cap1滿分不代表普遍保證。精確均勻 top6檢查有448,188/74,974,368=0.5977883%的cap1失敗；cap2能完成，但延遲顯著增加。

[完整機器結果](TRAIN512/report.json)、[凍結協定](protocol.frozen.json)、[獨立actual trace與地址綁定稽核](../g2-availability-full-ffn-independent/report.md)、[均勻支持集合反例](../g2-availability-headroom/uniform-addendum/report.md)、[Down-only條件下界](../g2-availability-headroom/stage-certificate/report.md)。

後續16個新自編 prompts 的凍結評估仍有平均改善，但TRAIN配置出現3/768個fallback；IDslot配置11/768，P99退步。完整見 [prospective結果](../prospective-full-ffn/README.md)。
