# 第二個模型 shape 的 native Down kernel envelope

這是路由無關、完整input／完整row的微核心掃描，沒有延伸成第二模型完整FFN效能或品質結果。

| shape | group width | pages/job | strict CR | explicit coldreset µs | ordinary µs |
|---|---:|---:|---|---:|---:|
| granite4tiny | 2 | 24 | 89.940µs | 89.940 | 181.240 |
| granite4tiny | 4 | 12 | 59.220µs | 59.220 | 95.740 |
| deepseek | 2 | 88 | PB overwrite，無效 | 272.540 | 637.240 |
| deepseek | 4 | 44 | 141.140µs | 141.140 | 323.740 |

DeepSeek的兩-chip shard需要22個full waves，copy500下strict CR不成立；explicitcoldreset才能跑到272.540µs。四-chip的11-wave shard為141.140µs，若一chip串接兩個nativejobs則282.280µs；這個相加沒有計FFN input/output calendar，不能當完整model gain。

Routed與half-shared每個對應geometry的係數bytes相同，但不同C使row milestones不同。本掃描把兩種role都實際驗證。較一般的編譯器需要根據完整row邊界、native cold啟動、有限PB及block容量選擇geometry；G4的六-wave結果不能直接套用所有shape。

[64個microkernel cases](report.json)、[凍結協定](protocol.frozen.json)。
