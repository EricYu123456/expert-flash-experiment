獨立 heldout 成本審核通過。全部202,752成本與下界、792份完整trace（675,372節點），以及所有模型／每題／dataset統計均重現。檢查未發現shared或Deep dense prefix漏算／重算。

| Model | ONFI copies | Command ns | 完整FFN平均改善 vs cyclic | 慢token | 改善prompt均值 |
|---|---:|---:|---:|---:|---:|
| deepseek | 1 | 250 | 3.23436% | 0/256 | 16/16 |
| deepseek | 1 | 5000 | 1.87786% | 0/256 | 16/16 |
| deepseek | 4 | 250 | 3.25753% | 0/256 | 16/16 |
| deepseek | 4 | 5000 | 1.86768% | 0/256 | 16/16 |
| granite4tiny | 1 | 250 | 2.64044% | 3/256 | 16/16 |
| granite4tiny | 1 | 5000 | 0.00000% | 0/256 | 0/16 |
| granite4tiny | 4 | 250 | 0.57975% | 66/256 | 16/16 |
| granite4tiny | 4 | 5000 | 0.00000% | 0/256 | 0/16 |

Deep四種固定prefix成本：copy1/250ns=1.852224ms，copy1/5000ns=1.994236ms，copy4/250ns=1.888128ms，copy4/5000ns=2.013724ms；每token只加一次。

這是未校準ordinary-read軟體服務模型的重現，非裝置量測。G4 copy4保留66個變慢token；G4高command全部相等。容量公平性亦未解：swap8強控制後G4 whole6.0GiB、cyclic5.8125GiB。16prompt各16decode位置有相關性，不能視為256獨立task。數值ABI、FIFO、輸入廣播與PE時序仍需硬體證據。
