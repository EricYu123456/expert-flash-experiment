# Round2 Stage2：自然文本局部品質篩選

固定的八算子版本通過預設2%相對subset-PPL預算，仍在第二輪；科學失敗數維持1、剩餘9次機會（含本輪）。這不是完整routed部署或投稿成功。

| 模型 | C相對B的PPL變化 | 兩模型共同計算的近似95%區間 | C相對原GGUF模型 |
|---|---:|---:|---:|
| deepseek | 0.292% | [-0.682%, 1.275%] | 0.327% |
| olmoe | 0.470% | [-0.222%, 1.167%] | 0.268% |

B是僅對相同八個算子換成canonical W8/A8的參考，C是刷新後局部仿射模型；其他函數維持原GGUF。PPL是每本256個token的自訂PG-19窗口，不能稱官方完整PG-19 word-PPL，不能跨模型tokenizer比較。D為預先固定PE6/G2 q512/W8/host-floatlatent對照；其C−D比較與所有原模型分解完整保留，沒有依驗證結果選winner。

採固定16本官方validation書，主分析32,768個NLL全部有限。每本書的四arm使用相同native joint-tokenization與GT512，但各自fresh KV；logits255..510評分256targets。10,000次共享書籍重抽樣、seed2026110302，固定bootstrap SD與centered max-|t|共同區間，cutoff2.209279491。Root從原始逐token分數獨立重算吻合。這是16書的近似發展性篩選，沒有有限樣本保證；OLM書42306的較大delta .05050533完整保留。

原FIT120與固定TRAIN16只做一次合併重建：Deep每op29,629列、OLM29,593列。新增教師110,592列／905,969,664 target bytes／826,244,333,568 integer MAC，共用於C與D。八op皆finite、0unsupported，archive/B1-batch/實際IPC通過。原資料補綁證明目前X經canonical函數等於歷史Y，不補造原始capture的歷史身分。

source/nohook、observer、copy及B/copycanonical於固定首本逐bit控制全部通過。470個FIT相依檔案在下載驗證書前與統計後SHA/stat相同；品質55個輸出檔Root獨立hash吻合。主程序05:06:35→05:18:01 UTC、OS0、686.02秒；八實體CPU核心，沒有GPU、重擬合或TEST存取。這些時間與RSS含原GGUF及被覆寫的原計算，是品質工具成本，不是壓縮部署延遲或8GB常駐證據。

## 證據範圍與後續

C在因果輸入0..510實際替換Deep3,603次、OLM4,772次，分別只約佔全模型routed slot calls的0.2825%與0.4560%。這不是NLL或延遲可按比例外推的分母。OLM L8E3只有5次scored-predecessor與16次prefix替換，不能單獨宣稱該expert品質已驗明。未使用的input511另列診斷。

黑臉與Root判定同輪續行：先固定強非線性／global／合法host-cache基準與可行成本窗，再決定完整routed覆蓋的前瞻版本和新保留驗證資料。已評估的八個maps與tree/P保持封存，不能用這16書回頭調參。all-host原OLM、低秩host及localbank快取都是必要基準；AiF短input、PE/FIFO/命令成本仍需具體合約。原合成TEST100與PGtest32保持封存。尚未量得完整模型≥10%延遲收益，亦未形成投稿結論。

- [完整執行收據](stage2-pg19-quality/QUALITY.executor_closure.json)
- [Root獨立統計核對](stage2-pg19-quality/Root_QUALITY_independent_statistics_audit.20261103.json)
- [黑臉正式判讀](stage2-pg19-quality/critic_scientific_adjudication.20261103T052217Z.json)
