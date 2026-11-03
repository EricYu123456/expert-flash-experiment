兩-view完整row副本構造審核通過，但原生數值ABI仍待證。主variant僅交換一對s↔(s xor8)，其餘14chips固定。

在B的logical row ownership可由構造決定時，額外rows下界δ=½Σ|A−B|可達：保留A的common prefix，把source tail複製到target suffix。A/B都各覆蓋全部rows一次。若B的logical rowsets已固定，此公式不保證可達。

| 主witness | 副本rows | 真係數bytes | 額外pages／wave bytes | 每viewcommands／waves | 短view需drop完整outputs |
|---|---:|---:|---:|---:|---:|
| granite4tiny.s0.t8 | 43 | 66048 | 65536 | 16／25 | 2 |
| granite4tiny.s3.t11 | 42 | 64512 | 65536 | 16／25 | 2 |
| qwen38_101_75_singleedge | 26 | 66560 | 65536 | 16／51 | 1 |
| qwen38_101_76_singleedge_samechannel_reordered | 25 | 64000 | 65536 | 16／51 | 0 |

已驗29,438組小型A/B向量、實際v4前3層5,952個single-edge候選、27,648份原logical row-start描述。未讀route或heldout，未改原plan。G4v4geometry不冒稱W8v5/v6winner。

同channel配對保留總NAND、ONFIinput/output、commands；機會僅來自chip occupancy與GU→Down ready時間重新分配。需比較同物理image的static A/B、route-blind selector、query-log-aware replicated declustering與64/1024組合的小exactvieworacle。

現有AiFchip的C/block/offset/pagecount欄位可描述prefix地址，但stock aif_post沒有展示sharedprefix aliases與row scatter。短view會讀到真append rows／partialrow；outputcount、drain/drop、accumulator reset與row-separability契約仍未公開確認。不能由代數證明推出native實測。

最直接prior包括[多fragmentation共享overlap（GvDB2016）](https://ceur-ws.org/Vol-1594/paper7.pdf)、[query-log-aware selective replication（TPDS2013）](https://repository.bilkent.edu.tr/bitstreams/3e48b3b0-a708-4f57-9eb5-321cbd19928c/download)。它們覆蓋shared intersections、partialcopies與query-conditioned selection；本輪未找到同一NAND prefix構造的直接證據，但不作first或投稿突破主張。
