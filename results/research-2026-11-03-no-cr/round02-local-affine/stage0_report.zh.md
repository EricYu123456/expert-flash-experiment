# 第二輪 Stage0 結果：未定，留在同一輪

第二輪已完成擬合、驗證與保留評分；黑臉與 Root 的結論均為未定，科學失敗次數不增加。第一輪有界失敗仍為唯一一次，剩九次機會，包含目前第二輪。

本輪以原生 GGUF 軌跡上的共同輸入，查詢選定專家的 canonical W8/A8 函數，再比較共用輸入分區的局部 affine map。DeepSeek 含原生 shared expert；OLMoE 沒有。只測四個固定模型／層組合中的八個專家，沒有完整模型品質或速度結論。

## 主設定的保留評分

C32、r128、normal PCA、unweighted、120 FIT、unguarded。下表是自然路由命中的算子 SSE／teacher energy 點估計；R1 有相同分區的自由 bias 與最佳 pooled slope。所有欄位均為描述性結果，沒有把整族未定改成子集顯著。

| 模型／層／expert | Prefill local | Prefill R1 | Decode local | Decode R1 | Decode 命中列／提示 |
|---|---:|---:|---:|---:|---:|
| deepseek L1 E14 | 0.0366 | 0.2662 | 0.5330 | 0.7936 | 27/21 |
| deepseek L1 E40 | 0.0502 | 0.2910 | 0.1483 | 0.4090 | 49/21 |
| deepseek L13 E10 | 0.1357 | 0.5169 | 0.7454 | 0.8011 | 9/8 |
| deepseek L13 E56 | 0.1387 | 0.3717 | 0.4795 | 0.6753 | 113/30 |
| olmoe L0 E6 | 0.0120 | 0.1333 | 0.1140 | 0.2699 | 182/40 |
| olmoe L0 E17 | 0.0091 | 0.2396 | 0.0307 | 0.4724 | 23/14 |
| olmoe L8 E3 | 0.1161 | 0.4962 | 0.4425 | 0.7389 | 153/34 |
| olmoe L8 E9 | 0.0781 | 0.3990 | 0.4414 | 0.7408 | 45/21 |

八個主設定的 prefill／decode 差值均較 R1 低。相同凍結 R1 在所有這16個評分點的工作量與儲存成本均不高於 local：每次 active coefficient MAC 和 host projection MAC 均262,144；此算子保留的完整 metadata/storage 為 local10,478,072B、R1 2,097,656B。這些是邏輯工作與儲存量，不是延遲量測。

## 為何不能判成功或失敗

Deep L13 E10 decode 僅9列命中、分布在40個提示中的8個。按凍結 seed 與10,000次 prompt bootstrap，draw2748及5167沒有正能量命中，造成原本全族推論未定。全部10,712組比較保留：1,438 exact function ties、1,148直接或合成的分母未定、其餘8,126受全族未定影響；沒有重抽、刪除或事後改分組。

Decode 誤差比 prefill 高，120份FIT也沒有建立學習曲線平台。FIT列以prefill為主，這只是校準目標與分布的限制；不能推論 prefill/decode 的計算階段本身是因果原因。單專家相對誤差也不等同 live-alpha 混合誤差、原生 shared branch 的影響、完整模型 NLL／任務品質或真正延遲。

## 執行與後續

FIT／VALID耗時7,971.88秒；SCORE耗時1,396.99秒並正常exit0。6,664配置點、10,712比較、1,961個FITVAL依賴檔案及原凍結sources都已驗證，四次FIT／VALID capture及兩次SCORE capture皆正常。離線實驗峰值記憶體不代表目標8GB部署可行。

查看評分數值前已凍結同輪強對照：完整輸入的全域低秩G-RRR、局部映射共用輸出基底PE-RRR；包含固定精度／秩、真正主機浮點與native整數路徑、N80／120重新學習supervised基底與完整成本。現階段只允許實作及合成驗證，實際資料執行仍須source review。後續重用既有SCORE會標成development；TEST100保持封存。

目前未量得任何完整模型延遲改善，因此沒有「整體提升≥10%但無投稿價值」項目可列。
