新的公平比較把原六 experts 的 optimized raw-x（各自已選 β／λ）完全保留為獨立 fallback。用該模型的實際 canonical W8A8 TRAIN 殘差擬合 shared128／256 小 B，再把 head fold 成單一 concat readout、重新 row-W8、whole-vector A8 與 TRAIN bias 校準。β={0,.5,.9,.99}、B ridge={.03,.3,3} 的選擇只用 calibration-VALID。

| Layer／Expert | shared128 最佳 VALID 誤差改善 | shared256 最佳 VALID 誤差改善 |
|---|---:|---:|
| 1／14 | -0.0079% | +0.0365% |
| 1／40 | +0.0709% | +0.0605% |
| 13／10 | +0.5598% | +0.6846% |
| 13／56 | +0.0763% | +0.2033% |
| 26／0 | +0.5156% | +0.5070% |
| 26／1 | +1.6183% | +2.2181% |

改善均以 selected-alpha² 加權的原生 expert 輸出 NMSE 計算。前兩層最大只有 0.685%，晚層 E1 最大 2.218%，此組候選未達約 10% 的投入門檻。VALID 選過超參數，所以這些數字也不是 blind TEST 的改善估計。

原先固定 ridge=1 的晚層 FP conditional 增益有明顯 shrinkage 混淆。保持 raw-x ridge=1 與原 indices，僅把 z-on-x 的 P ridge 從 1 降為 .01／.001 後，E1 的 shared128／256 最佳增益由約 14.56%／18.79% 降為 4.05%／4.97% 與 3.51%／4.21%。這比直接把舊增益歸為共享非線性訊號更符合證據。

144 個 B 候選與 48 個 β 匹配 B=0 控制均已評估；匯出各 width 的候選、帶原 raw fallback 的 registry 與 B0 registry。原 raw 所有 payload hash 不變，且獨立重算的 VALID 分數完全相同。所選 heads 與 B0 共 344,064 個輸出 floats 與 frozen C++ 逐 bit 一致；concat gather／F32 transform 亦已核對。B=0 concat 和原 raw 仍有小差異，因此沒有假定兩者 bit-equivalent。

shared128／256 每次替換的 raw 係數節省率為 48.48%／45.45%，raw-x 為 51.52%。因此還需約 6.25%／13.33% 更高、且品質相當的替換覆蓋率才能補回寬 head 的係數工作成本；本次沒有證明這種 coverage 改善，也沒有測量 flash 或部署速度。保留 raw fallback 的額外 resident bytes 仍須另計。

全部大產物與可直接供既有 adapter 使用的六列 registry 位於 NFS 的 spec2-frozen-raw-residual；BOS／非 BOS／早晚位置切片與 TRAIN／VALID 指標可由各 layer 的 selection.json 讀取。此次沒有開新 TEST。
