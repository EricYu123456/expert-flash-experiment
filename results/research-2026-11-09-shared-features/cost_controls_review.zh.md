成本審查針對 source SHA d55752507b9c76779127c5da17ec7c0c6636c87cc47b905157d672aa3a61fb60 的 research_20261109_cost_audit.py；主 agent 後續修正時可逐項核對。GU／D／expert 的 W8+scale 單位正確，但現版 ranking 與分母只取三層，對最終單層小 cohort 的 native cache 強控制不足。

已凍結兩份全 26 層 VALID-only cache 排名：NLL positions0..62 的 v1 與 nonBOS positions1..62 的 v2。它們可按任意非負 byte budget 取 GU／full-expert prefix；同份 VALID 估計成本偏樂觀，主要成本須用政策凍結後的 1000 TEST。

| Cohort | 方法增量 host MiB | nonBOS 方法／matched raw 節省 | 同 budget native GU cache | 同 budget native full-expert cache |
|---|---:|---:|---:|---:|
| 3 | 13.830 | 0.1902% | 0.2926% | 0.2380% |
| 6 | 22.127 | 0.3113% | 0.5350% | 0.4391% |
| 9 | 30.424 | 0.4610% | 0.6530% | 0.6256% |
| 12 | 38.721 | 0.6196% | 0.8840% | 0.8027% |

此表只列 native VALID 路由上的 uncached per-event 係數存取工作反事實，分母為全 26 層 routed FFN；不包含 shared FFN、attention、LMhead 工作，也不是 flash IO 或速度。所有 readouts 留 host 時，相同 eligibility／GU cache 的 raw 與 nonlinear readout 的這項工作完全相同；差別必須由品質、可替換覆蓋與 host 容量說明。

Host 常駐不代表計算免費。必須另列實際 required prototype rows、可抵扣的 native-owner overlap、額外 GU MAC、readout MAC、transform/A8 contexts 與 host traffic。width1408 時，一個 row 僅替換一個 expert 且 owner 未被選中，GU 2HI＋C HI 正好等於原 expert 3HI；沒有算術節省。learned GU 即使沿用 owner ID，也不能當作原生 self hit 抵扣。

原生全模型 modeled weights 約 14.718 GiB，尚需 KV／workspace 等容量。若 host budget 放得下，完整 native 常駐就是合法強控制，routed flash 係數工作為零。若採較小的增量 budget，需清楚列出共用 shared FFN／其他 weights 的常駐前提，以及 fallback 位在 flash 還是 host。

BOS router 重複只是一項觀察。208 KiB 的全層 F32 aggregate cache 是有條件的 payload 算術，尚未實作並驗證全層 bit-exact replacement，不能報成已驗證 exact cache 或實測收益。完整優先問題、actual rawsubset inventory 與數值表保存在 cost_controls_review.json。
