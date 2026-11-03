# Physical pool source／first-principles screen — Sol

## 授權與時間

- Root任務：只做8–12分鐘source／原理篩選，最多兩個具體機制；不是採納方法或正式30分鐘時計。
- 開始receipt：2026-11-02 21:09:15 UTC。
- 前pairconstruct已由Root於21:08:29 thought-reject；僅Round1一個有效實驗陰性，剩9。舊時計不沿用。
- 不寫候選code，不讀新模型weights／activations／routing，不跑數值probe；本檔只筆記／來源／符號算術。

## Pool 1：WL-DualSense／4-state two-page one-shot

已讀Gemini Block3，新主張為每TLCcell只用四個states，center一次sense出高logicalpage，兩outerrefs補低logicalpage；full W4兩cells/weight，單WL產32KiB，想從普通兩LSB56us變joint46us；hi-only28us作W2。其不保跨WL電位，概念上可與no-CR契約相容，但AiF沒有驗證此命令。

### 來源核實與新穎性界線

1. [US9851899B2 Fig14–16](https://patents.google.com/patent/US9851899B2/en)：main/cache/auxlatches支援one-shot多logicalbits；更直接的是Fig16說明段落（頁面行951–954）明說雖仍需逐bitread，read preparation只做一次，列init／XDEC／PUMP。因此『同WL多頁共享readpreparation』已有直接primary先例，不只是泛泛latch容器。
2. [US9230677B2 claim9](https://patents.google.com/patent/US9230677B2/en)：MLC2×Nbit對2Nbitcache、TLC3×Nbit對3Nbitcache；其HiNAND分段／多WL架構並非AiF免費能力，只能證多頁多cache思想舊，不能將其200ns或charge-sharing性能套到AiF。
3. AiF `paper/aif.txt` §4.3/4.4明說四page latches為一cache＋三readbuffers；be-enc單sense只需一readbuffer，另外兩readbuffers才挪作8–32KiBinput。Table2 (1,3,3) 的28／46／46us都是單logicalpage，不證joint32KiB46us。
4. Progressive2/4bit對照還須給AnyPrecision／DPA等同one-shot硬體；如果只在其上做alpha／sensitivity選precision，不能聲稱新device+algorithm。

### 必須閉合的具體機制

- 16KiB logicalpage有131072cell-bits；4state cell兩bits，一WL32768bytes；W4每weight兩cells，因此65536weights/WL。高page含每weight兩個hi bits，低page補兩個lo bits，尺寸自洽。
- 4state排序用Graymapping可令center給一個bit、兩outerrefs合出另一個bit；三個全局refs而非兩個是普通globalWLreference幾何的自然結果，不是新senseprimitive。
- 兩inputlatches已占後，僅剩main/cache兩latches；hi要保存在cache時，lo的兩次sense是否要main+第二scratch，必須有實際PB布林合併／覆寫時序。若要暫停／恢復ECC或cache-drain，也不憑空有第五latch。
- FullW4最好重建整個4bitweight後單MAC；hi/lo分開partialMAC雖可重疊後18us sense，卻把perweightMAC次數翻倍，I32partial／row-scale套用也必須正確。ECC throughput不等於PEMAC throughput。

### 只作示例的完整MAC算術（非測量）

4planes×32KiB W4包含262144weights。若假設PE6.4GMAC/s（AiF沒有給足證據，故僅參數例），單fullpass需40.96us；hi或lo各pass也需40.96us。單passdual46+40.96=86.96us；兩LSB W4各16KiB／plane共131072weights，每pass20.48us，兩次共2×(28+20.48)=96.96us，差10.31%，而非sense-only17.86%。若分hi/lo兩MACpasses且hi在28us開跑，理想max(46,28+40.96)+40.96=109.92us，反而較慢。此例忽略ECC／latch transfer／commands，只為阻止免費overlap過稱；不能作實際性能結論。

開放問題：pseudoMLC的BER／outergap與centererror哪個state主導，原AiF160chip數據不公開；ER/P2/P5/P7的spacing不代表已知分布寬度／ECC_LITE可行。模型格式若支援high-onlyW2，必須真正qualitymatched，而不是引用BitsMoE的W2退化後便稱fullW4消除了品質問題。

## Pool 2：長 frame entropy stream＋die decoder

不能把舊16／64KiB frame ceil後0wave視為entropy普遍無效。若每plane頁16KiB、四plane同步wave64KiB，原始frame有n整wave、無header且壓縮ratio c，省第一wave需`ceil(c n)<=n-1`，即`n>=1/(1-c)`。以舊sample的entropy僅作示例：h7.3bit→c.9125需至少12waves=768KiB；h7.7→c.9625需至少27waves=1.6875MiB。256KiB四waves在c>.75仍四waves，1MiB十六waves在h7.3可變15、h7.7仍16。這些是symbolic，不是新資料或碼率觀測。

Ideal g16整expert/chip Deep約528KiB=9waves，c.9125後ceil8.21仍9；OLMoE約384KiB=6waves仍6。大Mixtral expert每chip更多waves，原則上可出真gain；不能由小shape否決所有模型。若跨GateUp／Down長stream合併，最後GU頁中已讀的Down bytes要在nonlinearhostbarrier與inputPB覆寫間保留，否則重新read會吃掉所省wave。跨experts合併又可能讀到未selectedbytes，不能以全E平均長度免費攤尾頁。

### 新增極直接 primary

[Approaching Shannon Bound with Lossless LLM Weight Compression](https://arxiv.org/html/2606.15789v1) §IV/V已有layer-sharedcodebook、tile-offsetindex、independentANSsubstreams、fuseddecompression/GEMM與doublebuffer，格式含INT8／SQ8／INT4／AWQ，而非僅BF16。其§VI-B聲稱自家INT8可4–5bit、INT4 .6–1.0bit；不能套用repo的rawrowW8，也不能用其他model的entropy推本機碼率。它還明列小batch可能decoderbound，正是edgeB1必須入帳的成本。

[ZipMoE](https://arxiv.org/html/2601.21198v1) §2.2主要BF16exponentskew、LZ4HC／ZSTD與cache-affinityschedule；不能當rawW8已實證的壓縮率。[QMoE](https://arxiv.org/html/2310.16795v1)為GPTQ低bit／ternary後的entropy/dictionarydecode，壓縮與執行co-design早已有。Root指名US20260149462A1目前webopen未能存取，這裡不冒充已核其claims；iliriadocs是作者公開source，非paper品質實證，亦已有量化weightstream coding思想。

目前判斷：長frames可改ceilregime，但單純frame大小／ANS移植不足投稿新意。除非提出NAND-native decoder／PB carry／substreaminterleave等有清楚新必要能力，與既有同entropycodecbaseline比較後仍有大差，否則不建議正式clock。Read數少不等walltime少：decoder的符號/s必須追上PE，若Ccompressed變小但decodedW8仍同weights，MAC完全不減；高熵如7.3–7.7bit時8.75–3.75%bitheadroom常被新增decode／tables／尾頁吃掉，尚不可宣稱實際上界。

## 21:14:33 UTC receipt 後：PB 排程與比較前 MAC 的可檢驗變體

不能只數latches就宣告全方案不可能。設四states升序，C_lo／C_mid／C_hi表示超過三threshold的bits。Graylowbit可為`C_lo AND NOT C_hi`。若full-W4已事前確定，或可先outer_lo、outer_hi，用兩個free latches做in-placeBooleancombine，覆寫成lowbit並釋一latch，再center讀hi，最終main/cache放兩logicalpages、另兩input保持。這需要實際PB布林運算／回寫路由與時序證據，非AiF已證能力；更不提供28us時先算hi，因hi到第三sense才ready。Center-first若要高位早算，則hi保留與兩outertemporary的resourceconflict較嚴重。

Pool1內較具體但未驗證的變體是 **raw-comparator→bitserialMAC streaming，而不先物化兩個logicalpages**。數值ordered4state digit滿足`d=C_lo+C_mid+C_hi`；兩digits可構成W4 `q=4*d1+d0-8`，因此逐threshold 1-bit頁可直接加權累加，處理後釋PB，不需hi/lo全頁同時常駐。每W4weight需6個binaryweightedadd，而非1個4bitMAC；這不是免費算力，PE需實際binarylane吞吐／adder／accumarea。其負offset可用共同sum(input)補，但rowgroup量化格式仍需一致。

這個變體的primary風險非常具體：

- [US20190102359A1](https://patents.google.com/patent/US20190102359A1/en)已做thermometercodedmultibitMAC，為SRAM類analogCIM、將thermometercode物理展開；不能說thermometer arithmetic首次。
- [MCFlash2605.05119v1](https://arxiv.org/html/2605.05119v1)已用standardusercommands／shiftedreadrefs／SBR操作MLC rawbits，測50chips；它明說bypass ECC與randomizer，某些操作RBER>5%，SET_FEATURE切換有<10usoverhead、普通read40–90us。不能將其當AiF28us／46us／rawcode ECCfree的證據。
- Rawthresholdplanes不是標準Graylogicalpage ECC codewords。若要exact需重設parity／decoder或先做rawMAC再correctpartialsum，後者又有concurrentECC+MAC先例；若放棄ECC而走qualitymatched，須真cellBER／adjacentstateerror分布與free-running品質，不能拿iidbitflip代替校準。

此variant至少說得出新增interface與解除的PB依賴，較單純one-shotmaterialization具研究問題；但尚未確認其novelty或小改動，並非已採納／建議立即正式clock。

### Precision 與 E2E 分母修正

PseudoMLC high／low兩logicalpages不必只對W2／W4：W8用4cells/weight，每cell配一個hi4bits的bit與一個lo4bits的bit，也可highlogicalpage給W4、full兩頁給W8。此時highpage32768weights／28us，與普通LSBW4的informationrate相同；fullW8若真可32KiB／46us則仍優於兩LSB56us。所以W2品質退化不是物理pool的單獨否決證據，但DPA／AnyPrecision4/8則更加直接是同硬體強baseline。

若fraction f選high-only，其餘full，sense-only平均`28f+46(1-f)=46-18f`；相對sameprimitive全精度baseline的改善`18f/46`。這還須乘sense在完整模型delay分母的實際佔比，並算ECC／PE／latch／commands。17.9%one-shotvs兩LSB與precision省bytes不能重複記兩遍。

四statefullread的符號成本為19+3×9=46，8stateallbits則19+7×9=82，若套相同toy分解，bits/cell/time在b1=.03571、b2=.04348、b3=.03659，b2恰是sweetspot。但這只是把Gem的19／9假設延伸的代數，非任何測量、也非MLCread效率首次發現。

## Screen 收斂（21:17 UTC 後，無正式 candidate clock）

兩個pool都不建議按原版本直接採納：

1. WL-DualSense原materializedtwo-pages＋progressivequantization版本，核心same-WL one-shot／prep攤提已有直接prior，指定46us與免費MACoverlap未證。較可檢驗的剩餘問題是raw-comparator→binaryMACstreaming能否解除PBscratch衝突、在含ECC與PE新area後仍有真收益；此variant尚須primarynovelty與硬體契約，現在不夠凍30分鐘正式時計。
2. 長frameentropy可修正ceilregime，舊0wave不是所有codec陰性；但tile／ANS／fuseddecode先例直接。marginal7.3–7.7bitentropy也不是conditional／跨symbol編碼的普遍下界，只是memoryless碼率參照。需要真正新NAND-nativepredictor／code／decode-dataflow，而非把frame變長便叫新法。

本screen自21:09:15開始，進行source核讀、PB布林時序、MACthroughput條件、量化映射與ceil推導；沒有新模型／activation／routing probe、沒有實驗程式、沒有候選正式時計。最終receipt由tool回Root，仍1個validfail／9remaining。
