# MusicXML 與 YOLO Class 盤點及差異

## 1. 文件範圍

本文件以目前 Xia／BPSD 對齊資料為準，資料來源如下：

- YOLO class 定義：Xia `notes.json`。
- YOLO 實際數量：`output/dataset_inventory/class_counts.csv`。
- MusicXML：六首奏鳴曲的 `score_xml_repetitions/*.xml`。
- 盤點日期：2026-08-08。

重要觀念：YOLO 有明確的 `class_id` 和 `class name`；MusicXML 並沒有與
YOLO 共用的 class ID。MusicXML 使用階層式 element、attribute 和文字值描述
音樂內容。因此本文所稱「XML class」，實際上是可轉換成事件類型的 XML
element/value，例如 `<slur type="start">`、`<dynamics><f/></dynamics>`。

## 2. 整體摘要

| 項目 | YOLO | MusicXML |
|---|---|---|
| 資料形式 | 每行一個偵測框 | 階層式樂譜文件 |
| 身分 | `class_id`、TXT line、bbox | XML path、measure、note/direction/notation |
| 空間資訊 | `x`, `y`, `w`, `h` | 部分 element 有排版座標，但不是可靠的掃描圖座標 |
| 音樂時間 | 沒有 | measure、onset、duration、voice、staff |
| 音高與音符 | 通常只有圖形 class | pitch、MIDI 可推導、chord、rest、grace |
| 開始／結束 | 沒有 | slur、tie、tuplet、wedge、pedal、octave-shift 有 start/stop |
| 本批總量 | 162 個已定義 class；102 個實際使用；9,828 個框 | 20,854 個 `<note>`，另有 directions、notations、attributes 等 |

YOLO 回答的是「掃描圖片的哪裡看到了什麼圖形」；MusicXML 回答的是「這個
音樂事件在樂曲結構中的時間、音高、聲部與關係」。兩者必須經過 page、system、
measure、staff、時間與幾何位置對齊，不能直接依 class name 合併。

## 3. YOLO class 完整清單

「實際框數」為目前 49 頁資料的總和。`0` 代表 class 存在於 `notes.json`，但這批
YOLO TXT 沒有使用。

### 3.1 特殊文字與反覆指示

| ID | YOLO class | 實際框數 |
|---:|---|---:|
| 0 | `IlFine` | 1 |
| 1 | `LangsamUndSehnsuchtvoll` | 1 |
| 2 | `MarciaDaCapoAlFineSenzaRepetizione` | 1 |

### 3.2 臨時記號、琶音與演奏法

| ID | YOLO class | 實際框數 |
|---:|---|---:|
| 3 | `accidentalDoubleFlatSmall` | 0 |
| 4 | `accidentalDoubleSharpSmall` | 3 |
| 5 | `accidentalFlatSmall` | 0 |
| 6 | `accidentalNaturalSmall` | 24 |
| 7 | `accidentalSharpSmall` | 20 |
| 8 | `arpeggiato` | 0 |
| 9 | `articAccentAbove` | 3 |
| 10 | `articAccentBelow` | 13 |
| 11 | `articMarcatoAbove` | 0 |
| 12 | `articMarcatoBelow` | 0 |
| 13 | `articStaccatissimoAbove` | 0 |
| 14 | `articStaccatissimoBelow` | 0 |
| 15 | `articStaccato` | 1 |
| 16 | `articStaccatoAbove` | 447 |
| 17 | `articStaccatoBelow` | 212 |
| 18 | `articTenutoAbove` | 0 |
| 19 | `articTenutoBelow` | 0 |
| 20 | `beamSmall` | 88 |
| 21 | `caesura` | 0 |
| 22 | `coda` | 0 |

### 3.3 力度與漸強漸弱

| ID | YOLO class | 實際框數 |
|---:|---|---:|
| 23 | `dynamicCrescendo` | 28 |
| 24 | `dynamicCrescendoHairpin` | 48 |
| 25 | `dynamicCrescendoLong` | 130 |
| 26 | `dynamicDiminuendo` | 24 |
| 27 | `dynamicDiminuendoHairpin` | 61 |
| 28 | `dynamicDiminuendoLong` | 24 |
| 29 | `dynamicF` | 407 |
| 30 | `dynamicForteestaccato` | 0 |
| 31 | `dynamicM` | 2 |
| 32 | `dynamicP` | 267 |
| 33 | `dynamicR` | 0 |
| 34 | `dynamicRinforzando` | 0 |
| 35 | `dynamicS` | 232 |
| 36 | `dynamicZ` | 1 |

### 3.4 Fermata 與指法

| ID | YOLO class | 實際框數 |
|---:|---|---:|
| 37 | `fermataAbove` | 45 |
| 38 | `fermataBelow` | 10 |
| 39 | `fingering0` | 0 |
| 40 | `fingering1` | 941 |
| 41 | `fingering2` | 953 |
| 42 | `fingering3` | 951 |
| 43 | `fingering4` | 808 |
| 44 | `fingering5` | 727 |
| 45 | `fingeringSubstitution` | 99 |

### 3.5 Flag、notehead、stem

| ID | YOLO class | 實際框數 |
|---:|---|---:|
| 46 | `flag128thDownSmall` | 0 |
| 47 | `flag128thUpSmall` | 0 |
| 48 | `flag16thDownSmall` | 0 |
| 49 | `flag16thUpSmall` | 0 |
| 50 | `flag32ndDownSmall` | 0 |
| 51 | `flag32ndUpSmall` | 6 |
| 52 | `flag64thDownSmall` | 0 |
| 53 | `flag64thUpSmall` | 0 |
| 54 | `flag8thDownSmall` | 1 |
| 55 | `flag8thUpSmall` | 0 |
| 61 | `noteheadBlackInSpaceSmall` | 64 |
| 62 | `noteheadBlackOnLineSmall` | 65 |
| 63 | `noteheadDoubleWholeInSpaceSmall` | 0 |
| 64 | `noteheadDoubleWholeOnLineSmall` | 0 |
| 65 | `noteheadHalfInSpaceSmall` | 0 |
| 66 | `noteheadHalfOnLineSmall` | 0 |
| 67 | `noteheadWholeInSpaceSmall` | 0 |
| 68 | `noteheadWholeOnLineSmall` | 0 |
| 88 | `stemSmall` | 129 |
| 89 | `stemSmallAcciaccatura` | 0 |

### 3.6 鍵盤／踏板文字

| ID | YOLO class | 實際框數 |
|---:|---|---:|
| 56 | `keyboardMitEinerSaite` | 1 |
| 57 | `keyboardPed` | 2 |
| 58 | `keyboardPedalPed` | 58 |
| 59 | `keyboardPedalUp` | 53 |
| 60 | `keyboardSulUnaCorda` | 2 |

### 3.7 數字

| ID | YOLO class | 實際框數 |
|---:|---|---:|
| 69 | `numeral0` | 0 |
| 70 | `numeral1` | 5 |
| 71 | `numeral2` | 0 |
| 72 | `numeral3` | 0 |
| 73 | `numeral4` | 0 |
| 74 | `numeral5` | 0 |
| 75 | `numeral6` | 0 |
| 76 | `numeral7` | 0 |
| 77 | `numeral8` | 0 |
| 78 | `numeral9` | 0 |

### 3.8 裝飾音、ottava、slur

| ID | YOLO class | 實際框數 |
|---:|---|---:|
| 79 | `ornamentMordent` | 0 |
| 80 | `ornamentShortTrill` | 0 |
| 81 | `ornamentTrill` | 43 |
| 82 | `ornamentTurn` | 2 |
| 83 | `ornamentTurnInverted` | 0 |
| 84 | `ornamentWiggleTrill` | 36 |
| 85 | `ottavaBracket` | 54 |
| 86 | `segno` | 0 |
| 87 | `slur` | 1,458 |

### 3.9 Tempo

| ID | YOLO class | 實際框數 |
|---:|---|---:|
| 90 | `tempoATempo` | 23 |
| 91 | `tempoAdagio` | 5 |
| 92 | `tempoAllegretto` | 1 |
| 93 | `tempoAllegro` | 6 |
| 94 | `tempoAndante` | 0 |
| 95 | `tempoCalando` | 0 |
| 96 | `tempoGrave` | 0 |
| 97 | `tempoInTempo` | 4 |
| 98 | `tempoLargo` | 0 |
| 99 | `tempoModerato` | 1 |
| 100 | `tempoMunuetto` | 0 |
| 101 | `tempoPrestissimo` | 1 |
| 102 | `tempoPresto` | 1 |
| 103 | `tempoRallentando` | 0 |
| 104 | `tempoRitardando` | 11 |
| 105 | `tempoRitardandoLong` | 16 |
| 106 | `tempoTempo` | 5 |
| 107 | `tempoVivace` | 2 |

### 3.10 文字術語

| ID | YOLO class | 實際框數 |
|---:|---|---:|
| 108 | `termAllaMarcia` | 1 |
| 109 | `termAmabilita` | 1 |
| 110 | `termBen` | 1 |
| 111 | `termCantabile` | 4 |
| 112 | `termCon` | 1 |
| 113 | `termConAffetto` | 1 |
| 114 | `termConBrioEdAppasionato` | 1 |
| 115 | `termDolce` | 10 |
| 116 | `termE` | 3 |
| 117 | `termEd` | 1 |
| 118 | `termEspressivo` | 16 |
| 119 | `termEtwasLebhaftUndDerInnigstenEmpfindung` | 1 |
| 120 | `termLebhaft` | 1 |
| 121 | `termLegato` | 15 |
| 122 | `termLeggiermente` | 2 |
| 123 | `termMaNonTroppo` | 3 |
| 124 | `termMaestoso` | 1 |
| 125 | `termMarcato` | 1 |
| 126 | `termMarschmäBig` | 1 |
| 127 | `termMeno` | 3 |
| 128 | `termMezzo` | 1 |
| 129 | `termMitLebhaftigkeitUndDurchausMitEmpfindungUndAusdrunk` | 1 |
| 130 | `termMolto` | 5 |
| 131 | `termNachUndNachMehrereSaiten` | 1 |
| 132 | `termNon` | 2 |
| 133 | `termPiù` | 4 |
| 134 | `termPoco` | 13 |
| 135 | `termPocoAPoco` | 3 |
| 136 | `termPoiAPoiSemprePiùAllegro` | 2 |
| 137 | `termRinforz` | 2 |
| 138 | `termRitenente` | 4 |
| 139 | `termSanft` | 1 |
| 140 | `termSemplice` | 1 |
| 141 | `termSempre` | 19 |
| 142 | `termTutteLeCorde` | 2 |
| 143 | `termUn` | 1 |
| 144 | `termZurückhaltend` | 1 |

### 3.11 Tie、tremolo、tuplet

| ID | YOLO class | 實際框數 |
|---:|---|---:|
| 145 | `tie` | 987 |
| 146 | `tremolo1` | 0 |
| 147 | `tremolo2` | 0 |
| 148 | `tremolo3` | 0 |
| 149 | `tremolo4` | 0 |
| 150 | `tremolo5` | 0 |
| 151 | `tuplet1` | 0 |
| 152 | `tuplet12` | 1 |
| 153 | `tuplet2` | 0 |
| 154 | `tuplet3` | 36 |
| 155 | `tuplet4` | 0 |
| 156 | `tuplet5` | 25 |
| 157 | `tuplet6` | 20 |
| 158 | `tuplet7` | 0 |
| 159 | `tuplet8` | 0 |
| 160 | `tuplet9` | 1 |
| 161 | `tupletBracket` | 0 |

## 4. MusicXML 實際包含的事件／元素類型

以下數量是六份 repetition-preserving MusicXML 的 element 次數。start/stop
型 element 是端點數，不等於掃描圖上的曲線或括號框數。

### 4.1 音符與時間結構

| XML 類型 | 實際數量 | 內容 |
|---|---:|---|
| `<measure>` | 1,136 | 小節容器與小節編號 |
| `<note>` | 20,854 | 音符或休止符事件 |
| `<pitch>` | 19,081 | step、alter、octave |
| `<rest>` | 1,773 | 休止符 |
| `<chord>` | 5,394 | 與前一音共用 onset 的和弦成員 |
| `<grace>` | 67 | 裝飾／倚音事件 |
| `<duration>` | 22,729 | MusicXML divisions 時間長度 |
| `<voice>` | 16,416 | 聲部 |
| `<staff>` | 21,972 | staff 編號 |
| `<backup>` | 1,857 | 多聲部時間游標回退 |
| `<forward>` | 85 | 時間游標前進 |

音符種類 `<type>`：eighth 7,652、quarter 5,670、16th 4,918、32nd
1,221、half 1,138、whole 166、64th 89。另有 `<stem>` 13,612、`<beam>`
15,860、`<dot>` 1,409。

### 4.2 臨時記號與樂譜上下文

| XML 類型 | 實際內容 |
|---|---|
| `<accidental>` | natural 1,081；flat 928；sharp 697；double-sharp 33；flat-flat 15 |
| `<clef>` | 189 個 clef 事件，含 sign 與 line |
| `<key>` | 23 個 key signature 事件，含 fifths 與 mode |
| `<time>` | 15 個拍號事件，含 beats 與 beat-type |
| `<divisions>` | 每份樂譜的時間單位設定 |
| `<print>` | 243 個排版事件，含 system/page layout 資訊 |

### 4.3 Direction、力度與文字

`<direction-type>` 的實際子類型：

| XML 類型 | 數量 | 值／用途 |
|---|---:|---|
| `<dynamics>` | 485 | `sf` 192、`p` 135、`f` 75、`ff` 33、`pp` 29、`sfp` 13、`fp` 4、`mf` 2、`ppp` 1、`mp` 1 |
| `<words>` | 210 | 49 種實際字串，例如 `cresc.`、`dimin.`、`a tempo`、`ritard.`、`sempre`、`espressivo` |
| `<wedge>` | 206 | stop 103、diminuendo 69、crescendo 34 |
| `<pedal>` | 141 | start/stop，以及 line/sign 屬性 |
| `<octave-shift>` | 66 | down 33、stop 33，size=8 |
| `<metronome>` | 12 | beat-unit 與 per-minute |

### 4.4 Notation 與跨事件關係

| XML 類型 | 實際數量 | 說明 |
|---|---:|---|
| `<staccato>` | 573 | articulation；MusicXML 沒有拆成 above/below class |
| `<strong-accent>` | 1 | 接近 marcato 語意 |
| `<fermata>` | 56 | 全部為 normal；placement 資訊不完整 |
| `<slur>` | 1,309 endpoints | 656 start、653 stop；含 over/under orientation |
| `<tie>`＋`<tied>` | 各 1,494 endpoints | 同一 tie 的聲音與 notation 表示；各有 747 start、747 stop |
| `<tuplet>` | 302 endpoints | start/stop、above/below |
| `<time-modification>` | 618 notes | actual-notes、normal-notes、normal-type |
| `<trill-mark>` | 18 | trill ornament |
| `<wavy-line>` | 36 | trill 延伸線的 start/stop |
| `<turn>` | 1 | turn ornament |

本批 XML 沒有 `<technical>` 子項，因此沒有可直接使用的 `<fingering>`；也沒有
`<arpeggiate>`、`<tremolo>`、`<mordent>`、`<segno>`、`<coda>` 或
`<caesura>` element。

### 4.5 Repeat、結尾與文件資訊

| XML 類型 | 數量 | 說明 |
|---|---:|---|
| `<barline>` | 34 | 小節線語意 |
| `<repeat>` | 3 | forward/backward repeat |
| `<ending>` | 7 | volta ending |
| `<work-title>` | 6 | 每份樂譜的作品標題 |
| `<identification>` | 6 | encoder／版本來源資訊 |
| `<defaults>` | 6 | scaling、page layout、font 等排版預設 |

這些 XML 資訊大多沒有對應的 YOLO class，但仍應出現在未來的
`xml_nodes.csv` 或 `xml_events.csv`，不能因為沒有 bbox 就被刪除。

## 5. YOLO 與 MusicXML 的 class 對應差異

| YOLO 類別群 | MusicXML 表示 | 對應狀態 | 主要差異 |
|---|---|---|---|
| accidental | `<accidental>`、pitch alter | 間接可對應 | YOLO 只標 small 圖形；XML 表示音高語意，不保證有掃描座標 |
| staccato | `<articulations><staccato/>` | 可對應 | YOLO 拆 above/below；XML 本批未明確拆 placement |
| accent／marcato | `<accent>`／`<strong-accent>` | 部分可對應 | 本批 XML 只有 1 個 strong-accent，與 16 個 YOLO accent 框不等量 |
| dynamicF/P/S/M/Z | `<dynamics>` 的 `f`, `p`, `sf`, `mf` 等 | 多對一／一對多 | YOLO 常把複合力度拆成字母；XML 把 `sf`、`ff` 等存成完整值 |
| crescendo/diminuendo | `<words>` 或 `<wedge>` | 可對應但需分型 | YOLO 分文字、hairpin、long；XML 分 words 與 wedge start/stop |
| fermataAbove/Below | `<fermata>` | 可對應 | XML 有語意但 placement 不一定足以區分掃描方向 |
| fingering0–5 | 理論上 `<technical><fingering>` | 本批無直接對應 | XML 完全沒有 fingering element，必須依影像幾何連到 note，並人工確認 |
| flag/beam/notehead/stem | `<type>`、`<beam>`、`<stem>`、pitch | 間接推導 | XML 描述整個 note；YOLO 標的是 note 的視覺零件，沒有一對一 event |
| pedal/keyboard | `<pedal>` 或 `<words>` | 部分可對應 | XML pedal 是 start/stop span；YOLO 可能是 Ped glyph、release glyph 或文字 |
| numeral0–9 | tuplet、fingering、ending 或文字上下文 | 不可只看 class | 同一數字圖形可能代表完全不同的音樂語意 |
| trill/turn/wiggle | `<trill-mark>`、`<turn>`、`<wavy-line>` | 可對應 | trill 主符號與延伸線在 XML 是不同 element |
| ottavaBracket | `<octave-shift>` | 可對應 | XML 是 start/stop；YOLO 通常是一個可跨距離的視覺框 |
| slur | `<slur type=start/stop>` | 可對應但為 span | 一個 YOLO 曲線框需要連到兩個 XML note endpoints |
| tie | `<tie>`／`<tied>` start/stop | 可對應但為 span | XML 同時保存 playback tie 與 notation tie，必須去除雙重計數 |
| tempo/term | `<words>`、`<metronome>` | 文字正規化後可對應 | YOLO 常以單字 class 標註；XML 可能把整個片語放在一個 words element |
| tuplet | `<tuplet>`＋`<time-modification>` | 可對應但為群組 | YOLO 是數字／括號圖形；XML 是一組 notes 與 start/stop 關係 |
| special text | `<words>`、repeat/ending 或可能缺漏 | 本批多數無直接對應 | 三個 YOLO 特殊文字在六份 XML 的文字內容中找不到完全對應字串 |
| tremolo | 理論上 `<ornaments><tremolo>` | 本批無直接對應 | YOLO 定義存在但實際框數為 0，XML 也未出現 tremolo element |

## 6. 數量為什麼不會相等

即使 YOLO 與 XML 都表示同一種記號，數量也不應直接期待相等：

1. YOLO 是「一個視覺框」；XML 是「一個語意事件或端點」。
2. 一個 `sf` XML dynamic 可能對應 YOLO 的 `dynamicS` 與 `dynamicF` 兩個框。
3. 一個 slur YOLO 框對應 XML 的 start 和 stop 兩個 endpoint。
4. 一個 tuplet YOLO 數字可能對應多個具有 `time-modification` 的 notes。
5. XML 可能包含沒有被 YOLO 標註的事件，形成 `xml_only`。
6. YOLO 可能包含 XML 缺少的印刷內容，形成 `yolo_only`。
7. repeat-preserving XML 與 unfolded BPSD timeline 可能讓同一個印刷框對應多次演奏事件。

## 7. 後續 CSV 應如何保留差異

不能只建立「一列一個 YOLO box」的表，否則 XML-only 資料必然消失。建議至少
保留以下四張表：

- `yolo_boxes.csv`：每個 bbox 一列，保留全部 9,828 個 YOLO 框。
- `xml_nodes.csv`：每個 XML node 一列，完整保存 tag、attributes、text、path。
- `xml_events.csv`：將 note、rest、direction、notation、context、span 等轉成事件。
- `alignment_links.csv`：保存 bbox 與 XML event 的一對一、一對多或多對一關係。

最後再產生 `combined_master.csv`，明確標記：

- `aligned`：YOLO 與 XML 已對齊。
- `yolo_only`：YOLO 有框，但 XML 沒有對應資訊。
- `xml_only`：XML 有事件，但 YOLO 沒有對應框。
- `candidate`／`ambiguous`／`unresolved`：尚需人工確認。

如此才能同時保存 YOLO 的圖像 class 與 MusicXML 的完整音樂語意，而不把
「XML 沒有 bbox」或「YOLO 沒有時間資訊」誤當成資料錯誤。
