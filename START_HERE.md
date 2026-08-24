# BPSD XML–YOLO Aligner：15 分鐘上手

這份文件是新使用者與新維護者的第一個入口。完整研究背景與所有進階
選項請再閱讀 [README](README.md) 與 [網站操作](docs/WEB_USAGE.md)。

## 1. 這套系統做什麼

輸入掃描樂譜圖片、YOLO TXT、`notes.json`、repetition MusicXML、BPSD
note annotation CSV，以及建議提供的 unfolded MusicXML；系統會把每個
YOLO 符號對到 MusicXML/BPSD 的音樂時間，產生 BPS-OMR CSV 與人工檢查圖。

這是一套 human-in-the-loop alignment system。YOLO 是辨識模型；目前的
alignment 核心是 MusicXML 規則、譜面幾何與保守的 heuristic confidence，
不是另一個已訓練的神經網路模型。

## 2. 安裝與確認

```bash
git clone https://github.com/itsivyma/BPSD-xml-yolo-aligner.git
cd BPSD-xml-yolo-aligner
python3 -m venv .venv
source .venv/bin/activate
python -m pip install ".[dev]"
bpsd-aligner --version
python -m pytest -q
```

Windows PowerShell 請用 `.venv\Scripts\Activate.ps1` 啟用環境。

## 3. 第一次使用網站

```bash
bpsd-aligner web
```

瀏覽器開啟終端顯示的網址，通常是 `http://localhost:8501`。上傳：

| 網站欄位 | 正確資料版本 |
|---|---|
| Score images | 掃描頁面圖片，可多選 |
| YOLO TXT files | 與圖片同名的 YOLO TXT，可多選 |
| Repetition MusicXML | `score_xml_repetitions/*.xml`，必要 |
| Unfolded MusicXML | `score_xml_unfolded/*.xml`，建議 |
| BPSD note annotations | `ann_score_note/*.csv`，必要 |
| YOLO class map | `notes.json`，必要 |
| Clean repetition PDF | `score_pdf_repetitions/*.pdf`，建議 |

不要上傳 `.sib`，也不要用 unfolded PDF 取代 repetition PDF。

按 **Align all uploaded pages** 後可離開頁面；工作具有進度、逐頁
checkpoint 與 resume。完成後先檢查 Review workspace，再下載最上方的
strict `bps_omr_final.csv`。不確定或沒有 XML 證據的值必須留空。

## 4. 維護者最常用的指令

```bash
# 網站
bpsd-aligner web

# 查看單頁或批次 alignment 的參數
bpsd-aligner align --help
bpsd-aligner batch-align --help

# 修改程式後先跑快速測試
python -m pytest -q

# 再跑固定十頁真實回歸；已完成頁會 resume
bpsd-aligner regression-smoke \
  --manifest regression/representative_pages.json \
  --dataset-root "/path/to/中研院" \
  --output-dir output/regression-smoke \
  --baseline regression/representative_baseline.json \
  --resume
```

上面是既有 `Xia/` 113-class 測試回歸。正式 `finished/Xia/` 162-class
資料請改用 `regression/finished_xia_pages.json`，不要覆蓋前者；若已有人工
答案，再加 `--ground-truth /path/to/evaluation_ground_truth.csv`。

只有在人工確認結果變更是正確的情況下，才能加
`--update-baseline`。不要為了讓測試變綠而直接更新 baseline。

## 5. 接下來讀什麼

- [ARCHITECTURE.md](docs/ARCHITECTURE.md)：資料流與模組責任。
- [DEVELOPMENT.md](docs/DEVELOPMENT.md)：新增 class、修改 matcher 與測試。
- [HANDOVER.md](docs/HANDOVER.md)：已完成、限制與下一階段建議。
- [XML/YOLO class 比較](docs/XML_YOLO_CLASS_COMPARISON.md)：兩種來源的資訊差異。
- [BPS-OMR CSV schema](docs/CSV_SCHEMA_PROPOSAL.md)：正式與診斷輸出欄位。

## 6. 三條重要原則

1. Repetition XML 負責印刷版小節與頁面幾何；unfolded XML/BPSD 負責
   展開反覆後的演奏時間，不能互換。
2. Confidence 是 heuristic score，不是真實機率；不要只靠降低 threshold
   來減少 Review。
3. 重構時不要重跑整個資料集。先跑單元測試，再跑固定十頁回歸，並使用
   checkpoint/resume。
