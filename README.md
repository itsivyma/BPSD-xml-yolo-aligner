# BPSD XML–YOLO Aligner

把掃描樂譜上的 YOLO bounding boxes 對齊 MusicXML 與 BPSD note
annotations，輸出符合 BPS-OMR 欄位的 CSV，並提供圖片式人工複核。

```bash
git clone https://github.com/itsivyma/BPSD-xml-yolo-aligner.git
cd BPSD-xml-yolo-aligner
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
bpsd-aligner web
```

Windows PowerShell 使用 `.venv\Scripts\Activate.ps1`。瀏覽器通常會開啟
`http://localhost:8501`。

### macOS／Python 3.14 安裝修復

一般使用者請採用上面的標準安裝，不需要 editable mode。如果終端已存在
`bpsd-aligner`，執行時卻出現 `ModuleNotFoundError: bpsd_aligner`，請在專案目錄
重新安裝並驗證：

```bash
python -m pip uninstall -y bpsd-xml-yolo-aligner
python -m pip install .
python -c "import bpsd_aligner; print(bpsd_aligner.__version__)"
bpsd-aligner --help
```

部分 macOS／Python 3.14 環境會略過 editable install 產生且被標記為隱藏的
`.pth` 檔；標準安裝不依賴該 import hook。若要開發與執行測試，重新安裝時
使用 `python -m pip install ".[dev]"`。

## 網站輸入

| 欄位 | 應上傳的版本 | 必要 |
|---|---|---|
| Score images | 掃描頁面圖片；可一次多選 | 是 |
| YOLO TXT files | 與圖片同 stem 的 TXT；可一次多選 | 是 |
| YOLO class map | 同一批 YOLO 的 `notes.json` | 是 |
| Repetition MusicXML | `score_xml_repetitions/*.xml` | 是 |
| BPSD note annotations | `ann_score_note/*.csv` | 是 |
| Unfolded MusicXML | `score_xml_unfolded/*.xml`，只作驗證 | 否 |
| Clean repetition PDF | `score_pdf_repetitions/*.pdf` | 建議 |

不要上傳 `.sib` 或 `score_pdf_unfolded`。圖片與 TXT 必須一一同名配對。

網站主流程只有四步：

1. 上傳整首曲目的頁面與共用資料。
2. 按 **Align all uploaded pages**；工作在背景逐頁 checkpoint。
3. 在 Review workspace 檢查 overview、單一 class、跨頁端點，並直接點音頭更正。
4. 下載頁面頂端的 `bps_omr_final.csv`。

## 最終 CSV

每個 YOLO box 一列，欄位的單一來源是
[`bpsd_aligner/schema.py`](bpsd_aligner/schema.py)：

```text
class_id,x,y,w,h,class,musical_time,start_meas,end_meas,
start_note,end_note,connected_note,stem_dir,human_corrected,
is_repeated_measure
```

- 不確定、沒有 XML/BPSD 證據或不存在的值保持空白。
- `human_corrected=1` 表示人工更正，不是單純機器確認。
- `is_repeated_measure=1` 表示該 written measure 在演奏順序中重複出現。
- slur/tie 端點若是和弦，`connected_note` 保留完整端點和弦 note IDs。
- 最終列依實際音樂時間排序；未確認時間只作隱藏排序依據，不會硬寫入 CSV。

## 時間與反覆

- Repetition MusicXML 是印刷版小節、staff、voice、符號語意與版面來源。
- 系統直接解析 repeat barline 與第一／第二結尾，建立 performance occurrence。
- Unfolded MusicXML 只驗證結構推導，不取代 repetition XML。
- BPSD note CSV 提供官方 note ID 與 `start_meas` / `end_meas` 時間座標。
- D.C.、D.S.、Coda、Fine 等尚未實作的導覽會強制進入人工 review，不會猜測。

Confidence 是 heuristic score，不是真實機率。調低 threshold 前，應先累積人工
ground truth 並用 `calibrate-thresholds` 評估。

## 終端使用

```bash
bpsd-aligner --help
bpsd-aligner align --help
bpsd-aligner batch-align --help
bpsd-aligner regression-smoke --help
bpsd-aligner job-admin status
```

網站預設只產生 final CSV、review 必要資料、驗證 JSON 與可簽章 resume
checkpoint。研究用 XML node dump、完整 combine/timeline 請明確使用
`xml-export` 或 `combine` CLI，不會拖慢一般網站工作。

## 修改後驗證

```bash
python -m pip install ".[dev]"
python -m pytest -q

bpsd-aligner regression-smoke \
  --manifest regression/finished_xia_pages.json \
  --dataset-root "/path/to/中研院" \
  --output-dir output/regression-finished-xia \
  --baseline regression/finished_xia_baseline.json \
  --resume
```

`finished/Xia` 是正式 162-class profile；`Xia` 113-class profile 是保留的測試資料。
只有在人眼確認語意變更正確後才能使用 `--update-baseline`。

## 部署

```bash
cp .env.example .env
docker build -t bpsd-xml-yolo-aligner .
docker run --env-file .env -p 8501:8501 \
  -v bpsd-jobs:/var/lib/bpsd-aligner \
  bpsd-xml-yolo-aligner
```

Production mode 必須設定 `BPSD_ALIGNER_USERS_FILE` 或
`BPSD_ALIGNER_ACCESS_TOKEN`。Checkpoint ZIP 以部署密鑰 HMAC 簽章並綁定
owner、輸入 fingerprint、pipeline version 與 code signature；多台主機交換
checkpoint 時必須共用 `BPSD_ALIGNER_CHECKPOINT_SECRET`。

預設限制與清理值見 [`.env.example`](.env.example)。正式服務仍應置於 HTTPS
reverse proxy／SSO 後方，並監控 `bpsd-aligner job-admin status` 的失敗工作與容量。

## 維護入口

1. [START_HERE.md](START_HERE.md) — 新使用者與維護者快速上手。
2. [Architecture](docs/ARCHITECTURE.md) — 資料流與模組責任。
3. [Development](docs/DEVELOPMENT.md) — 新 class、matcher 與回歸流程。
4. [Handover](docs/HANDOVER.md) — 已知限制與下一步研究工作。
5. [Web usage](docs/WEB_USAGE.md) — 上傳、review、部署細節。

主要程式責任：

- `bpsd_aligner/schema.py`：正式 CSV 欄位唯一來源。
- `bpsd_aligner/web_pipeline.py`：多頁 orchestration 與 compact outputs。
- `bps_xml_alignment.py`：相容入口與 matcher；新邏輯優先拆到 package module。
- `repeat_mapping.py`：反覆 occurrence 與安全狀態。
- `bpsd_aligner/job_store.py`：持久化、簽章 checkpoint、quota 與 retention。
- `bpsd_aligner/review_corrections.py`：人工答案驗證與 final CSV 套用。

Beethoven 原始資料、產生的 CSV／圖片與本機 job 目錄不在 repository 中。

## License

[MIT](LICENSE)
