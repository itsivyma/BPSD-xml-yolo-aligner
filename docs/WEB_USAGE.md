# Website workflow

The website runs a raw, single-page alignment and returns CSV and full-page
review images. It does not require the user to prepare the intermediate master
CSV manually.

## Start the website

From an activated virtual environment in the repository:

```bash
python -m pip install .
bpsd-aligner web
```

Open the Streamlit address printed in the terminal, normally
`http://localhost:8501`.

## Prepare one page

The image and YOLO TXT must describe the same page. Upload these five required
files in **Run alignment**:

| Upload | Purpose |
| --- | --- |
| Score image | Detect systems/staves and render full-page review overlays |
| YOLO TXT | Preserve every class ID and normalized bounding box |
| Written/repetition MusicXML | Supply measures, notes, pitch, staff, voice, dynamics, ties, slurs, and other score structure |
| `ann_score_note.csv` | Supply the official BPSD note IDs and musical timeline |
| `notes.json` | Translate every YOLO class ID into its class name |

Upload unfolded MusicXML as well when the written score contains repeats. If it
is omitted, the system uses an identity repeat mapping and adds a warning to the
validation report.

Enter a stable **Score ID**, and set **MusicXML page** to the page shown by the
uploaded image. Page numbering starts at 1.

## Run and review

Click **Align all information**. The page reports six checkpoint-like stages:

1. validate uploads;
2. map written and unfolded measures;
3. align YOLO boxes;
4. export every MusicXML event and source node;
5. build the lossless combined CSV;
6. validate and package outputs.

After completion, choose an overlay under **Review images**. The all-symbol
overlay is the quickest way to verify that every YOLO rectangle is present.
Dynamics and fingering overlays make those target classes easier to inspect.

The same input fingerprint is reused during the current Streamlit session. For
large multi-page datasets or recovery after a server restart, use the CLI
dataset commands, whose page checkpoints are persisted on disk.

## Downloads

- **All-information CSV** is the most complete single CSV. Its
  `source_record_type` column identifies `yolo`, `xml_event`, and `xml_node`
  records.
- **Combined event CSV** is easier to analyze as BPS-OMR data. It contains every
  YOLO box and represents every extracted MusicXML event.
- **Detailed YOLO CSV** contains match source, status, confidence, target staff,
  pitch, and note-link evidence for each box.
- **Validation JSON** records counts, warnings, validation errors, and produced
  files.
- **All outputs ZIP** includes the CSV files, XML-node export, link table, JSON,
  and full-page review images.

## Accuracy boundary

The website preserves all source records, but preservation is different from a
confirmed match. Dynamics currently have direct MusicXML matching rules.
Fingering links are optional geometric candidates because the source MusicXML
does not contain fingering elements. Other classes that lack a confirmed rule
are retained with unresolved fields for later review rather than guessed.
