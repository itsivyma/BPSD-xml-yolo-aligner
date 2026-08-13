from bpsd_aligner.cli import _help, main


def test_help_lists_terminal_and_web_commands(capsys):
    main(["--help"])
    output = capsys.readouterr().out
    assert "bpsd-aligner align" not in output
    assert "align" in output
    assert "web" in output
    assert "combine" in output


def test_version(capsys):
    main(["--version"])
    assert capsys.readouterr().out.strip() == "0.3.2"


def test_help_is_stable():
    assert _help().startswith("BPSD MusicXML–YOLO Aligner")
