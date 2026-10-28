import csv

from scripts.dataset_balance import main


def _write_labels(path, rows):
    fieldnames = ["image_id", "season", "undertone", "face_shape", "lighting", "makeup", "notes"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({**{k: "" for k in fieldnames}, **r})


def test_empty_labels_reports_nothing_to_report(tmp_path, monkeypatch, capsys):
    path = tmp_path / "labels.csv"
    _write_labels(path, [])
    monkeypatch.setattr("sys.argv", ["dataset_balance.py", "--labels", str(path)])
    main()
    out = capsys.readouterr().out
    assert "nothing to report yet" in out


def test_warns_when_below_minimum_total(tmp_path, monkeypatch, capsys):
    path = tmp_path / "labels.csv"
    _write_labels(path, [{"image_id": f"AM-EVAL-{i:04d}", "season": "Spring"} for i in range(5)])
    monkeypatch.setattr("sys.argv", ["dataset_balance.py", "--labels", str(path)])
    main()
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "minimum collection floor" in out


def test_warns_on_missing_season_entirely(tmp_path, monkeypatch, capsys):
    rows = [{"image_id": f"AM-EVAL-{i:04d}", "season": "Spring"} for i in range(70)]
    path = tmp_path / "labels.csv"
    _write_labels(path, rows)
    monkeypatch.setattr("sys.argv", ["dataset_balance.py", "--labels", str(path)])
    main()
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "no samples at all" in out
    assert "['Summer', 'Autumn', 'Winter']" in out


def test_no_warning_when_dataset_is_well_balanced(tmp_path, monkeypatch, capsys):
    rows = []
    for season in ("Spring", "Summer", "Autumn", "Winter"):
        rows += [{"image_id": f"AM-EVAL-{season}-{i:02d}", "season": season} for i in range(16)]
    path = tmp_path / "labels.csv"
    _write_labels(path, rows)
    monkeypatch.setattr("sys.argv", ["dataset_balance.py", "--labels", str(path)])
    main()
    out = capsys.readouterr().out
    assert "No collection-balance warnings" in out


def test_reports_disagreement_from_notes(tmp_path, monkeypatch, capsys):
    rows = [{"image_id": "AM-EVAL-0001", "notes": "season disagreement: R1=Spring, R2=Autumn"}]
    path = tmp_path / "labels.csv"
    _write_labels(path, rows)
    monkeypatch.setattr("sys.argv", ["dataset_balance.py", "--labels", str(path)])
    main()
    out = capsys.readouterr().out
    assert "Disagreements" in out
    assert "R1=Spring" in out
