"""Tests for the CSV and SQLite exporters."""

import csv
import sqlite3

from flowcount.analytics import CSVExporter, Event, SQLiteExporter


def test_csv_exporter_writes_tracks_and_events(make_track, make_ctx, tmp_path):
    path = str(tmp_path / "out.csv")
    exporter = CSVExporter(path)
    ctx = make_ctx([make_track(1, [10, 10, 20, 20], velocity=(3.0, 4.0))],
                   frame_index=1, timestamp=0.0)
    exporter.write_tracks(ctx)
    exporter.write_events([Event("line_cross", 1, "person", 1, 0.0, {"line": "L"})])
    exporter.close()

    rows = list(csv.reader(open(path)))
    assert rows[0][0] == "frame_index"
    assert len(rows) == 2                 # header + 1 track row
    assert rows[1][3] == "person"

    events_path = path.rsplit(".", 1)[0] + "_events.csv"
    event_rows = list(csv.reader(open(events_path)))
    assert len(event_rows) == 2           # header + 1 event row
    assert event_rows[1][2] == "line_cross"


def test_sqlite_exporter_writes_tables(make_track, make_ctx, tmp_path):
    path = str(tmp_path / "out.db")
    exporter = SQLiteExporter(path)
    ctx = make_ctx([make_track(1, [10, 10, 20, 20])], frame_index=1, timestamp=0.0)
    exporter.write_tracks(ctx)
    exporter.write_events([Event("zone_enter", 1, "person", 1, 0.0, {"zone": "z"})])
    exporter.close()

    conn = sqlite3.connect(path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
        kind = conn.execute("SELECT kind FROM events").fetchone()[0]
        assert kind == "zone_enter"
    finally:
        conn.close()
