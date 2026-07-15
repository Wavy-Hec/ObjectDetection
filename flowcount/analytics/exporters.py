"""Data exporters for tracks and events.

Both exporters use only the standard library (``csv`` / ``sqlite3``) so there
are no extra dependencies. They write per-frame track rows and discrete events.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

from .base import Event, FrameContext


class CSVExporter:
    """Writes one CSV row per track per frame, plus an events CSV alongside."""

    TRACK_FIELDS = [
        "frame_index",
        "timestamp",
        "track_id",
        "class_label",
        "x1",
        "y1",
        "x2",
        "y2",
        "speed",
    ]
    EVENT_FIELDS = ["frame_index", "timestamp", "kind", "track_id", "class_label", "data"]

    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._tracks_file = open(path, "w", newline="")  # noqa: SIM115 (closed in close())
        self._tracks_writer = csv.writer(self._tracks_file)
        self._tracks_writer.writerow(self.TRACK_FIELDS)

        events_path = path.rsplit(".", 1)[0] + "_events.csv"
        self._events_file = open(events_path, "w", newline="")  # noqa: SIM115 (closed in close())
        self._events_writer = csv.writer(self._events_file)
        self._events_writer.writerow(self.EVENT_FIELDS)

    def write_tracks(self, ctx: FrameContext) -> None:
        for t in ctx.tracks:
            x1, y1, x2, y2 = t.bbox
            self._tracks_writer.writerow(
                [
                    ctx.frame_index,
                    f"{ctx.timestamp:.3f}",
                    t.id,
                    t.class_label,
                    f"{x1:.1f}",
                    f"{y1:.1f}",
                    f"{x2:.1f}",
                    f"{y2:.1f}",
                    f"{t.get_speed():.2f}",
                ]
            )

    def write_events(self, events: list[Event]) -> None:
        for e in events:
            self._events_writer.writerow(
                [
                    e.frame_index,
                    f"{e.timestamp:.3f}",
                    e.kind,
                    e.track_id,
                    e.class_label,
                    json.dumps(e.data),
                ]
            )

    def close(self) -> None:
        self._tracks_file.close()
        self._events_file.close()


class SQLiteExporter:
    """Writes tracks and events into a SQLite database."""

    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS tracks ("
            "frame_index INTEGER, timestamp REAL, track_id INTEGER, class_label TEXT, "
            "x1 REAL, y1 REAL, x2 REAL, y2 REAL, speed REAL)"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS events ("
            "frame_index INTEGER, timestamp REAL, kind TEXT, track_id INTEGER, "
            "class_label TEXT, data TEXT)"
        )
        self.conn.commit()
        self._since_commit = 0

    def write_tracks(self, ctx: FrameContext) -> None:
        rows = [
            (
                ctx.frame_index,
                ctx.timestamp,
                t.id,
                t.class_label,
                t.bbox[0],
                t.bbox[1],
                t.bbox[2],
                t.bbox[3],
                t.get_speed(),
            )
            for t in ctx.tracks
        ]
        if rows:
            self.conn.executemany("INSERT INTO tracks VALUES (?,?,?,?,?,?,?,?,?)", rows)
            self._maybe_commit()

    def write_events(self, events: list[Event]) -> None:
        rows = [
            (e.frame_index, e.timestamp, e.kind, e.track_id, e.class_label, json.dumps(e.data))
            for e in events
        ]
        if rows:
            self.conn.executemany("INSERT INTO events VALUES (?,?,?,?,?,?)", rows)
            self._maybe_commit()

    def _maybe_commit(self) -> None:
        self._since_commit += 1
        if self._since_commit >= 50:
            self.conn.commit()
            self._since_commit = 0

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()
