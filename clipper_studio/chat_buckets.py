"""Disk-backed replay buckets with bounded ingestion and lookup batches."""

import sqlite3
import time
from collections import Counter
from contextlib import closing, contextmanager
from dataclasses import dataclass

BIN_SECONDS = 5
MAX_BUCKET_ENTRIES = 8192
MAX_BUCKET_BYTES = 2 * 1024 * 1024
BUCKET_BUDGET_WARNING = "Chat replay exceeded the per-interval analysis budget."


@dataclass(frozen=True)
class Message:
    offset_seconds: float
    author: str
    text: str


class ChatBuckets:
    def __init__(self, path, duration):
        self.path = path
        self.duration = duration
        self.counts = Counter()
        self.warning = None

    def __len__(self):
        return sum(self.counts.values())

    @contextmanager
    def writer(self):
        with closing(sqlite3.connect(self.path)) as connection:
            connection.executescript("""
                PRAGMA cache_size = -4096;
                CREATE TABLE replay_ids (id TEXT PRIMARY KEY) WITHOUT ROWID;
                CREATE TABLE messages (
                    bucket INTEGER, offset REAL, author TEXT, text TEXT,
                    UNIQUE (bucket, offset, author, text)
                );
            """)
            yield BucketWriter(self, connection)

    @contextmanager
    def reader(self):
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("PRAGMA cache_size = -4096")

            def group(index):
                if not self.counts[index]:
                    return ()
                records = connection.execute(
                    "SELECT offset, author, text FROM messages WHERE bucket = ? ORDER BY offset, rowid",
                    (index,),
                )
                return tuple(Message(*record) for record in records)

            yield group, self.counts.__getitem__


class BucketWriter:
    def __init__(self, buckets, connection):
        self.buckets = buckets
        self.connection = connection
        self.batch = []
        self.batch_bytes = 0
        self.size = 512
        self.entries = Counter()
        self.bytes = Counter()
        self.limited = False

    def add(self, message, message_id):
        if 0 <= message.offset_seconds < self.buckets.duration:
            index = int(message.offset_seconds // BIN_SECONDS)
            size = len(message.author.encode()) + len(message.text.encode())
            identity_bytes = len(message_id.encode()) if isinstance(message_id, str) else 0
            self.batch.append((index, message, message_id if identity_bytes else None, size))
            self.batch_bytes += size + identity_bytes
            if len(self.batch) >= self.size or self.batch_bytes >= MAX_BUCKET_BYTES:
                self.flush()

    def flush(self):
        if not self.batch:
            return
        start = time.monotonic()
        with self.connection:
            for index, message, message_id, size in self.batch:
                if message_id and not self.connection.execute(
                    "INSERT OR IGNORE INTO replay_ids VALUES (?)", (message_id,),
                ).rowcount:
                    continue
                self.buckets.counts[index] += 1
                record = index, message.offset_seconds, message.author, message.text
                if self.entries[index] >= MAX_BUCKET_ENTRIES or self.bytes[index] + size > MAX_BUCKET_BYTES:
                    if not self.connection.execute(
                        "SELECT 1 FROM messages WHERE bucket = ? AND offset = ? AND author = ? AND text = ?", record,
                    ).fetchone():
                        self.buckets.warning = BUCKET_BUDGET_WARNING
                        self.limited = True
                    continue
                if self.connection.execute("INSERT OR IGNORE INTO messages VALUES (?, ?, ?, ?)", record).rowcount:
                    self.entries[index] += 1
                    self.bytes[index] += size
        self.batch.clear()
        self.batch_bytes = 0
        elapsed = time.monotonic() - start
        if elapsed > .05:
            self.size = max(128, self.size // 2)
        elif elapsed < .015:
            self.size = min(2048, self.size * 2)
