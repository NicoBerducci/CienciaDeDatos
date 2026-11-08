"""Checkpoints de normalizacion completos; nunca contienen historiales parciales.

El Bronze es append-only. Los controles/codigo se identifican por SHA-256 y los
archivos de entrada por ruta, tamano y mtime_ns. No editar payloads in-place ni
restaurar artificialmente sus timestamps: eso viola el contrato del Bronze.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import logging
import os
import zlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4


CHECKPOINT_FORMAT = 1
ENDPOINT_KEYS = ("detail", "games", "players_stats", "short_players_stats", "game_steam_profiles")
LOGGER = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_fingerprint(paths: Iterable[Path], schema: list[str], batch_size: int) -> str:
    payload = {
        "format": CHECKPOINT_FORMAT, "batch_size": batch_size, "schema": schema,
        "inputs": [(str(path), sha256_file(path)) for path in paths],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def source_signature(items: list[tuple[str, dict[str, Any]]]) -> list[list[Any]]:
    """Detecta un payload reemplazado/desaparecido sin volver a descomprimirlo."""
    result = []
    for key, indexed in items:
        for endpoint in ENDPOINT_KEYS:
            path_value = indexed.get(endpoint)
            if not path_value:
                continue
            path = Path(path_value)
            # Una ruta registrada debe existir. No cachear un error de montaje
            # como si fuera una estadistica realmente ausente en la fuente.
            stat = path.stat()
            result.append([key, endpoint, str(path), stat.st_size, stat.st_mtime_ns])
    return result


@contextmanager
def atomic_text_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        # Solo nuestro temporal exacto; un checkpoint publicado nunca se borra.
        temporary.unlink(missing_ok=True)


def write_csv_rows_atomic(path: Path, rows: Iterable[dict[str, Any]], columns: list[str]) -> int:
    count = 0
    with atomic_text_writer(path) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


class CheckpointStore:
    def __init__(self, root: Path, fingerprint: str, fact_columns: list[str]):
        self.root = root / fingerprint
        self.fingerprint = fingerprint
        self.fact_columns = fact_columns

    def path(self, number: int) -> Path:
        return self.root / f"block={number:05d}.jsonl.gz"

    def _header(self, ids: list[int], sources: list[list[Any]]) -> dict[str, Any]:
        return {"format": CHECKPOINT_FORMAT, "fingerprint": self.fingerprint,
                "ids": ids, "sources": sources}

    def load(self, number: int, ids: list[int], sources: list[list[Any]] | None = None):
        """Solo devuelve registros luego de verificar TODO el bloque y su cierre."""
        path = self.path(number)
        if not path.exists():
            return None
        try:
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                header = json.loads(stream.readline())
                if (header.get("format") != CHECKPOINT_FORMAT
                        or header.get("fingerprint") != self.fingerprint
                        or header.get("ids") != ids
                        or (sources is not None and header.get("sources") != sources)):
                    raise ValueError("version, entradas o IDs diferentes")
                records, digest = [], hashlib.sha256()
                for match_id in ids:
                    line = stream.readline()
                    digest.update(line.encode("utf-8"))
                    item = json.loads(line)
                    row, quality = item["fact"], item["quality"]
                    if quality["match_id"] != match_id:
                        raise ValueError("IDs de calidad incorrectos o duplicados")
                    if row is not None:
                        if row["match_id"] != match_id or set(row) != set(self.fact_columns):
                            raise ValueError("fila o esquema incompletos")
                        # Compartir claves del esquema; no conservar 152 strings
                        # nuevos por cada fila decodificada desde JSON.
                        row = {column: row[column] for column in self.fact_columns}
                    records.append((row, quality))
                footer = json.loads(stream.readline())
                if footer != {"count": len(ids), "sha256": digest.hexdigest()} or stream.read():
                    raise ValueError("checksum, cantidad o cierre incorrectos")
            return records
        except (ValueError, KeyError, TypeError, EOFError, gzip.BadGzipFile, zlib.error) as exc:
            LOGGER.warning("Checkpoint %s descartado: %s", number, exc)
            return None

    def save(self, number: int, ids: list[int], sources: list[list[Any]], records: list):
        if len(ids) != len(set(ids)) or len(records) != len(ids):
            raise ValueError("No se puede confirmar un bloque incompleto o duplicado")
        for match_id, (row, quality) in zip(ids, records):
            if quality["match_id"] != match_id or (row is not None and (
                row["match_id"] != match_id or set(row) != set(self.fact_columns)
            )):
                raise ValueError("Registros del checkpoint fuera de contrato")
        path = self.path(number)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as raw:
                with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=1, mtime=0) as stream:
                    stream.write((json.dumps(self._header(ids, sources), separators=(",", ":")) + "\n").encode())
                    digest = hashlib.sha256()
                    for row, quality in records:
                        line = (json.dumps({"fact": row, "quality": quality}, separators=(",", ":")) + "\n").encode()
                        digest.update(line)
                        stream.write(line)
                    stream.write((json.dumps({"count": len(ids), "sha256": digest.hexdigest()}) + "\n").encode())
                raw.flush()
                os.fsync(raw.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def log_memory(logger: logging.Logger, phase: str) -> None:
    """Telemetria Linux de solo lectura; no hace fallar la tarea en Windows."""
    metrics = {}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            name, _, value = line.partition(":")
            if name in {"VmRSS", "VmHWM"}:
                metrics[name + "_MiB"] = round(int(value.split()[0]) / 1024, 1)
        for line in Path("/proc/meminfo").read_text().splitlines():
            name, _, value = line.partition(":")
            if name in {"MemAvailable", "SwapFree"}:
                metrics[name + "_MiB"] = round(int(value.split()[0]) / 1024, 1)
        for field in ("memory.current", "memory.peak"):
            path = Path("/sys/fs/cgroup") / field
            if path.exists():
                metrics[field + "_MiB"] = round(int(path.read_text()) / 1024**2, 1)
    except (OSError, ValueError):
        pass
    logger.info("Memoria [%s]: %s", phase, json.dumps(metrics) if metrics else "no disponible")
