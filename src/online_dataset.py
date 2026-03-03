from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path
from queue import Queue
from typing import Any
from typing import Iterable


PACKED_COLUMNS = {
    "sequences": "pkl",
    "returns": "pkl",
    "advantages": "pkl",
    "action_mask": "pkl",
    "attention_mask": "pkl",
    "action_log_probs": "pkl",
}


_SENTINEL = object()


def _join_uri(root: str, leaf: str) -> str:
    return f"{root.rstrip('/')}/{leaf.lstrip('/')}"


def _write_stream_partition(
    sid: int,
    local_root: str,
    remote_root: str | None,
    sample_queue: Queue,
    compression: str | None,
    size_limit: int | str,
    keep_local: bool,
    counters: list[int],
    errors: list[Exception],
) -> None:
    try:
        from streaming import MDSWriter

        stream_leaf = f"stream_{sid:03d}"
        local_subdir = str(Path(local_root) / stream_leaf)
        Path(local_subdir).mkdir(parents=True, exist_ok=True)

        out_path: str | tuple[str, str] = local_subdir
        if remote_root:
            out_path = (local_subdir, _join_uri(remote_root, stream_leaf))

        with MDSWriter(
            out=out_path,
            columns=PACKED_COLUMNS,
            keep_local=keep_local,
            compression=compression,
            size_limit=size_limit,
            exist_ok=True,
        ) as out:
            while True:
                sample = sample_queue.get()
                if sample is _SENTINEL:
                    break
                out.write(sample)
                counters[sid] += 1
    except Exception as exc:
        errors.append(exc)


def write_packed_rollout_dataset(
    samples: Iterable[dict[str, Any]],
    out_root: str | None,
    remote_root: str | None = None,
    num_streams: int = 4,
    compression: str | None = None,
    size_limit: int | str = "64mb",
    keep_local: bool = True,
) -> dict[str, Any]:
    if out_root is None and remote_root is None:
        raise ValueError("one of out_root or remote_root must be set")

    use_temp_local = out_root is None and remote_root is not None
    if use_temp_local:
        out_root = tempfile.mkdtemp(prefix="rollout_mds_")

    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)

    keep_local = bool(keep_local or (remote_root is None))
    if use_temp_local:
        keep_local = False

    stream_count = max(1, int(num_streams))
    queues = [Queue() for _ in range(stream_count)]
    counters = [0] * stream_count
    errors: list[Exception] = []
    threads: list[threading.Thread] = []
    for sid in range(stream_count):
        t = threading.Thread(
            target=_write_stream_partition,
            args=(
                sid,
                str(root),
                remote_root,
                queues[sid],
                compression,
                size_limit,
                keep_local,
                counters,
                errors,
            ),
            daemon=True,
        )
        t.start()
        threads.append(t)

    for idx, sample in enumerate(samples):
        if errors:
            break
        queues[idx % stream_count].put(sample)
    for q in queues:
        q.put(_SENTINEL)
    for t in threads:
        t.join()
    if errors:
        raise errors[0]

    written = sum(counters)
    if written <= 0:
        raise ValueError("cannot write streaming dataset: no rollout samples")
    stream_paths = [
        _join_uri(remote_root, f"stream_{sid:03d}") if remote_root else str(root / f"stream_{sid:03d}")
        for sid in range(stream_count)
    ]
    dataset_path = remote_root or str(root)

    local_path: str | None = str(root)
    if use_temp_local and not keep_local:
        shutil.rmtree(root, ignore_errors=True)
        local_path = None

    return {
        "dataset_path": dataset_path,
        "dataset_streams": stream_paths,
        "local_path": local_path,
        "num_samples": written,
        "num_streams": stream_count,
    }
