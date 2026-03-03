from __future__ import annotations

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

        subdir = Path(local_root) / f"stream_{sid:03d}"
        subdir.mkdir(parents=True, exist_ok=True)
        out_path: str | tuple[str, str] = str(subdir)
        if remote_root:
            out_path = (str(subdir), _join_uri(remote_root, subdir.name))
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
    out_root: str,
    remote_root: str | None = None,
    num_streams: int = 4,
    compression: str | None = None,
    size_limit: int | str = "64mb",
    keep_local: bool = True,
) -> dict[str, Any]:
    from streaming.base.util import merge_index

    keep_local = bool(keep_local or (remote_root is None))
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)

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

    # Merge stream_* indexes into root/index.json that trainers can open directly.
    merge_out: str | tuple[str, str] = str(root)
    if remote_root:
        merge_out = (str(root), remote_root)
    merge_index(merge_out, keep_local=keep_local)
    index_file = root / "index.json"
    if keep_local and not index_file.is_file():
        raise RuntimeError(f"failed to merge streaming index at {index_file}")
    dataset_path = remote_root or str(root)

    return {
        "dataset_path": dataset_path,
        "local_path": str(root),
        "num_samples": written,
        "num_streams": stream_count,
        "index_file": _join_uri(dataset_path, "index.json"),
    }
