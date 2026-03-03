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
    local_root: str | None,
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
        remote_subdir = _join_uri(remote_root, stream_leaf) if remote_root else None
        #local_subdir = str(Path(local_root) / stream_leaf) if local_root else None
        #if local_subdir:
        #    Path(local_subdir).mkdir(parents=True, exist_ok=True)
        #if remote_subdir and local_subdir:
        #    out_path: str | tuple[str | None, str] = (local_subdir, remote_subdir)
        #elif remote_subdir:
        #    out_path = (None, remote_subdir)
        #else:
        #    out_path = str(local_subdir)

        out_path = remote_subdir

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
    from streaming.base.util import merge_index

    if out_root is None and remote_root is None:
        raise ValueError("one of out_root or remote_root must be set")
    keep_local = bool(keep_local or (remote_root is None))
    if out_root is None:
        keep_local = False
    root: Path | None = None
    if out_root is not None:
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
                str(root) if root is not None else None,
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
    if root is not None and remote_root:
        merge_out: str | tuple[str | None, str] = (str(root), remote_root)
    elif remote_root:
        merge_out = (None, remote_root)
    else:
        merge_out = str(root)
    merge_index(merge_out, keep_local=keep_local)
    index_file: str
    if remote_root:
        index_file = _join_uri(remote_root, "index.json")
    else:
        assert root is not None
        local_index = root / "index.json"
        if not local_index.is_file():
            raise RuntimeError(f"failed to merge streaming index at {local_index}")
        index_file = str(local_index)
    dataset_path = remote_root or str(root)

    return {
        "dataset_path": dataset_path,
        "local_path": str(root) if root is not None else None,
        "num_samples": written,
        "num_streams": stream_count,
        "index_file": index_file,
    }
