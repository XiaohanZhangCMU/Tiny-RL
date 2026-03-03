"""Rollout server that owns vLLM GPUs and exposes rollout + weight-sync endpoints."""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, Request

from utils import (
    load_config,
    load_model,
    create_vllm_engine,
    destroy_vllm_engine,
    vllm_rollout,
    group_advantages,
    init_rng,
)
from weight_sync.transport import build_transport

log = logging.getLogger(__name__)
app = FastAPI()


def _join_uri(root: str, leaf: str) -> str:
    return f"{root.rstrip('/')}/{leaf.lstrip('/')}"


def _resolve_engine_model(engine):
    candidates = [
        ["llm_engine", "model_executor", "driver_worker", "model_runner", "model"],
        ["llm_engine", "model_executor", "driver_worker", "worker", "model_runner", "model"],
        ["engine", "model"],
    ]
    for path in candidates:
        cur = engine
        ok = True
        for part in path:
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok and hasattr(cur, "named_parameters"):
            return cur
    return None


def _collect_param_metadata_local(engine) -> list[dict[str, Any]]:
    model = _resolve_engine_model(engine)
    if model is None:
        return []
    out: list[dict[str, Any]] = []
    for name, p in model.named_parameters():
        if not isinstance(p, torch.Tensor):
            continue
        if not p.is_cuda:
            continue
        out.append(
            {
                "name": name,
                "ptr": int(p.data_ptr()),
                "nbytes": int(p.numel() * p.element_size()),
                "dtype": str(p.dtype).replace("torch.", ""),
                "shape": list(p.shape),
            }
        )
    return out


def _collect_memory_regions_local() -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    if not torch.cuda.is_available():
        return regions
    try:
        snapshot = torch.cuda.memory_snapshot()
    except Exception:
        return regions
    for seg in snapshot:
        size = int(seg.get("total_size", 0))
        ptr = int(seg.get("address", 0))
        if ptr and size > 0:
            regions.append({"ptr": ptr, "size": size})
    return regions


class RolloutServer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.mcfg = cfg["model"]
        self.rcfg = cfg["rollout"]
        self.scfg = cfg.get("server", {})
        self.wcfg = self.scfg.get("weight_sync", {})
        self.num_infer_gpus = len(cfg.get("gpu_split", {}).get("inference", [0]))
        self.transport = build_transport(self.wcfg.get("transport", "mock"))
        self.last_synced_step = -1

        _, self.tokenizer = load_model(
            self.mcfg["name"],
            trust_remote_code=self.mcfg["trust_remote_code"],
            bf16=self.mcfg["bf16"],
            device_map="cpu",
        )
        weight_backend = self.wcfg.get("backend", "disk")
        self.engine = create_vllm_engine(
            self.mcfg["name"],
            self.rcfg["gpu_memory_utilization"],
            self.rcfg["max_length"],
            tensor_parallel_size=self.num_infer_gpus,
            weight_transfer_backend="nccl" if weight_backend == "nccl" else "",
        )
        self.pad_id = self.tokenizer.eos_token_id
        ocfg = self.scfg.get("online_dataset", {})
        local_root = ocfg.get("local_root", None)
        self.online_dataset_local_root = str(local_root) if local_root else None
        self.online_dataset_remote_root = ocfg.get("remote_root")
        self.online_dataset_num_streams = int(ocfg.get("num_streams", 4))
        self.online_dataset_compression = ocfg.get("compression")
        self.online_dataset_size_limit = ocfg.get("size_limit", "64mb")
        self.online_dataset_keep_local = bool(ocfg.get("keep_local", True))
        self.online_dataset_max_keep = int(ocfg.get("max_keep_local_steps", 2))
        self.online_dataset_run_id = str(ocfg.get("run_id", f"run_{int(time.time())}"))
        self._dataset_step = 0
        self._recent_dataset_dirs: list[str] = []

    def _remember_dataset_dir(self, local_path: str):
        if self.online_dataset_max_keep <= 0:
            return
        self._recent_dataset_dirs.append(local_path)
        while len(self._recent_dataset_dirs) > self.online_dataset_max_keep:
            old_path = self._recent_dataset_dirs.pop(0)
            shutil.rmtree(old_path, ignore_errors=True)

    def generate(self, questions: list[str], answers: list[str]) -> dict[str, Any]:
        t0 = time.perf_counter()
        results = vllm_rollout(
            self.engine,
            self.tokenizer,
            questions,
            answers,
            group_size=self.rcfg["group_size"],
            max_length=self.rcfg["max_length"],
            temperature=self.rcfg["temperature"],
            top_p=self.rcfg["top_p"],
        )
        rollout_time = max(time.perf_counter() - t0, 1e-6)

        all_rewards: list[float] = []
        stats = {"generated_tokens": 0}
        step_id = self._dataset_step
        self._dataset_step += 1
        local_dir = None
        if self.online_dataset_local_root:
            local_dir = Path(self.online_dataset_local_root) / self.online_dataset_run_id / f"step_{step_id:08d}"
        remote_dir = None
        if self.online_dataset_remote_root:
            remote_dir = _join_uri(
                _join_uri(self.online_dataset_remote_root, self.online_dataset_run_id),
                f"step_{step_id:08d}",
            )

        def _iter_samples():
            for seq_ids, returns, act_mask, vllm_lp, _completions in results:
                attn_mask = seq_ids != self.pad_id
                adv = group_advantages(returns)
                all_rewards.extend(returns.squeeze(-1).tolist())
                stats["generated_tokens"] += int(act_mask.sum())
                yield {
                    "sequences": seq_ids.tolist(),
                    "returns": returns.tolist(),
                    "advantages": adv.tolist(),
                    "action_mask": act_mask.tolist(),
                    "attention_mask": attn_mask.tolist(),
                    "action_log_probs": vllm_lp.tolist(),
                }

        # Lazy import: keep streaming package out of startup path so vLLM CUDA
        # worker bootstrap happens before any optional dataset stack side effects.
        from online_dataset import write_packed_rollout_dataset
        dataset_info = write_packed_rollout_dataset(
            _iter_samples(),
            out_root=str(local_dir) if local_dir is not None else None,
            remote_root=remote_dir,
            num_streams=self.online_dataset_num_streams,
            compression=self.online_dataset_compression,
            size_limit=self.online_dataset_size_limit,
            keep_local=self.online_dataset_keep_local,
        )
        if self.online_dataset_keep_local and dataset_info["local_path"]:
            self._remember_dataset_dir(dataset_info["local_path"])

        reward_mean = sum(all_rewards) / max(len(all_rewards), 1)
        accuracy = sum(1 for r in all_rewards if r >= 0.8) / max(len(all_rewards), 1)
        rollout_tokens_per_sec = stats["generated_tokens"] / rollout_time
        return {
            "status": "ok",
            "dataset_path": dataset_info["dataset_path"],
            "dataset_streams": dataset_info["dataset_streams"],
            "num_samples": dataset_info["num_samples"],
            "num_streams": dataset_info["num_streams"],
            "reward_mean": reward_mean,
            "accuracy": accuracy,
            "rollout_tokens_per_sec": rollout_tokens_per_sec,
        }

    def init_weight_sync(self, mode: str, init_info: dict[str, Any]) -> dict[str, Any]:
        if mode == "nccl":
            try:
                self.engine.init_weight_transfer_engine({"init_info": init_info})
            except TypeError:
                self.engine.init_weight_transfer_engine(init_info=init_info)
            return {"status": "ok", "mode": "nccl"}

        if mode == "rdma":
            merged = dict(self.wcfg.get("transport_init", {}))
            merged.update(init_info)
            merged.setdefault("role", "client")
            master_address = str(init_info.get("master_address", merged.get("master_address", "0.0.0.0")))
            master_port = int(init_info.get("master_port", merged.get("master_port", 6000)))
            if merged["role"] == "client":
                merged.setdefault("peer_host", master_address)
                merged.setdefault("peer_port", master_port)
            else:
                merged.setdefault("listen_host", "0.0.0.0")
                merged.setdefault("listen_port", master_port)
            merged.setdefault("tcp_timeout_s", 300)
            return self.transport.init_endpoint(**merged)

        return {"status": "ok", "mode": "disk"}

    def update_weights_nccl(self, update_info: dict[str, Any]) -> dict[str, Any]:
        try:
            self.engine.update_weights({"update_info": update_info})
        except TypeError:
            self.engine.update_weights(update_info=update_info)
        return {"status": "ok", "mode": "nccl"}

    def reload_from_disk(self, model_path: str) -> dict[str, Any]:
        destroy_vllm_engine(self.engine)
        self.engine = create_vllm_engine(
            model_path,
            self.rcfg["gpu_memory_utilization"],
            self.rcfg["max_length"],
            tensor_parallel_size=self.num_infer_gpus,
        )
        return {"status": "ok", "mode": "disk", "model_path": model_path}

    def get_param_metadata(self) -> dict[str, Any]:
        data = [{"rank": 0, "params": _collect_param_metadata_local(self.engine)}]
        return {"status": "ok", "workers": data}

    def get_memory_regions(self) -> dict[str, Any]:
        data = [{"rank": 0, "regions": _collect_memory_regions_local()}]
        return {"status": "ok", "workers": data}

    def register_mrs(self, regions: list[dict[str, Any]]) -> dict[str, Any]:
        mrs = self.transport.register_memory_regions(regions)
        return {"status": "ok", "mrs": mrs}

    def apply_rdma_routes(self, routes: list[dict[str, Any]], step: int) -> dict[str, Any]:
        total_bytes = sum(int(r.get("nbytes", 0)) for r in routes)
        self.last_synced_step = max(self.last_synced_step, int(step))
        log.info("RDMA weight sync step %d: %d routes, %.1f MB",
                 step, len(routes), total_bytes / (1024 * 1024))
        return {
            "status": "ok",
            "mode": "rdma",
            "num_routes": len(routes),
            "total_bytes": total_bytes,
            "ack_step": self.last_synced_step,
        }


server: RolloutServer | None = None


@app.get("/health")
async def ep_health():
    return {"status": "ok", "synced_step": server.last_synced_step if server else -1}


@app.post("/generate")
async def ep_generate(request: Request):
    data = await request.json()
    return server.generate(data["questions"], data["answers"])


@app.post("/init_weight_sync")
async def ep_init_weight_sync(request: Request):
    data = await request.json()
    return server.init_weight_sync(mode=data["mode"], init_info=data.get("init_info", {}))


@app.post("/update_weights_nccl")
async def ep_update_weights_nccl(request: Request):
    data = await request.json()
    return server.update_weights_nccl(update_info=data["update_info"])


@app.post("/reload_from_disk")
async def ep_reload_from_disk(request: Request):
    data = await request.json()
    return server.reload_from_disk(model_path=data["model_path"])


@app.post("/get_param_metadata")
async def ep_get_param_metadata():
    return server.get_param_metadata()


@app.post("/get_memory_regions")
async def ep_get_memory_regions():
    return server.get_memory_regions()


@app.post("/register_mrs")
async def ep_register_mrs(request: Request):
    data = await request.json()
    return server.register_mrs(regions=data.get("regions", []))


@app.post("/apply_rdma_routes")
async def ep_apply_rdma_routes(request: Request):
    data = await request.json()
    return server.apply_rdma_routes(routes=data.get("routes", []), step=int(data.get("step", -1)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--port", type=int, default=7000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    init_rng(cfg["training"]["seed"])
    server = RolloutServer(cfg)

    log.info("Rollout server listening on port %d", args.port)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
