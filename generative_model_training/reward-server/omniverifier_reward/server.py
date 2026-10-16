"""HTTP service with eight independent vLLM replicas."""

import argparse
import logging
import os
import signal
from pathlib import Path

from . import config, prompts
from .dispatch import RewardPool
from .worker import VerifierWorker


def validate_payload(payload, mode):
    if not isinstance(payload, dict) or payload.get("mode") != mode:
        raise ValueError(f"This service requires mode={mode}")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("samples must be a nonempty list")
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or sample.get("id") != index:
            raise ValueError("Sample IDs must match input order: 0, 1, ...")
        fields = ["instruction", "initial_image", "generated_image"]
        if mode == "pairwise":
            fields.append("reference_image")
        elif "reference_image" in sample:
            raise ValueError("Pointwise requests must omit reference_image")
        for field in fields:
            if not isinstance(sample.get(field), str) or not sample[field].strip():
                raise ValueError(f"Sample {index} requires nonempty {field}")
    return samples


def create_app(settings, pool):
    from flask import Flask, jsonify, request

    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify(
            ready=True,
            process_id=os.getpid(),
            mode=settings["mode"],
            workers=len(pool.workers),
            max_input_tokens=settings["max_input_tokens"],
            max_output_tokens=settings["max_output_tokens"],
        )

    @app.post("/score")
    def score():
        try:
            samples = validate_payload(request.get_json(), settings["mode"])
        except ValueError as error:
            return jsonify(error=str(error)), 400
        try:
            results = pool.score(samples)
        except Exception:
            app.logger.exception("Verifier inference failed")
            return jsonify(error="Verifier inference failed; check reward server logs"), 500
        return jsonify(
            mode=settings["mode"],
            scores=[result["score"] for result in results],
            valid=[result["valid"] for result in results],
            errors=[result["error"] for result in results],
        )

    return app


def parse_settings(mode, argv=None):
    parser = argparse.ArgumentParser(description=f"Eight-GPU {mode} OmniVerifier reward service")
    default_model = getattr(config, mode + "_verifier_model_path")
    parser.add_argument("--model-path", default=os.getenv(mode.upper() + "_VERIFIER_MODEL_PATH", default_model))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18085 if mode == "pointwise" else 18086)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = parser.parse_args(argv)
    evaluation = getattr(prompts, mode + "_evaluation")
    if not args.model_path.strip():
        parser.error(f"Fill {mode}_verifier_model_path in omniverifier_reward/config.py or pass --model-path")
    if not evaluation.strip():
        parser.error(f"Fill {mode}_evaluation in omniverifier_reward/prompts.py")
    if args.max_input_tokens <= 0 or args.max_output_tokens <= 0:
        parser.error("Token limits must be positive")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("GPU memory utilization must be between 0 and 1")
    return {"mode": mode, "evaluation": evaluation, **vars(args)}


def main(mode):
    settings = parse_settings(mode)
    logging.basicConfig(level=logging.INFO)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    import ray

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    package_root = str(Path(__file__).resolve().parents[1])
    worker_python_path = package_root + os.pathsep + os.environ.get("PYTHONPATH", "")
    ray.init(include_dashboard=False, runtime_env={"env_vars": {"PYTHONPATH": worker_python_path}})
    try:
        if ray.cluster_resources().get("GPU", 0) < 8:
            raise RuntimeError("Reward deployment requires eight visible GPUs")
        worker_class = ray.remote(num_gpus=1)(VerifierWorker)
        workers = [worker_class.remote(settings) for _ in range(8)]
        ray.get([worker.ready.remote() for worker in workers])
        pool = RewardPool(workers, lambda worker, samples: worker.score.remote(samples), ray.get)
        create_app(settings, pool).run(host=settings["host"], port=settings["port"], threaded=True, use_reloader=False)
    finally:
        ray.shutdown()
