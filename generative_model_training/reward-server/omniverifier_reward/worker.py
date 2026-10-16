"""A vLLM replica created inside a Ray actor owning exactly one GPU."""

import base64
import logging
from io import BytesIO

from .prompts import build_messages
from .scoring import extract_global_score

logger = logging.getLogger(__name__)


class VerifierWorker:
    def __init__(self, settings):
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        self.settings = settings
        self.processor = AutoProcessor.from_pretrained(settings["model_path"], trust_remote_code=True)
        self.llm = LLM(
            model=settings["model_path"],
            trust_remote_code=True,
            tensor_parallel_size=1,
            max_model_len=settings["max_input_tokens"] + settings["max_output_tokens"],
            max_num_batched_tokens=settings["max_input_tokens"] + settings["max_output_tokens"],
            limit_mm_per_prompt={"image": 3 if settings["mode"] == "pairwise" else 2},
            gpu_memory_utilization=settings["gpu_memory_utilization"],
            dtype="bfloat16",
        )
        self.sampling = SamplingParams(temperature=0.0, max_tokens=settings["max_output_tokens"], n=1)

    def ready(self):
        return True

    @staticmethod
    def _decode_image(encoded):
        from PIL import Image

        with Image.open(BytesIO(base64.b64decode(encoded, validate=True))) as source:
            return source.convert("RGB")

    def _prepare(self, sample):
        images = [self._decode_image(sample["initial_image"]), self._decode_image(sample["generated_image"])]
        reference = None
        if self.settings["mode"] == "pairwise":
            reference = self._decode_image(sample["reference_image"])
            images.append(reference)
        messages = build_messages(self.settings["evaluation"], sample["instruction"], images[0], images[1], reference)
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Count expanded image tokens too; never truncate an image or instruction.
        processed = self.processor(text=[prompt], images=images, return_tensors="pt")
        token_count = processed["input_ids"].shape[-1]
        if token_count > self.settings["max_input_tokens"]:
            raise ValueError(
                f"Verifier input has {token_count} tokens; maximum is {self.settings['max_input_tokens']}"
            )
        return {"prompt": prompt, "multi_modal_data": {"image": images}}

    def score(self, samples):
        requests, indices = [], []
        results = [None] * len(samples)
        for index, sample in enumerate(samples):
            try:
                requests.append(self._prepare(sample))
                indices.append(index)
            except (ValueError, OSError) as error:
                logger.warning("Invalid verifier input for sample %s: %s", sample["id"], error)
                results[index] = {"score": None, "valid": False, "error": str(error)}
        if requests:
            outputs = self.llm.generate(requests, self.sampling, use_tqdm=False)
            if len(outputs) != len(requests):
                raise RuntimeError("vLLM returned fewer outputs than requested")
            for index, output in zip(indices, outputs):
                result = extract_global_score(output.outputs[0].text)
                if not result["valid"]:
                    logger.warning("Invalid verifier output for sample %s: %s", samples[index]["id"], result["error"])
                results[index] = result
        return results
