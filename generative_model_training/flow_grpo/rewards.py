"""HTTP client for pointwise and pairwise OmniVerifier rewards."""

import base64
import io
import json
import math
from urllib.request import Request, urlopen


def _encode_image(image):
    from PIL import Image

    if isinstance(image, bytes):
        image_bytes = image
    else:
        if hasattr(image, "detach"):
            image = image.detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
        if not isinstance(image, Image.Image):
            import numpy as np

            pixels = np.asarray(image)
            if pixels.dtype.kind == "f":
                pixels = np.clip(pixels, 0, 1) * 255
            image = Image.fromarray(pixels.astype(np.uint8))
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
    return base64.b64encode(image_bytes).decode("ascii")


class VerificationReward:
    def __init__(self, mode, server_url, timeout=1800, transport=None):
        if mode not in ("pointwise", "pairwise"):
            raise ValueError(f"Unknown verification mode: {mode}")
        self.mode = mode
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.transport = transport or self._post

    def _post(self, payload):
        request = Request(
            self.server_url + "/score",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def __call__(self, images, instructions, initial_images, reference_images=None):
        count = len(images)
        if len(instructions) != count or len(initial_images) != count:
            raise ValueError("Generated images, instructions and initial images must have the same length")
        if self.mode == "pairwise" and (
            reference_images is None
            or len(reference_images) != count
            or any(image is None for image in reference_images)
        ):
            raise ValueError("Pairwise verification requires a reference_image for every generated image")
        if not count:
            return {self.mode + "_verification": [], "avg": []}, {"valid": [], "errors": []}

        samples = []
        for index in range(count):
            sample = {
                "id": index,
                "instruction": instructions[index],
                "initial_image": _encode_image(initial_images[index]),
                "generated_image": _encode_image(images[index]),
            }
            if self.mode == "pairwise":
                sample["reference_image"] = _encode_image(reference_images[index])
            samples.append(sample)

        response = self.transport({"mode": self.mode, "samples": samples})
        if response.get("mode") != self.mode:
            raise ValueError("Reward server returned the wrong verification mode")
        scores = response["scores"]
        valid = response.get("valid", [True] * count)
        errors = response.get("errors", [None] * count)
        if len(valid) != count or len(errors) != count:
            raise ValueError("Reward diagnostics must have the same length as the input batch")
        if len(scores) != count or any(
            not isinstance(is_valid, bool)
            or (
                is_valid
                and (isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score))
            )
            for score, is_valid in zip(scores, valid)
        ):
            raise ValueError("Reward server must return one finite numeric score per valid input, in input order")
        scores = [score if is_valid else None for score, is_valid in zip(scores, valid)]
        return {self.mode + "_verification": scores, "avg": scores}, {"valid": valid, "errors": errors}
