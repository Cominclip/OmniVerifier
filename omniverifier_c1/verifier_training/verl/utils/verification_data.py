"""Image inputs for pointwise and pairwise verifier training."""

from typing import Any, Optional


def _single_image(value: Any, key: str) -> Any:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Verifier field '{key}' must contain exactly one image.")
        value = value[0]
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"Verifier field '{key}' must contain exactly one image.")
    return value


def prepare_verification_example(
    example: dict[str, Any],
    mode: Optional[str],
    image_key: str = "images",
    reference_image_key: str = "reference_image",
    initial_image_key: str = "initial_image",
) -> dict[str, Any]:
    """Copy a row and assemble initial, generated, and optional reference images."""
    prepared = dict(example)
    if mode is None:
        return prepared
    if mode not in ("pointwise", "pairwise"):
        raise ValueError(f"Unsupported verification mode: {mode}")
    if len({image_key, initial_image_key, reference_image_key}) != 3:
        raise ValueError("Initial, generated, and reference images must use different keys.")

    initial_image = _single_image(prepared.pop(initial_image_key, None), initial_image_key)
    images = [initial_image, _single_image(prepared.get(image_key), image_key)]
    reference_image = prepared.pop(reference_image_key, None)
    if mode == "pairwise":
        images.append(_single_image(reference_image, reference_image_key))
    prepared[image_key] = images
    return prepared


def build_verification_content(prompt: str, mode: str) -> list[dict[str, Any]]:
    """Insert labelled image slots without requiring dataset image placeholders."""
    if mode not in ("pointwise", "pairwise"):
        raise ValueError(f"Unsupported verification mode: {mode}")
    content = [
        {"type": "text", "text": "Initial Image:"},
        {"type": "image"},
        {"type": "text", "text": "Generated Image:"},
        {"type": "image"},
    ]
    if mode == "pairwise":
        content.extend([{"type": "text", "text": "Reference Image:"}, {"type": "image"}])
    content.append({"type": "text", "text": prompt.replace("<image>", "").strip()})
    return content
