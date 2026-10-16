from setuptools import find_packages, setup

setup(
    name="omniverifier-diffusion-nft",
    version="0.1.0",
    description="Qwen-Image-Edit-2509 DiffusionNFT with pointwise/pairwise verifier rewards",
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.6",
        "torchvision",
        "transformers>=4.57",
        "accelerate",
        "diffusers>=0.36.0",
        "numpy",
        "pillow>=10.4",
        "tqdm",
        "wandb",
        "peft>=0.17.0",
        "absl-py",
        "ml_collections",
        "safetensors",
        "sentencepiece",
    ],
    extras_require={"dev": ["pytest", "ruff"]},
)
