<div align="center">
<h1 style="font-size: 0.8em; margin: 0.4em 0;">Generative Universal Verifier as Multimodal <br>Meta-Reasoner</h1></div>

<p align="center">
  <a href="https://arxiv.org/abs/2510.13804">
    <img
      src="https://img.shields.io/badge/Paper-OmniVerifier-red?logo=OmniVerifier&logoColor=red"
      alt="OmniVerifier paper on arXiv"
    />
  </a>
  <a href="https://arxiv.org/abs/2605.28805">
    <img
      src="https://img.shields.io/badge/Paper-OmniVerifier--M1-red?logo=OmniVerifier-M1&logoColor=red"
      alt="OmniVerifier-M1 paper on arXiv"
    />
  </a>
  <a href="https://arxiv.org/abs/2610.22355">
    <img
      src="https://img.shields.io/badge/Paper-OmniVerifier--C1-red?logo=OmniVerifier-C1&logoColor=red"
      alt="OmniVerifier-C1 paper on arXiv"
    />
  </a>
</p>

<p align="center">
  <a href="https://huggingface.co/datasets/comin/ViVerBench">
    <img 
        src="https://img.shields.io/badge/ViVerBench-Hugging%20Face%20Data-orange?logo=huggingface&logoColor=yellow" 
        alt="Coding Datasets on Hugging Face"
    />
  </a>
  <a href="https://huggingface.co/comin/OmniVerifier-7B">
    <img 
        src="https://img.shields.io/badge/OmniVerifier%207B-Hugging%20Face%20Model-FFCC00?logo=huggingface&logoColor=yellow" 
        alt="ReasonFlux Coders on Hugging Face"
    />
  </a>
</p>

### Introduction

We introduce **Generative Universal Verifier**, a novel concept and plugin designed for next-generation multimodal reasoning in vision-language models and unified multimodal models, providing the fundamental capability of reflection and refinement on visual outcomes during the reasoning and generation process. 

- **ViVerBench**: A comprehensive benchmark spanning 16 categories of critical tasks for evaluating visual outcomes in multimodal reasoning. 
- **OmniVerifier**: Trained on large-scale visual verification data, the first omni-capable generative verifier trained for universal visual verification and achieves notable gains on ViVerBench(+8.3). 
- **OmniVerifier-TTS**: A sequential test-time scaling paradigm that leverages the universal verifier to bridge image generation and editing within unified models, enhancing the upper bound of generative ability through iterative fine-grained optimization.
- **OmniVerifier-M1**: A generalist multimodal meta-verifier that leverages symbolic meta-verification and decoupled RL training to achieve robust visual verification, fine-grained error localization, and state-of-the-art performance on ViVerBench.
- **M1-TTS**: An agentic correction system that support robust visual verification and effective region-level self-correction across diverse generative foundation models.
- **OmniVerifier-C1**: A decomposed graded verifier trained with Decoupled RL to provide fine-grained, interpretable continuous rewards for text-to-image generation, with reference-based pairwise verification enabling more accurate and reliable feedback for rule-governed world modeling.


The OmniVerifier series advances visual verification to critique generated outputs and provide reliable feedback for improving generation quality during both training and inference, guiding test-time refinement or generative model optimization through reinforcement learning.


### New Updates

**[2026.09]** [OmniVerifier-C1](https://arxiv.org/abs/2610.22355) is accepted at **NeurIPS 2026**! It introduces graded scoring and pairwise verification to provide finer-grained, more reliable feedback for rule-governed world modeling.

**[2026.05]** [OmniVerifier-M1](https://arxiv.org/abs/2605.28805) is accepted by **ICML 2026** , advancing multimodal verifiers through symbolic meta-verification.

**[2026.02]** [OmniVerifier](https://arxiv.org/abs/2510.13804) is accepted by ICLR 2026 **(Oral Paper, Top 1%)**.

**[2025.11]** Inference code of two automated pipelines for visual verifier data construction is released.

**[2025.10]** Inference code of Sequential OmniVerifier-TTS (based on Qwen-Image) is released.

**[2025.10]** Evaluation code of ViVerBench is released.

**[2025.10]** Training code of OmniVerifier is released.



### Installation

```bash
git clone https://github.com/Cominclip/OmniVerifier.git
cd OmniVerifier
pip install -e .
```

### Quick Start: Generated Image Verification

Use the following command to test **OmniVerifier-7B** on a generated image:

```shell
python inference.py
```

Please modify `image_path` and `prompt` to your own settings.

The model will output both an **answer** and an **explanation**, indicating whether the image is strictly aligned with the given prompt.

Run the commands below from the repository root.

### 1. ViVerBench Evaluation

Save model responses as a JSON file, for example `your_model.json`. Choose either evaluation method:

```bash
python viverbench_eval_rule_based.py --model_response your_model.json
```

The model-based evaluator uses GPT-4.1 through Azure OpenAI. Fill in `base_url`, `ak`, and `model_name` in `viverbench_eval_model_based.py` before running it:

```bash
python viverbench_eval_model_based.py --model_response your_model.json
```

### 2. Verifier Training

Run the following commands from the repository root. OmniVerifier and OmniVerifier-M1 use `examples/config.yaml`; OmniVerifier-C1 uses `examples/config_c1.yaml`.

#### OmniVerifier

Train Qwen2.5-VL-7B with DAPO.

```bash
bash examples/qwen2_5_vl_7b_dapo.sh
```

#### OmniVerifier-M1


**Decoupled training:**

```bash
bash examples/M1_decoupled_training.sh
```

**Joint training:**

```bash
bash examples/M1_joint_training.sh
```

#### OmniVerifier-C1


**Pointwise verifier** Evaluate the generated image against the initial image.

```bash
bash examples/pointwise_c1.sh
```

**Pairwise verifier** Evaluate the generated image against the initial image and a reference image.

```bash
bash examples/pairwise_c1.sh
```


### 3. TTS

OmniVerifier-TTS uses OmniVerifier-7B to verify each Qwen-Image-Edit revision. Set `prompt` and `step0_image_path` in `sequential_omniverifier_tts.py`, then run:

```bash
python sequential_omniverifier_tts.py
```

### 4. Generative Model Training

Both configurations use `generative_model_training/config/qwen_image_edit_nft.py` to train Qwen-Image-Edit-2509 with DiffusionNFT and use the matching C1 verifier for rewards.

**Pointwise verification:**
```bash
bash generative_model_training/examples/pointwise_verification.sh
```

**Pairwise verification:**

```bash
bash generative_model_training/examples/pairwise_verification.sh
```

## Citation

```
@article{zhang2025generative,
  title={Generative Universal Verifier as Multimodal Meta-Reasoner},
  author={Zhang, Xinchen and Zhang, Xiaoying and Wu, Youbin and Cao, Yanbin and Zhang, Renrui and Chu, Ruihang and Yang, Ling and Yang, Yujiu},
  journal={arXiv preprint arXiv:2510.13804},
  year={2025}
}
@article{zhang2026omniverifier,
  title={OmniVerifier-M1: Multimodal Meta-Verifier with Explicit Structured Recalibration},
  author={Zhang, Xinchen and Liu, Bowei and Liu, Jiale and Shi, Chufan and Zhang, Yizhen and Liu, Junhong and Zhang, Youliang and Li, Zhiheng and Yang, Yujiu and Yang, Ling},
  journal={arXiv preprint arXiv:2605.28805},
  year={2026}
}
@article{liu2026omniverifier,
  title={OmniVerifier-C1: Decomposed Graded Verifier for Rule-Governed Generative World Modeling},
  author={Liu, Bowei and Zhang, Xinchen and Li, Xuhuan and Zhang, Tianlin and Jiang, Kaian and Liu, Jingwei and Zhao, Congyang and Qian, Yifei and Liu, Junhong and Li, Zhiheng and Yang, Yujiu and Cai, Yang and Yang, Ling},
  journal={arXiv preprint arXiv:2610.22355},
  year={2026}
}
```

## Acknowledgements

OmniVerifier is built upon several solid works. Thanks to [EasyR1](https://github.com/hiyouga/EasyR1) and [veRL](https://github.com/volcengine/verl) for their wonderful work and codebase! 
