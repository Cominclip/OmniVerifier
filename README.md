<div align="center">
<h1>Generative Universal Verifier as Multimodal Meta-Reasoner</h1></div>

<p align="center">
  <a href="https://arxiv.org/abs/2509.06949">
    <img
      src="https://img.shields.io/badge/Paper-Arxiv-red?logo=arxiv&logoColor=red"
      alt="CURE Paper on arXiv"
    />
  <a href="https://huggingface.co/datasets/comin/ViVerBench">
    <img 
        src="https://img.shields.io/badge/ViVerBench-Hugging%20Face%20Data-orange?logo=huggingface&logoColor=yellow" 
        alt="Coding Datasets on Hugging Face"
    />
  </a>
  <a href="https://huggingface.co/collections/Gen-Verse/trado-series-68beb6cd6a26c27cde9fe3af">
    <img 
        src="https://img.shields.io/badge/OmniVerifier%207B-Hugging%20Face%20Model-FFCC00?logo=huggingface&logoColor=yellow" 
        alt="ReasonFlux Coders on Hugging Face"
    />
  </a>
</p>
    
### Introduction

We introduce **Generative Universal Verifier**, a novel concept and plugin designed for next-generation multimodal reasoning in vision-language models and unified multimodal models, providing the fundamental capability of reflection and refinement on visual outcomes during the reasoning and generation process. 

- **ViVerBench**: a comprehensive benchmark spanning 16 categories of critical tasks for evaluating visual outcomes in multimodal reasoning. 
- **OmniVerifier-7B**: Trained on large-scale visual verification data, the first omni-capable generative verifier trained for universal visual verification and achieves notable gains on ViVerBench(+8.3). 
-  **OmniVerifier-TTS**, a sequential test-time scaling paradigm that leverages the universal verifier to bridge image generation and editing within unified models, enhancing the upper bound of generative ability through iterative fine-grained optimization. 

OmniVerifier advances both reliable reflection during generation and scalable test-time refinement, marking a step toward more trustworthy and controllable next-generation reasoning systems.

### New Updates

**[2025.10]** Inference code of Sequential OmniVerifier-TTS (based on Qwen-Image) is released.

**[2025.10]** Evaluation code of ViVerBench is released.

**[2025.10]** Training code of OmniVerifier is released.

### TODO

- [ ] Two automated data construction pipelines
- [ ] Sequential OmniVerifier-TTS on different backbones
- [ ] Parallel OmniVerifier-TTS

### Installation

```bash
git clone https://github.com/Cominclip/OmniVerifier.git
cd OmniVerifier
pip install -e .
```

### Part1: ViVerBench Evaluation

We provide two evaluation approaches: **rule-based** and **model-based**. As a first step, store the model outputs in a JSON file such as `your_model.json`.

For rule-based evaluation:

```shell
python viverbench_eval_rule_based.py --model_response your_model.json
```

For model-based evaluation, we use GPT-4.1 as the judge model:

```shell
python viverbench_eval_model_based.py --model_response your_model.json
```

### Part2: OmniVerifier RL Training

We apply DAPO to directly train Qwen2.5VL-7B without cold start:

```bash
bash examples/qwen2_5_vl_7b_dapo.sh
```

After training, you should merge checkpoint in Hugging Face format:

```bash
python3 scripts/model_merger.py --local_dir checkpoints/omniverifier/exp_name/global_step_1/actor
```

### Part3: OmniVerifier-TTS

We provide the code for sequential Omniverifier-TTS using Qwen-Image. You should first generate the step0 image and use this script for iteratively self-refine:

```shell
python sequential_omniverifier_tts.py
```

## Citation

```
@article{zhang2025generative,
  author  = {Zhang, Xinchen and Zhang, Xiaoying and Wu, Youbin and Cao, Yanbin and Zhang, Renrui and Chu, Ruihang and Yang, Ling and Yang, Yujiu},
  title   = {Generative Universal Verifier as Multimodal Meta-Reasoner},
  journal = {arXiv preprint arXiv:2410.07171},
  year    = {2025}
}
```

## Acknowledgements

OmniVerifier is builded upon several solid works. Thanks to [EasyR1](https://github.com/hiyouga/EasyR1) and [veRL](https://github.com/volcengine/verl) for their wonderful work and codebase! 
