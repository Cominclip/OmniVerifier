import base64
import argparse
import torch
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams

def load_single_input(prompt, image_path=None, processor=None, system_prompt=""):

    inputs = []
    content = []
    with open(image_path, "rb") as f:
        img_data = f.read()
    encoded_image = base64.b64encode(img_data).decode("utf-8")
    content.append({
        "type": "image",
        "image": f"data:image/png;base64,{encoded_image}"
    })
    content.append({
        "type": "text",
        "text": prompt + system_prompt
    })

    messages = [{"role": "user", "content": content}]
    formatted_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, _ = process_vision_info(messages)
    inputs.append({
        "prompt": formatted_prompt,
        "multi_modal_data": {"image": image_inputs}
    })

    return inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str,
                          default='comin/OmniVerifier-7B',
                       help='Model path')
    parser.add_argument('--temperature', type=float, default=0.01)
    parser.add_argument('--top_p', type=float, default=0.001)
    parser.add_argument('--top_k', type=int, default=-1)
    parser.add_argument('--max_new_tokens', type=int, default=2048)
    parser.add_argument('--tp', type=int, default=4, help='Tensor parallel size')
    parser.add_argument('--max_num_seqs', type=int, default=32)
    args = parser.parse_args()
    
    SYS_PROMPT = " You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here"
    
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    inputs= load_single_input(
        prompt="Please describe the content of the image.",
        image_path="image_test.png",
        processor=processor,
        system_prompt=SYS_PROMPT,
        tokenizer=tokenizer
    )
    
    llm = LLM(
        model=args.model_name_or_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        limit_mm_per_prompt={"image": 10, "video": 2},
        enforce_eager=True,
        max_num_seqs=args.max_num_seqs,
        dtype=torch.bfloat16,
    )
    
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id] + getattr(tokenizer, 'additional_special_tokens_ids', [])
    )
    
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()