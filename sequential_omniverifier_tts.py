import torch
import json
import random
import os
import time
from PIL import Image
import openai
import base64
import shutil
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from diffusers import QwenImageEditPipeline
from qwen_vl_utils import process_vision_info

pipe = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")
print("pipeline loaded")
pipe.to(torch.bfloat16)
pipe.to("cuda")
pipe.set_progress_bar_config(disable=None)

# --------------------Verifier Initialization--------------------
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "comin/OmniVerifier-7B", torch_dtype=torch.bfloat16, device_map="auto"
)
processor = AutoProcessor.from_pretrained("comin/OmniVerifier-7B")


def verify_image(image_path: str, prompt: str):
    SYS_PROMPT = " You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here"
    question = f"""This image was generated from the prompt: {prompt}. 
    Please carefully analyze the image and determine whether all the objects, attributes, and spatial relationships mentioned in the prompt are correctly represented in the image. 

    If the image accurately reflects the prompt, please answer 'true'; otherwise, answer 'false'.  

    When the answer is false, you must:
    1. Identify the main error and describe it briefly in "explanation".
    2. In "edit_prompt", provide a **concrete image editing instruction** to fix the error.  
    - The instruction must specify the exact action (e.g., add / remove / replace / move).  
    - The instruction must specify the location or reference point (e.g., "delete the bottle in the bottom-right corner", "add a dog next to the left pillar").  
    - Do not give vague instructions such as "add more bottles" or "ensure the count is correct". Be precise and actionable.  

    Respond strictly in the following JSON format: """ + """

    {
        "answer": true/false,
        "explanation": "If the answer is false, briefly summarize the main error.",
        "edit_prompt": "If the answer is false, provide a concrete and location-specific editing instruction."
    }
    """
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": question + SYS_PROMPT},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    try:
        output_json = json.loads(output_text[0].split('</think>')[1])
        answer = output_json.get("answer", False)
        edit_prompt = output_json.get("edit_prompt", 'remain unchanged')
    except:
        return False, 'remain unchanged'
    return answer, edit_prompt


def sequential_scaling(image_input_path: str, prompt: str, save_folder=None, max_steps=10):
    verification_data = []
    os.makedirs(save_folder, exist_ok=True)
    shutil.copy(image_input_path, os.path.join(save_folder, f"step_0.png"))
    answer, edit_prompt = verify_image(image_input_path, prompt)
    print(f"Initial verification, answer: {answer}, edit prompt: {edit_prompt}")
    verification_data.append({
        "prompt": prompt,
        "step": 0,
        "image_path": os.path.join(save_folder, f"step_0.png"),
        "answer": answer,
        "edit_prompt": edit_prompt
    })
    need_refine = not answer
    step = 0
    while need_refine and step < max_steps-1:
        print(f"Refinement step {step+1}, edit prompt: {edit_prompt}")
        image = pipe(
            image=Image.open(os.path.join(save_folder, f"step_{step}.png")).convert("RGB"),
            prompt=edit_prompt,
            width=1024,
            height=1024,
            num_inference_steps=50,
            true_cfg_scale=4.0,
            generator=torch.Generator(device="cuda").manual_seed(random.randint(0, 10000))
        ).images[0]

        step += 1   
        image.save(os.path.join(save_folder, f"step_{step}.png"))
        answer, edit_prompt = verify_image(os.path.join(save_folder, f"step_{step}.png"), prompt)
        verification_data.append({
            "step": step,
            "image_path": os.path.join(save_folder, f"step_{step}.png"),
            "answer": answer,
            "edit_prompt": edit_prompt
        })
        need_refine = not answer
    if not need_refine:
        if step == 0:
            shutil.copy(image_input_path, os.path.join(save_folder, f"step_final.png"))
        else:
            print(f"Final verification passed at step {step}.")
            image.save(os.path.join(save_folder, f"step_final.png"))
            verification_data.append({
                "step": "final",
                "image_path": os.path.join(save_folder, f"step_final.png"),
                "answer": True,
                "edit_prompt": None
            })
    with open(os.path.join(save_folder, "verification.json"), 'w') as f:
        json.dump(verification_data, f, indent=4)



prompt = 'a dog and two cats are sitting under a table, with a large book on top of the table'
step0_image_path = "images/step_0.png"
sequential_scaling(step0_image_path, prompt, save_folder="images", max_steps=10)
