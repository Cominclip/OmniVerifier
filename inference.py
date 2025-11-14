import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "comin/OmniVerifier-7B", torch_dtype=torch.bfloat16, device_map="auto"
)
# default processer
processor = AutoProcessor.from_pretrained("comin/OmniVerifier-7B")

SYS_PROMPT = " You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here"

image_path = '' # please replace it with your own image path
prompt = '' # please replace it with the prompt you use to generate the image

question = f"""This image was generated from the prompt: {prompt}. 
    Please carefully analyze the image and determine whether all the objects, attributes, and spatial relationships mentioned in the prompt are correctly represented in the image. 

    If the image accurately reflects the prompt, please answer 'true'; otherwise, answer 'false'.  

    Respond strictly in the following JSON format: """ + """

    {
        "answer": true/false,
        "explanation": "If the answer is false, briefly summarize the main error.",
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
            {"type": "text", "text": SYS_PROMPT + question},
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
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
