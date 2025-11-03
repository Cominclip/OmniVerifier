import json
import openai
import time
import random
from tqdm import tqdm
import base64

base_url = ""
api_version = "2024-03-01-preview"
ak = ""
model_name = "gpt-5-chat-2025-08-07"
max_tokens = 10000

client = openai.AzureOpenAI(
    azure_endpoint=base_url,
    api_version=api_version,
    api_key=ak,
)

def get_completion_with_retry(client, model, messages, max_tokens, max_retries=10):
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
            )
            return completion
        except:
            if attempt == max_retries - 1:
                return None
            wait_time = 5 + random.random()
            time.sleep(wait_time)

with open('complex-images-synthetic/data_with_selected_mask.json', 'r') as f:
    data = json.load(f)

final = []

for i in tqdm(range(len(data))):
    prompt = f"""**Image Captioning Instructions **

        **Role**
        You are a high-precision image captioner.
        Your goal is to generate an accurate, concise, and fact-based caption based on the given image with a red bounding box indicating the target object: {data[i]['selected_object']['class_name']}.

        ## **Generation Rules**

        1. **Object Inclusion**
        - The caption **must mention the object inside the bounding box**, describing its appearance and location accurately.
        - The caption should also include other important objects in the scene.
        2. **Attribute Details**
        - Describe observable attributes of objects, such as color, texture, shape, material, lighting, facial expressions, and actions.
        3. **Spatial Relationships**
        - Clearly describe spatial relations between objects (e.g., top/bottom, left/right, front/back, near/far).
        4. **Interactions**
        - Describe actions or interactions between objects (e.g., “sitting on,” “holding,” “leaning against”).
        5. **Factual Accuracy**
        - Only describe content that can be confirmed from the image.
        - Do not add guesses, subjective imagination, or irrelevant details.
        - If an attribute or detail cannot be confirmed, do not mention it.
        6. **Distinguishing Similar Objects**
        - If there are other objects of the same category in the image, clearly distinguish and describe the bounding box object's specific location and features.
        8. **Length Limit**
        - The caption must not exceed 60 English words.

        ## **Loop-Enabled Self-Correction Process**

        **Step 1 — Initial Draft**
        - Generate a caption following all rules.
        **Step 2 — Self-Check**
        - Verify the following:
        1. Caption explicitly mentions the bounding box object.
        2. Object's attributes and location are accurately described.
        3. Spatial and interaction relationships are included when visible.
        4. Description is based only on observable facts (no guesses or imaginary details).
        5. Caption length is ≤ 60 words.
        **Step 3 — Correction Loop**
        - If **any** check fails:
        - Revise the caption to meet all rules.
        - Re-run the self-check.
        - Repeat until all checks pass. """ + """

        ## **Final Output Format**

        Always output only the **final, corrected** caption in this format:

        [Your English caption, no more than 50 words]
        
        ## **Example**

        **Input**
        Bounding box object: teddy bear

        **Output**

        "Two girls stand on grass. The girl on the left, wearing a dress, holds a teddy bear. The girl on the right holds a red balloon, with a white house visible in the distance."

        -------------- """ + f"""

        **Input**
        Bounding box object: {data[i]['selected_object']['class_name']}

        **Output**"""
    
    image_path = f"bbox_images/{data[i]['id']}.png"
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")
    completion = get_completion_with_retry(
        client=client,
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=max_tokens
    )
    if completion is None or completion.model_dump_json() is None:
        output = None
    else:
        try:
            output = json.loads(completion.model_dump_json())['choices'][0]['message']['content']
        except:
            output = None

    
    final.append({
        "id": data[i]['id'],
        "image": data[i]['image'],
        "output": output,
    })

    with open(f'complex-images-synthetic/complex_image_recaption.json', 'w') as f:
        json.dump(final, f, indent=4)
    