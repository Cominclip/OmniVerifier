import json
import openai
import time
import random
from tqdm import tqdm

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
            wait_time = 2 + random.random()
            time.sleep(wait_time)

with open('complex_image_recaption.json', 'r') as f:
    data = json.load(f)

final = []

for i in tqdm(range(len(data))):

    prompt = f"""You are a powerful image caption editor. Your task is to make **subtle yet important modifications** to key attribute details in an image caption (e.g., color, texture, shape, material, pose, facial expression, numbers, letters, quantities, etc.). The requirement is that the modified detail must be **significant enough that an image generated from the original caption would not satisfy the modified caption**.

    **Rules:**

    1. You are allowed to make **only one modification**.
    2. Your modifications should be **diverse** and not limited to a single type of attribute(such as color). Any attribute-related description can be the target of your edits, such as shape, number, color, letter, quantity, expression, pose, and so on.
    3. Vague or ambiguous modifications (e.g., changing “white” to “off-white,” or “some” to “a few”) are **not valid** because their boundaries are hard to define.
    4. You must output the json format, with the following two keys:
    - The modified prompt (after your edit).
    - The specific key detail you changed. """ + """


    **Example 1:**
    **Input:**

    A movie scene: a black-and-white photograph showing an old man sitting inside a car. He is wearing a mask and looking into the camera through the front rearview mirror. A bird-shaped ornament hangs on the left side of the rearview mirror. There is also a GPS navigation device below the center console, with the screen displaying the number '700'. Other parts of the car include some buttons and controls, creating a simple and cozy atmosphere. The background is blurred, with a faint view of the outside.

    **Output:** 
    {
        Modified prompt: "A movie scene: a black-and-white photograph showing an old man sitting inside a car. He is wearing a mask and looking into the camera through the front rearview mirror. A bird-shaped ornament hangs on the left side of the rearview mirror. There is also a GPS navigation device below the center console, with the screen displaying the number '710'. Other parts of the car include some buttons and controls, creating a simple and cozy atmosphere. The background is blurred, with a faint view of the outside.",
        Changed detail: "The screen number was changed from '700' to '710'."
    }

    **Now your turn:** """ + f"""

    **Input:** {data[i]['output']}

    **Output:**"""
    
    
    completion = get_completion_with_retry(
        client=client,
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
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
        "image_path": data[i]['image_path'],
        "prompt": data[i]['prompt_true'],
        "prompt_false": output,
    })

with open(f'complex_image_recaption_modify.json', 'w') as f:
    json.dump(final, f, indent=4, ensure_ascii=False)
