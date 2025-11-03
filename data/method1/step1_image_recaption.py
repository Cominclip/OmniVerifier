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
            wait_time = 2 + random.random()
            time.sleep(wait_time)

with open('complex_images.json', 'r') as f:
    data = json.load(f)

final = []
for i in tqdm(range(len(data))):
    prompt = f"""You are a powerful Image Captioner.

                Given an image, you must generate a caption that is **accurate and faithful** to the visual content.

                You must strictly follow these guidelines:

                1.	Accuracy First
                •	Describe only the elements that can be clearly identified in the image (objects, attributes, spatial relationships, scenes, etc.).
                •	No guesses, subjective assumptions, or irrelevant details are allowed.
                •	Do not fabricate elements that are not present in the image.
                2.	Description Elements
                •	Focus on object attributes (color, texture, shape, material).
                •	Include visible scene elements (background, environment, lighting).
                •	Cover actions, poses, expressions, and spatial relationships.
                •	Accurately capture any numbers, letters, or quantities that are discernible.
                3.	Language Style
                •	Use an objective and neutral tone.
                •	Avoid subjective evaluations (e.g., “beautiful,” “cute”).
                •	Keep the caption clear and precise, without unnecessary embellishment.
                •	The length of the caption should be determined by the complexity of the image, but it must not exceed 60 words.

                Output Format Requirement:
                Output only one caption, without any explanation or additional text."""
    
    image_path = data[i]['image_path']

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
        "id": i,
        "image_path": image_path,
        "prompt_true": output,
    })

with open(f'complex_image_recaption.json', 'w') as f:
    json.dump(final, f, indent=4, ensure_ascii=False)
