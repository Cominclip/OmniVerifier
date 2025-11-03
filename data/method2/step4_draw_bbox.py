import json
import os
from PIL import Image, ImageDraw
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 读入数据
with open('complex-images-synthetic/data_with_selected_mask.json', 'r') as f:
    data = json.load(f)


def process_item(item):
    try:
        image_path = item['image']
        bbox = item['selected_object']['bbox']  
        x1, y1, x2, y2 = map(int, bbox)

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=6)

        save_path = f"complex-images-synthetic/images_with_selected_bbox/{item['id']}.png"
        image.save(save_path)
        return save_path
    except Exception as e:
        return f"Error processing {item['id']}: {e}"

num_workers = os.cpu_count() or 8
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    futures = [executor.submit(process_item, item) for item in data]

    for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
        pass