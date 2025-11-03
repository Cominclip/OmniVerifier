import json
from tqdm import tqdm
import numpy as np
import cv2
import random
from concurrent.futures import ProcessPoolExecutor, as_completed


def get_mask_ratio(image_path):
    mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    polygon_area = cv2.countNonZero(mask)
    image_area = mask.shape[0] * mask.shape[1]
    mask_ratio = polygon_area / image_area
    return mask_ratio


def process_image(image_data):

    for obj in image_data['detections']:
        obj['mask_ratio'] = get_mask_ratio(obj['mask'])

    if len(image_data['detections']) < 2:
        return None

    sorted_obj = sorted(image_data['detections'], key=lambda x: x['mask_ratio'], reverse=True)
    n = len(sorted_obj)
    easy_idx = int(n * 0.2)    
    medium_idx = int(n * 0.7)  

    r = random.random()
    if r < 0.2:
        candidates = sorted_obj[:easy_idx]
        select_obj = random.choice(candidates) if candidates else sorted_obj[0]
        difficulty = 'easy'
    elif r < 0.7:
        candidates = sorted_obj[easy_idx:medium_idx]
        select_obj = random.choice(candidates) if candidates else sorted_obj[len(sorted_obj) // 2]
        difficulty = 'medium'
    else:
        candidates = sorted_obj[medium_idx:]
        select_obj = random.choice(candidates) if candidates else sorted_obj[-1]
        difficulty = 'hard'

    image_data['selected_object'] = select_obj
    image_data['difficulty'] = difficulty
    return image_data


if __name__ == "__main__":
    with open('complex-images-synthetic/data_with_masks.json', 'r') as f:
        data = json.load(f)

    final_data = []

    with ProcessPoolExecutor(max_workers=8) as executor: 
        futures = [executor.submit(process_image, d) for d in data]

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result is not None:
                final_data.append(result)

    with open('complex-images-synthetic/data_with_selected_mask.json', 'w') as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)

