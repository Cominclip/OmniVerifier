
import os
import torch
import json
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO
import multiprocessing as mp

image_folder = 'complex-images-synthetic/images'
save_bbox_dir = "complex-images-synthetic/images_with_bboxes"
os.makedirs(save_bbox_dir, exist_ok=True)

def process_images(rank, image_names):

    model = YOLO('Yolov11/yolo11x.pt')  
    model.to(f"cuda:{rank}")

    results_list = []
    for image_name in tqdm(image_names, position=rank):
        id = int(image_name.split('.')[0])
        image_path = os.path.join(image_folder, image_name)
        results = model.predict(image_path, iou=0.8, device=rank)

        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        masks = r.masks.data.cpu().numpy() if r.masks is not None else None
        classes = r.boxes.cls.cpu().numpy()
        confidences = r.boxes.conf.cpu().numpy()
        names = [r.names[int(c)] for c in classes]

        detections = []
        for i in range(len(classes)):
            detections.append({
                "class_id": int(classes[i]),
                "class_name": names[i],
                "confidence": float(confidences[i]),
                "bbox": boxes[i].tolist(),
                "mask": masks[i].astype(int).tolist() if masks is not None else None
            })

        results_list.append({
            "id": id,
            "image": image_path,
            "detections": detections
        })

        im_plot = r.plot()
        im_pil = Image.fromarray(im_plot[..., ::-1])
        im_pil.save(os.path.join(save_bbox_dir, f"{id}_bbox.jpg"))

    return results_list


import multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method("spawn") 

    all_images = sorted(os.listdir(image_folder), key=lambda x: int(x.split('.')[0]))

    num_gpus = 4
    chunk_size = len(all_images) // num_gpus
    chunks = [all_images[i*chunk_size:(i+1)*chunk_size] for i in range(num_gpus)]

    with mp.Pool(processes=num_gpus) as pool:
        all_results = pool.starmap(process_images, [(i, chunks[i]) for i in range(num_gpus)])