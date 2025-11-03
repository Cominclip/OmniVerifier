import os
import json
import numpy as np
import torch
from PIL import Image
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tqdm import tqdm

def save_mask_as_bw_image(mask, save_path, image_shape):

    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    
    h, w = image_shape[:2]
    mask_image = np.zeros((h, w), dtype=np.uint8)
    
    mask_image[mask > 0] = 255
    
    cv2.imwrite(save_path, mask_image)

def save_image_with_masks(image, masks, save_path, random_color=False, borders=True):
    result_image = image.copy()
    
    if isinstance(masks, np.ndarray):
        if masks.ndim == 4:  
            masks = [masks[i, 0] for i in range(masks.shape[0])]
        elif masks.ndim == 3: 
            masks = [masks[i] for i in range(masks.shape[0])]
    
    for mask in masks:
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        
        h, w = mask.shape
        mask_uint8 = (mask > 0).astype(np.uint8) * 255  
        
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        
        colored_mask = np.zeros((h, w, 4), dtype=np.float32)
        for i in range(3): 
            colored_mask[:, :, i] = (mask_uint8 / 255.0) * color[i]
        colored_mask[:, :, 3] = (mask_uint8 / 255.0) * color[3]  
        
        result_image_float = result_image.astype(np.float32) / 255.0
        alpha = colored_mask[:, :, 3:4]
        result_image_float = result_image_float * (1 - alpha) + colored_mask[:, :, :3] * alpha
        result_image = (result_image_float * 255).astype(np.uint8)
        
        if borders:
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result_image, contours, -1, (255, 255, 255), 1)
    
    cv2.imwrite(save_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))

def setup_sam2():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    sam2_checkpoint = "sam2.1-hiera-large/sam2.1_hiera_large.pt"
    model_cfg = "sam2.1-hiera-large/sam2.1_hiera_l.yaml"
    
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    
    return predictor, device

def process_single_image(predictor, image_data, output_dir):

    image_path = image_data["image"]
    image_id = image_data["id"]
    detections = image_data["detections"]
    
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return image_data
    
    try:
        image = Image.open(image_path)
        image = np.array(image.convert("RGB"))
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return image_data
    
    boxes = []
    for detection in detections:
        bbox = detection["bbox"]
        boxes.append(bbox)
    
    if not boxes:
        print(f"No detections found for image {image_id}")
        return image_data
    
    boxes = np.array(boxes)
    
    predictor.set_image(image)
    
    try:
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False
        )
        
        masks_dir = os.path.join(output_dir, "individual_masks")
        os.makedirs(masks_dir, exist_ok=True)

        for i, detection in enumerate(detections):
            detection["sub_id"] = i
            
            if i < len(masks):
                mask = masks[i][0]  
                
                mask_filename = f"image_{image_id}_bbox_{i}.png"
                mask_path = os.path.join(masks_dir, mask_filename)
                
                save_mask_as_bw_image(mask, mask_path, image.shape)
                
                detection["mask"] = mask_path
            else:
                detection["mask"] = None
        
        combined_mask_filename = f"image_{image_id}_combined_masks.png"
        combined_mask_path = os.path.join(output_dir, combined_mask_filename)
        
        os.makedirs(output_dir, exist_ok=True)
        
        save_image_with_masks(
            image,
            masks[:, 0], 
            save_path=combined_mask_path,
            random_color=True,
            borders=True
        )
        

        image_data["combined_mask_image_path"] = combined_mask_path
        print(f"Processed image {image_id}: {len(masks)} masks generated")
        
    except Exception as e:
        print(f"Error processing image {image_id}: {e}")
    
    return image_data

def main():
    json_file_path = "complex-images-synthetic/data_bboxes.json"
    output_json_path = "complex-images-synthetic/data_with_masks.json"
    output_images_dir = "complex-images-synthetic/images_with_masks"
    
    os.makedirs(output_images_dir, exist_ok=True)
    
    print("Initializing SAM2 model...")
    predictor, device = setup_sam2()
    
    print("Loading JSON data...")
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return
    
    print(f"Found {len(data)} images to process")
    
    processed_data = []
    for image_data in tqdm(data, desc="Processing images"):
        processed_image = process_single_image(predictor, image_data.copy(), output_images_dir)
        processed_data.append(processed_image)
        
        if len(processed_data) % 100 == 0:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            print(f"Progress saved: {len(processed_data)}/{len(data)} images processed")
    
    print("Saving final results...")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    print(f"Processing completed! Results saved to:")
    print(f"- JSON file: {output_json_path}")
    print(f"- Individual masks: {output_images_dir}/individual_masks/")
    print(f"- Combined mask images: {output_images_dir}/")
    print(f"Total processed: {len(processed_data)} images")

if __name__ == "__main__":
    main()