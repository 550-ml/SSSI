import json
import os
import matplotlib.pyplot as plt
import models
import numpy as np
import torch
import yaml
from tqdm import tqdm
from paddleocr import PaddleOCR
from PIL import Image
from sam2.build_sam import build_sam2
from models.sampredictor import SAMPREDICTOR
from sam2.sam2_image_predictor import SAM2ImagePredictor
from uitl import (
    ocr_remove,
    deduplicate_masks,
    generate_score_matrix,
    merge_ocr_sam_box,
    pick_best_match_per_ocr,
    dynamic_point_sampling,
    get_center_point,
    get_random_point,
    save_combined_masks,
    mask_to_coco_format,
    convex_hull_smoothing,
)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

image_folder = "../dataset/images"
save_folder = "../dataset/result"
json_save_path = "../dataset/result/annotations.json"
strategy = "center"
point_num = 1
annotations_file = []

def load_components(config, preseg_model_path, sam2_checkpoint, model_cfg):
    with open(config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model = models.make(config["model"])
    sam_checkpoint = torch.load(preseg_model_path, map_location=device)
    model.load_state_dict(sam_checkpoint, strict=True)
    model.to(device)
    input_size = config["test"]["inp_size"]
    sampredictor = SAMPREDICTOR(model, input_size=input_size)
    
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    
    ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False)
    return sampredictor, predictor, ocr

def sam2_predict(predictor, image, processed_image, best_box, save_path, idx):
    predictor.set_image(processed_image)
    predict_masks = []
    for id, box in enumerate(best_box):
        if strategy == "center":
            point = get_center_point(box, point_num=point_num)
        elif strategy == "random":
            point = get_random_point(box, point_num=point_num)
        label = [1] * point_num
        input_box = np.array(box)
        input_point = np.array(point)
        input_label = np.array(label)
        predict_mask, score = dynamic_point_sampling(
            input_box, input_point, input_label, predictor, image
        )
        if predict_mask is not None:
            predict_masks.append((predict_mask, score))
    unique_masks, unique_scores = deduplicate_masks(predict_masks)
    save_combined_masks(image, unique_masks, save_path)
    for mask, score in zip(unique_masks, unique_scores):
        annotations = mask_to_coco_format(mask, idx + 1, 1, score)
        annotations_file.extend(annotations)

def run_image(image_file, sampredictor, predictor, ocr,idx):
    image_path = os.path.join(image_folder, image_file)
    save_path = os.path.join(save_folder, image_file)
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))
    #pre_seg
    processed_image, ocr_boxes = ocr_remove(image, ocr)
    plt.imsave(save_path, image)
    sam_boxes = sampredictor.generate(image_path=save_path, save_path=save_path, is_smooth=True)
    score_mat = generate_score_matrix(
        ocr_boxes,
        sam_boxes,
        w_center=0.3,
        w_iou=0.5,
        w_bound=0.2,
        d_max=300.0,  
        b_max=50.0,  
    )
    best_results = pick_best_match_per_ocr(score_mat)
    best_box = []
    for i, (best_j, best_score) in enumerate(best_results):
        best_box.append(merge_ocr_sam_box(ocr_boxes[i], sam_boxes[best_j]))
    sam2_predict(predictor, image, processed_image, best_box, save_path,idx)
    
def run_batch(image_files, sampredictor, predictor, ocr):
    for idx, image_file in tqdm(enumerate(image_files), total=len(image_files), desc="Processing images"):
        run_image(image_file, sampredictor, predictor, ocr,idx)

def main(config, preseg_model_path, sam2_checkpoint, model_cfg):
    sampredictor, predictor, ocr = load_components(config, preseg_model_path, sam2_checkpoint, model_cfg)
    image_files = sorted(os.listdir(image_folder))
    run_batch(image_files, sampredictor, predictor, ocr)
    with open(json_save_path, "w") as f:
        json.dump(annotations_file, f, indent=4)
        
if __name__ == "__main__":
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    config = "./cod-sam-vit-l.yaml"
    preseg_model_path = "../checkpoint/preseg.pth"
    sam2_checkpoint = "../checkpoint/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    main(config, preseg_model_path, sam2_checkpoint, model_cfg)