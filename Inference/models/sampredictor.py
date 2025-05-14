import json
from re import I
import torch
import numpy as np
import cv2

from typing import Any, Dict, Generator, ItemsView, List, Tuple
from torchvision import transforms
from PIL import Image

from .sam import SAM
from .utils.amg import remove_small_regions, apply_blue_mask_with_emphasis
import matplotlib.pyplot as plt
import os


class SAMPREDICTOR:
    """在sam基础上封装一层，实现对输入图片的处理，还有对预测结果的后处理，保存，可视化"""

    def __init__(
        self,
        model: SAM,
        input_size: int = 1024,
    ) -> None:
        self.model = model
        self.img_transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ],
        )
        self.image = None
        self.mask = None
        self.image_name = None

    def pre_process_image(self, image_path: str) -> np.ndarray:
        """记录图片信息，转换图片格式"""
        self.image = Image.open(image_path).convert("RGB")
        self.image_name = os.path.basename(image_path)
        self.image_width, self.image_height = self.image.size
        image = self.img_transform(self.image)
        return image

    def generate_mask(
        self, image: np.ndarray, threshold: float = -0.0, min_area=10, is_smooth=True
    ) -> np.ndarray:
        """预测mask"""
        # 增强健壮性
        if image is Image:
            image = np.array(image)
        self.model.eval()
        with torch.no_grad():
            if image.dim() == 3:
                image = image.unsqueeze(0)
            self.model.set_image(image)
            mask = self.model.forward()  # 4维的
            # remove small regions
            mask = mask.squeeze().cpu().numpy()
            _, threshold = cv2.threshold(
                mask.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            # print("te", threshold)
            mask = (mask > threshold).astype(np.uint8)
            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            mask = mask.astype(np.uint8)
            if is_smooth:
                mask = self.convex_hull_smoothing(mask)
        self.mask = mask
        return mask

    def convex_hull_smoothing(self, mask: np.ndarray) -> np.ndarray:
        """
        对mask边缘进行凸包拟合和平滑处理
        Args:
            mask (np.ndarray): 二值化的mask
        Returns:
            np.ndarray: 经过凸包和抗锯齿处理后的mask
        """
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 创建一个空白图像用于绘制抗锯齿效果
        anti_aliasing_image = np.zeros_like(mask)

        # 遍历每个轮廓，并对每个轮廓进行凸包处理
        for contour in contours:
            hull = cv2.convexHull(contour)
            # 使用抗锯齿的方式绘制凸包
            cv2.fillConvexPoly(anti_aliasing_image, hull, 255, lineType=cv2.LINE_AA)

        # 去除过渡像素（将小于255但不等于0的像素设为0）
        for i in range(anti_aliasing_image.shape[0]):
            for j in range(anti_aliasing_image.shape[1]):
                if anti_aliasing_image[i, j] != 255:
                    anti_aliasing_image[i, j] = 0

        return anti_aliasing_image

    def post_process_mask(self) -> np.ndarray:
        """后处理mask，恢复成原来的大小"""
        self.mask = cv2.resize(
            self.mask,
            (self.image_width, self.image_height),
            interpolation=cv2.INTER_NEAREST,
        )
        return self.mask

    def mask_save(self, save_path: str) -> None:
        """保存mask"""
        boxes_list = self.get_boxes_list(save_dir=save_path)
        return boxes_list

    def generate(
        self,
        image_path: str,
        threshold: float = -0.0,
        min_area=500,
        is_smooth=True,
        save_path: str = None,
    ) -> np.ndarray:
        """生成mask"""
        image = self.pre_process_image(image_path)
        self.generate_mask(image, threshold, min_area, is_smooth=is_smooth)
        self.post_process_mask()
        if save_path:
            boxes_list = self.mask_save(save_path)
        return boxes_list

    def save_individual_masks(self, save_dir: str):
        """将每个独立的mask区域按照其实际形状保存为单独的图像
        Args:
            image_path (str): 原始图像的路径
            save_dir (str): 保存每个独立mask的目录路径
        """
        # 找到独立的轮廓（即每个独立的mask区域）
        contours, _ = cv2.findContours(
            self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        original_image = np.array(self.image)
        save_dir = os.path.join(save_dir, self.image_name[:-4])
        # 确保保存目录存在
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # 遍历每个轮廓，提取并保存每个独立区域
        for i, contour in enumerate(contours):
            # 创建一个与原图大小相同的空白掩码
            mask = np.zeros_like(original_image)

            # 将轮廓绘制到掩码上
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

            # 提取与轮廓对应的区域
            region = cv2.bitwise_and(original_image, mask)

            # 获取轮廓的边界框 (x, y, w, h) 用于裁剪区域
            x, y, w, h = cv2.boundingRect(contour)

            # 裁剪出感兴趣的区域 (ROI)
            roi = region[y : y + h, x : x + w]

            # 保存该区域为单独的图像
            save_path = os.path.join(save_dir, f"region_{i+1}.png")
            cv2.imwrite(save_path, roi)
            print(f"Saved region {i+1} to {save_path}")

    def save_individual_masks_with_overlay(self, save_dir: str):
        """在原始图像上为每个掩码单独叠加蒙版、加粗边缘并保存。

        Args:
            save_dir (str): 保存处理后图像的目录。
        """
        # 获取原始图像和掩码
        original_image = np.array(self.image)
        mask = self.mask
        save_dir = os.path.join(save_dir, self.image_name[:-4])

        # 确保保存目录存在
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 查找掩码的轮廓（每个轮廓代表一个单独的掩码区域）
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 创建一个副本，用于叠加每个掩码的蓝色蒙版和边框效果
        final_image = original_image.copy()

        # 遍历每个轮廓，对每个掩码区域进行处理
        for i, contour in enumerate(contours):
            # 创建与原始图像相同大小的空白掩码
            single_mask = np.zeros_like(mask)

            # 在掩码上绘制当前轮廓
            cv2.drawContours(single_mask, [contour], -1, (255), thickness=cv2.FILLED)

            # 应用蓝色蒙版和边缘强调
            image_with_overlay = apply_blue_mask_with_emphasis(
                original_image, single_mask
            )

            # 将结果叠加到最终的图像上
            final_image[single_mask == 255] = image_with_overlay[single_mask == 255]

            # 保存该掩码单独应用的图像
            save_path = os.path.join(save_dir, f"overlay_image_with_mask_{i+1}.png")
            cv2.imwrite(save_path, image_with_overlay)
            print(f"Saved overlay image {i+1} to {save_path}")

        # 保存最终合并后的图像
        final_save_path = os.path.join(save_dir, "final_image_with_all_masks.png")
        cv2.imwrite(final_save_path, final_image)
        print(f"Saved final image with all masks to {final_save_path}")

    def save_individual_masks_with_json(self, save_dir: str) -> None:
        contours, _ = cv2.findContours(
            self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        original_image = np.array(self.image)
        save_dir = os.path.join(save_dir, self.image_name[:-4])
        os.makedirs(save_dir, exist_ok=True)

        masks_data = []  # 存储所有掩码区域的边界框信息

        for i, contour in enumerate(contours):
            mask = np.zeros_like(original_image)
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
            region = cv2.bitwise_and(original_image, mask)
            x, y, w, h = cv2.boundingRect(contour)
            roi = region[y : y + h, x : x + w]
            save_path = os.path.join(save_dir, f"region_{i+1}.png")
            cv2.imwrite(save_path, roi)

            # 获取顶点坐标
            top_left = [float(x), float(y)]
            top_right = [float(x + w), float(y)]
            bottom_right = [float(x + w), float(y + h)]
            bottom_left = [float(x), float(y + h)]

            box = [x, y, x + w, y + h]
            # 构建单个掩码区域的数据格式
            masks_data.append(
                {
                    "text": f"Region {i + 1}",
                    "coordinates": {
                        "top_left": top_left,
                        "top_right": top_right,
                        "bottom_right": bottom_right,
                        "bottom_left": bottom_left,
                    },
                    "box": box,
                    "mask_path": save_path,
                }
            )

        json_path = os.path.join(save_dir, "masks_data.json")
        with open(json_path, "w") as f:
            json.dump({"merged_boxes": masks_data}, f, indent=4)
        print(f"Saved JSON metadata to {json_path}")

    def get_boxes_list(self, save_dir: str) -> None:
        contours, _ = cv2.findContours(
            self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        original_image = np.array(self.image)
        boxes_list = []
        for i, contour in enumerate(contours):
            mask = np.zeros_like(original_image)
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
            x, y, w, h = cv2.boundingRect(contour)
            box = [x, y, x + w, y + h]
            # 构建单个掩码区域的数据格式
            # print(box)
            boxes_list.append(box)
        return boxes_list
