from cgitb import small
import numpy as np
from scipy import stats
import torch
import cv2
import math
from copy import deepcopy
from itertools import product
from typing import Any, Dict, Generator, ItemsView, List, Tuple


def remove_small_regions(
    mask: np.ndarray, area_thresh: float, mode: str
) -> Tuple[np.ndarray, bool]:
    """remove the samll region and hole in she mask, return the new mask and the flag indicating whether the mask is

    Args:
        mask (np.ndarray): 二值化的mask
        area_thresh (float): 面积阈值，小于这个将会被移除
        mode (str): 是否被修改

    Returns:
        Tuple[np.ndarray, bool]: _description_
    """
    import cv2

    assert mode in ["holes", "islands"]
    correct_holes = mode == "holes"
    # mask将被取反
    working_mask = (correct_holes ^ mask).astype(np.uint8)

    # 找到所有的连通区域
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(
        working_mask, connectivity=8
    )  # 多少个黑色连通域； 属于哪个连通域； 连通域的统计信息； 每个连通域的像素数
    sizes = stats[:, -1][1:]
    small_rigions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_rigions) == 0:
        return mask, False
    fill_labels = [0] + small_rigions  # 设置要填充的标签,把背景加上
    if not correct_holes:  # 如果不是处理孔洞，则保留所有大区域
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # 如果所有区域都小于阈值，保留最大的区域
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True

def apply_blue_mask_with_emphasis(image, mask, thickness=10, transparency=0.4, darken_factor=0.3):
    """
    为掩码区域应用蓝色蒙版，并在边缘使用更深的蓝色，同时极度暗化背景区域，效果更为夸张。
    
    Args:
        image (numpy.ndarray): 原始图像。
        mask (numpy.ndarray): 二值化的掩码。
        thickness (int): 边框厚度，用于强调轮廓。
        transparency (float): 蓝色蒙版的透明度。
        darken_factor (float): 暗化背景的因子，值越低背景越暗。
    
    Returns:
        numpy.ndarray: 应用了夸张蒙版和深蓝色边缘处理的图像。
    """
    # 确保掩码是二值化的单通道图像
    mask = mask.astype(np.uint8)
    # print(mask)
    # 创建原图的副本以避免修改原图
    masked_image = image.copy()
    
    # 浅蓝色和深蓝色的BGR值
    blue_color = np.array([255, 144, 30], dtype=np.uint8)   # 浅蓝色，BGR顺序
    deep_blue_color = np.array([139, 0, 0], dtype=np.uint8)  # 深蓝色，BGR顺序

    # 应用浅蓝色蒙版到掩码区域
    if np.any(mask == 255):
        overlay = masked_image.copy()  # 创建图像的副本用于叠加
        overlay[mask == 255] = blue_color  # 将浅蓝色应用到掩码区域
        cv2.addWeighted(overlay, transparency, masked_image, 1 - transparency, 0, masked_image)  # 透明度叠加

    # 查找并绘制深蓝色边缘
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(masked_image, contours, -1, deep_blue_color.tolist(), thickness=thickness, lineType=cv2.LINE_AA)

    # 将掩码区域外的背景极度暗化
    background_mask = mask == 0  # 背景区域为掩码为0的部分
    masked_image[background_mask] = (masked_image[background_mask] * darken_factor).astype(np.uint8)
    
    return masked_image
