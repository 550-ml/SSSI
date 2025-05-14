import math
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment
from sympy import N

def ocr_remove(image, ocr, remove_text=True, y_gap_threshold=10, x_gap_threshold=10):
    # 10原来的
    # 30 50 100
    "detect_text and remove"
    result = ocr.ocr(image)
    image_copy = image.copy()
    ocr_boxes = []

    for idx, line in enumerate(result):
        for element in line:
            box = element[0]  # 坐标 (x1, y1), (x2, y2), (x3, y3), (x4, y4)
            x_min = int(min([point[0] for point in box]))
            y_min = int(min([point[1] for point in box]))
            x_max = int(max([point[0] for point in box]))
            y_max = int(max([point[1] for point in box]))

            ocr_boxes.append([x_min, y_min, x_max, y_max])

            if remove_text:
                # 使用平均背景色替代文字区域
                background_color = (
                    image_copy[max(0, y_min - 1) : y_min, x_min:x_max]
                    .mean(axis=(0, 1))
                    .astype(int)
                )
                image_copy[y_min:y_max, x_min:x_max] = background_color

    # 合并边界框
    merged_bboxes = merge_boxes(ocr_boxes, y_gap_threshold, x_gap_threshold)
    return image_copy, merged_bboxes


def merge_boxes(bboxes, y_gap_threshold, x_gap_threshold):
    """
    merge_text_boxes
    """
    merged = []
    used = set()

    for i, box1 in enumerate(bboxes):
        if i in used:
            continue
        # 初始化当前合并的区域
        x1_min, y1_min, x1_max, y1_max = box1
        for j, box2 in enumerate(bboxes):
            if i == j or j in used:
                continue
            x2_min, y2_min, x2_max, y2_max = box2

            # 计算水平和垂直间距
            x_gap = max(0, max(x2_min - x1_max, x1_min - x2_max))
            y_gap = max(0, max(y2_min - y1_max, y1_min - y2_max))

            # 判断是否满足合并条件
            if x_gap <= x_gap_threshold and y_gap <= y_gap_threshold:
                # 合并两个框
                x1_min = min(x1_min, x2_min)
                y1_min = min(y1_min, y2_min)
                x1_max = max(x1_max, x2_max)
                y1_max = max(y1_max, y2_max)
                used.add(j)

        # 保存合并后的框
        merged.append([x1_min, y1_min, x1_max, y1_max])
        used.add(i)
    return merged

def generate_score_matrix(
    ocr_bboxes,
    sam_bboxes,
    w_center=0.3,
    w_iou=0.5,
    w_bound=0.2,
    d_max=1000.0,
    b_max=200.0,
):
    """
    生成 OCR bbox 与 SAM bbox 的评分矩阵 (n x m)，其中
    n = len(ocr_bboxes), m = len(sam_bboxes)
    """
    n = len(ocr_bboxes)
    m = len(sam_bboxes)
    score_matrix = np.zeros((n, m), dtype=np.float32)

    for i in range(n):
        for j in range(m):
            score_matrix[i, j] = compute_score(
                ocr_bboxes[i], sam_bboxes[j], w_center, w_iou, w_bound, d_max, b_max
            )
    return score_matrix

def compute_score(
    bbox1, bbox2, w_center=0.3, w_iou=0.5, w_bound=0.2, d_max=1000.0, b_max=200.0
):
    """
    给定两个 bbox, 计算它们的综合匹配得分。
    - bbox1: OCR 的 bbox
    - bbox2: SAM 的 bbox
    - w_center, w_iou, w_bound: 三个指标的权重 (可调)
    - d_max: 用于归一化中心点距离, 具体取值可根据图像尺寸而定
    - b_max: 用于归一化 boundary distance

    示例评分公式：
        score = w_center * (1 - centerDist/d_max)
               + w_iou    * iouVal
               + w_bound  * (1 - boundaryDist/b_max)
    你可根据实际情况调整或扩展
    """
    dist_cent = center_distance(bbox1, bbox2)
    iou_val = iou(bbox1, bbox2)
    dist_bound = boundary_distance(bbox1, bbox2)

    # 避免超过最大值时出现负分
    center_score = max(0.0, 1.0 - dist_cent / d_max)
    bound_score = max(0.0, 1.0 - dist_bound / b_max)

    score = w_center * center_score + w_iou * iou_val + w_bound * bound_score
    return score

def center_distance(bbox1, bbox2):
    """
    计算两个 bbox 的中心点欧几里得距离
    bbox 格式: (x_min, y_min, x_max, y_max)
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    cx1 = (x1_min + x1_max) / 2.0
    cy1 = (y1_min + y1_max) / 2.0
    cx2 = (x2_min + x2_max) / 2.0
    cy2 = (y2_min + y2_max) / 2.0

    return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


def iou(bbox1, bbox2):
    """
    计算两个 bbox 的 IoU (Intersection over Union)
    bbox 格式: (x_min, y_min, x_max, y_max)
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # 交集部分坐标
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0.0  # 没有重叠
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)

    # 各自面积
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    # 并集面积
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def boundary_distance(bbox1, bbox2):
    """
    计算 bbox1 的四个顶点到 bbox2 的“最小外部距离”：
    - 若顶点在 bbox2 内部，则该顶点距离为 0
    - 若顶点在 bbox2 外部，计算该顶点到 bbox2 边界矩形的最短距离
    然后取 4 个顶点距离的最小值（或平均值也行，这里示例取最小值）
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    corners = [(x1_min, y1_min), (x1_min, y1_max), (x1_max, y1_min), (x1_max, y1_max)]

    def corner_dist_to_box(cx, cy, bxmin, bymin, bxmax, bymax):
        # 如果在 box 内部，距离记为 0
        if (bxmin <= cx <= bxmax) and (bymin <= cy <= bymax):
            return 0.0
        # 在外部时，计算到边界的最短距离
        dx = 0
        dy = 0
        if cx < bxmin:
            dx = bxmin - cx
        elif cx > bxmax:
            dx = cx - bxmax

        if cy < bymin:
            dy = bymin - cy
        elif cy > bymax:
            dy = cy - bymax

        return math.sqrt(dx * dx + dy * dy)

    dist_list = []
    for cx, cy in corners:
        dist_list.append(corner_dist_to_box(cx, cy, x2_min, y2_min, x2_max, y2_max))

    return min(dist_list)  


def pick_best_match_per_ocr(score_matrix):
    """
    给定评分矩阵 (n x m)，对每个 OCR (行) 找到得分最高的 SAM (列)。
    返回一个列表 best_match，其中 best_match[i] = (best_j, best_score)
    表示第 i 个 OCR bbox 匹配到第 best_j 个 SAM bbox，得分为 best_score。
    """
    best_match = []
    n, m = score_matrix.shape
    for i in range(n):
        row_scores = score_matrix[i, :]
        best_j = np.argmax(row_scores)
        best_score = row_scores[best_j]
        best_match.append((best_j, best_score))
    return best_match


def pick_best_match_per_ocr_linear(score_matrix):
    """
    使用匈牙利算法进行最优匹配。

    给定评分矩阵 (n x m)，对每个 OCR (行) 找到与之匹配的最佳 SAM (列)。
    匹配的目标是最大化总得分，即最优的对齐。

    返回一个列表 best_match，其中 best_match[i] = (best_j, best_score)
    表示第 i 个 OCR bbox 匹配到第 best_j 个 SAM bbox，得分为 best_score。
    """
    # 由于匈牙利算法是最小化问题，如果是最大化问题，我们需要对得分进行反向处理
    cost_matrix = -score_matrix  # 反向得分矩阵

    # 使用匈牙利算法进行最优匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 存储最优匹配
    best_match = []
    for i, j in zip(row_ind, col_ind):
        best_score = score_matrix[i, j]
        best_match.append((j, best_score))  # 保存 SAM bbox 索引和得分

    return best_match

def merge_ocr_sam_box(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    new_xmin, new_ymin = min(x1_min, x2_min), min(y1_min, y2_min)
    new_xmax, new_ymax = max(x1_max, x2_max), max(y1_max, y2_max)
    return new_xmin, new_ymin, new_xmax, new_ymax

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            import cv2

            contours, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            # Try to smooth contours
            contours = [
                cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
                for contour in contours
            ]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)


def display_boxes_on_image(
    image_path, boxes, box_color=(255, 0, 0), thickness=2, is_xywh=True
):
    """
    Display bounding boxes on an image.

    Args:
        image_path (str): Path to the input image.
        boxes (list[list[float]]): List of bounding boxes, each in format [x, y, w, h].
        box_color (tuple): Color of the bounding box in BGR format (default is red).
        thickness (int): Thickness of the bounding box lines (default is 2).

    Returns:
        None: Displays the image with bounding boxes.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found at the specified path: {}".format(image_path))

    # Ensure box coordinates are integers
    boxes = [[int(coord) for coord in box] for box in boxes]

    # Draw each bounding box on the image
    for box in boxes:
        if is_xywh:
            x, y, w, h = box
            top_left = (x, y)
            bottom_right = (x + w, y + h)
        else:
            x, y, w, h = box
            top_left = (x, y)
            bottom_right = (w, h)
        cv2.rectangle(image, top_left, bottom_right, box_color, thickness)

    # Convert the image to RGB (from BGR) for displaying with matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.figure(figsize=(4, 4))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.show()





def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [
            cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours
        ]
        mask_image = cv2.drawContours(
            mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2
        )
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def show_masks(
    image,
    masks,
    scores,
    point_coords=None,
    box_coords=None,
    input_labels=None,
    borders=True,
):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(4, 4))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis("off")
        plt.show()


def dynamic_point_sampling(input_box, input_point, input_label, predictor, image):
    # 第一轮推理
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
        box=input_box,
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]

    # 合成初始mask输入
    mask_input = logits[np.argmax(scores), :, :]  # 选择模型的最佳mask
    x_min, y_min, x_max, y_max = input_box
    # 进行后续推理，生成新的mask
    for i in range(5):
        if i == 0:
            # 第一轮，使用bbox、点和mask输入进行推理
            # print("第一轮")
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                mask_input=mask_input[None, :, :],
                box=input_box,
                multimask_output=True,
            )
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]
            threshold = 0.9
            # 如果置信度超过0.9，直接结束
            if scores[0] > threshold:
                return masks[0], scores[0]
        else:
            # 后续轮次，随机采样两个点
            # print("第二轮")
            point_list = []
            for _ in range(2):
                random_x = random.uniform(x_min, x_max)
                random_y = random.uniform(y_min, y_max)
                point_list.append([random_x, random_y])
            label = [1] * 2
            input_point = np.array(point_list)
            input_label = np.array(label)
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                mask_input=mask_input[None, :, :],
                box=input_box,
                multimask_output=True,
            )
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]
            threshold = 0.9
            # 如果置信度超过0.9，直接结束
            if scores[0] > threshold:
                return masks[0], scores[0]
    # 如果没有提前返回，选择分数最高的mask
    return masks[0], scores[0]


def one_box_predict(input_box, input_point, input_label, predictor, image):
    # first inference
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_box,
        multimask_output=True,
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]

    mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        # box=input_box,
        mask_input=mask_input[None, :, :],
        multimask_output=True,
    )
    # masks = masks[sorted_ind]
    # scores = scores[sorted_ind]
    # threshold = 0.9
    # high_score_masks = masks[scores > threshold]
    # if len(high_score_masks) > 0:
    #     sum_mask = np.logical_or.reduce(high_score_masks).astype(np.uint8)
    #     score = scores[0]  # 对应的score
    # else:
    #     # 否则，选择分数最高的那个mask
    #     sum_mask = masks[0]
    #     score = scores[0]
    sum_mask = masks[0]
    score = scores[0]
    # show_masks(image, [sum_mask], scores=[score])
    return sum_mask, score


def calculate_iou(mask1, mask2):
    """
    计算两个二值掩码的 IoU (Intersection over Union)
    :param mask1: 第一个二值掩码
    :param mask2: 第二个二值掩码
    :return: IoU 值
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0


def deduplicate_masks(predict_masks, iou_threshold=0.5):
    """
    根据 IoU 去重掩码
    :param predict_masks: 一个包含(掩码, 分数)元组的列表
    :param iou_threshold: IoU 阈值，大于此值的掩码会被认为是重复的
    :return: 去重后的掩码和分数列表
    """
    unique_masks = []
    unique_scores = []

    for mask, score in predict_masks:
        is_duplicate = False

        # 比较当前掩码与已有的掩码
        for existing_mask in unique_masks:
            iou = calculate_iou(mask, existing_mask)
            if iou > iou_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_masks.append(mask.astype(np.uint8))
            unique_scores.append(score)

    return unique_masks, unique_scores


def binary_mask_to_rle(binary_mask):
    """
    将二值 mask 转换为 RLE 编码
    """
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")  # 需要将 counts 转为字符串
    return rle


def mask_to_coco_format(mask, image_id, category_id, score=1.0):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotations = []
    for contour in contours:
        if len(contour) >= 3:
            segmentation = contour.flatten().tolist()
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if area < 500:
                continue
            annotations.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": [segmentation],
                    "bbox": [x, y, w, h],
                    "area": float(area),
                    "score": float(score),
                }
            )
    return annotations


def get_center_point(box, point_num=1):
    """
    计算边界框的中心点。

    参数:
        box (list): 边界框坐标 [x_min, y_min, x_max, y_max]。

    返回:
        list: 中心点坐标 [x, y]。
    """
    x_min, y_min, x_max, y_max = box
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    point_list = []
    point_list.append([center_x, center_y])
    for _ in range(point_num - 1):
        random_x = random.uniform(x_min, x_max)
        random_y = random.uniform(y_min, y_max)
        point_list.append([random_x, random_y])
    return point_list

def get_random_point(box, point_num):
    """
    在边界框内随机生成一个点。

    参数:
        box (list): 边界框坐标 [x_min, y_min, x_max, y_max]。

    返回:
        list: 随机点坐标 [x, y]。
    """
    x_min, y_min, x_max, y_max = box
    point_list = []
    for _ in range(point_num):
        random_x = random.uniform(x_min, x_max)
        random_y = random.uniform(y_min, y_max)
        point_list.append([random_x, random_y])
    return point_list

def save_combined_masks(
    image, unique_masks, save_path, random_color=True, borders=True
):
    """
    将所有掩码以叠放方式添加到灰度底图上并保存，掩码保持彩色
    :param image: 输入图像 (numpy array)
    :param unique_masks: 去重后的掩码列表，每个掩码是一个二值图
    :param save_path: 保存叠加结果图像的路径
    :param random_color: 是否为掩码分配随机颜色（否则使用固定颜色）
    :param borders: 是否绘制掩码边界
    """
    # 将原始图像转换为灰度图，并扩展为3通道以支持彩色掩码
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_image_3ch = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

    # 复制灰度图像以避免修改原图
    result = gray_image_3ch.copy()

    # 获取图像尺寸
    h, w, _ = image.shape

    # 为每个掩码分配颜色并以叠放方式添加到结果图像
    for i, mask in enumerate(unique_masks):
        # 确保掩码尺寸与图像一致
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # 转换为二值掩码（0 或 1）
        mask = mask.astype(np.uint8)

        # 选择颜色（随机或固定）
        if random_color:
            color = np.concatenate(
                [np.random.randint(100, 255, size=3), np.array([1.0])], axis=0
            )  # 高对比度随机颜色
        else:
            color = np.array([30, 144, 255, 255])  # 固定蓝色

        # 将掩码区域着色（叠放效果：底图灰度信息与彩色掩码混合）
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        # 在掩码区域，保留底图灰度信息的 30%，添加掩码颜色的 70%
        result[mask > 0] = result[mask > 0] * 0.3 + mask_image[mask > 0, :3] * 0.7

        # 如果需要绘制边界
        if borders:
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            contours = [
                cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
                for contour in contours
            ]
            result = cv2.drawContours(
                result, contours, -1, (255, 255, 255), thickness=2
            )

    # 保存叠加后的结果图像
    cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    
def convex_hull_smoothing(mask: np.ndarray) -> np.ndarray:
    """
    对mask边缘进行凸包拟合和平滑处理
    Args:
        mask (np.ndarray): 二值化的mask
    Returns:
        np.ndarray: 经过凸包和抗锯齿处理后的mask
    """

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    anti_aliasing_image = np.zeros_like(mask)
    for contour in contours:
        hull = cv2.convexHull(contour)
        cv2.fillConvexPoly(anti_aliasing_image, hull, 255, lineType=cv2.LINE_AA)
    for i in range(anti_aliasing_image.shape[0]):
        for j in range(anti_aliasing_image.shape[1]):
            if anti_aliasing_image[i, j] != 255:
                anti_aliasing_image[i, j] = 0

    return anti_aliasing_image
