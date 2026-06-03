import torch
from torch import nn
import math
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import copy
import os
from pathlib import Path


def get_gaussian_kernel(kernel_size=3, sigma=0.6, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, stride=1, padding=1, groups=channels, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


def f(point):
    key_point = np.zeros((256, 256), np.uint8)
    cv2.fillPoly(key_point, point.reshape(1, 6, 2), color=255)

    TP = np.sum((key_point > 0) * (label > 0))
    FP = np.sum((key_point > 0) * (1 - (label > 0)))
    FN = np.sum((1 - (key_point > 0)) * (label > 0))
    TN = np.sum((1 - (key_point > 0)) * (1 - (label > 0)))

    iou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
    return iou


def Get_edge(img):
    """
    使用 OpenCV 膨胀/腐蚀逻辑获取边界
    """
    img_uint8 = img.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(img_uint8, kernel, iterations=1)
    edge = img_uint8 - erosion
    return edge


def Get_Edge_position(img):
    """
    直接获取所有非零点的坐标
    """
    coords = np.where(img > 0)
    return coords[0], coords[1]


def process_single_image(image_file_path, output_dir, output_idx):
    """
    处理单张掩码图片，生成边界点并保存
    output_idx: 输出文件名编号 (0.png, 1.png, ...)
    """
    file_name = os.path.basename(image_file_path)
    print(f"\nProcessing [{output_idx}]: {file_name}")

    # 读取掩码
    raw_img = np.array(Image.open(image_file_path).convert('L'))
    img_bool = (raw_img > 127)

    label = np.zeros_like(raw_img, dtype=np.uint8)
    label[img_bool] = 255

    # 全局变量 label 供 f(point) 使用
    globals()['label'] = label

    # 提取边缘
    edge_img = Get_edge(img_bool.astype('uint8'))
    edgex, edgey = Get_Edge_position(edge_img)

    print(f"  Found {len(edgex)} edge pixels.")

    label_ori = label.copy()
    contours, _ = cv2.findContours(label_ori, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"  Warning: No contours found in {file_name}!")
        return False
    contours = contours[0].squeeze(1)

    # 遗传算法配置
    N = 300
    cross_rate = 0.1
    variation_rate = 0.05
    best_val = 0
    best_point = np.array([])
    lst = []

    # 初始种群
    for t in range(2000):
        index = np.random.randint(0, len(contours), size=6)
        index = sorted(index)
        point = np.array([contours[i] for i in index])
        iou = f(point)
        if iou > best_val:
            best_val = iou
            best_point = index
        lst.append((index, iou))

    print(f"  Initial Best IoU: {best_val:.4f}")

    lst = sorted(lst, key=lambda val: -val[1])
    animals = [lst[i][0] for i in range(N)]

    def get_fitness(animals):
        fitness = []
        for i in range(len(animals)):
            animals[i] = sorted(animals[i])
        for animal in animals:
            arr = np.array([contours[i] for i in animal])
            fitness.append(f(arr))
        return np.array(fitness)

    def select_animal(animals, fitness):
        idx = np.random.choice(np.arange(N), size=N, replace=True,
                               p=(fitness / (np.sum(fitness) + 1e-8)))
        return [animals[id] for id in idx]

    def variation(child, variation_rate):
        child = sorted(child)
        new_child = []
        for i in range(len(child)):
            if np.random.rand() < variation_rate:
                if i == 0:
                    new_val = np.random.randint(0, child[1] + 1) if np.random.rand() < 0.5 else np.random.randint(
                        child[5], len(contours))
                elif i == 5:
                    new_val = np.random.randint(0, child[0] + 1) if np.random.rand() < 0.5 else np.random.randint(
                        child[4], len(contours))
                else:
                    new_val = np.random.randint(child[i - 1], child[i + 1] + 1)
                new_child.append(new_val)
            else:
                new_child.append(child[i])
        return new_child

    def crossover_and_variation(animals, cross_rate, is_randomise=True):
        new_animals = []
        for father in animals:
            child = list(father)
            if np.random.rand() < cross_rate:
                mother = animals[np.random.randint(N)]
                if is_randomise:
                    cross_points = np.random.randint(low=0, high=2, size=6)
                    for idx in range(6):
                        if cross_points[idx] == 1:
                            child[idx] = mother[idx]
            child = variation(child, variation_rate)
            new_animals.append(child)
        return new_animals

    def evaluation(animals):
        fitness = get_fitness(animals)
        idx = np.argmax(fitness)
        return fitness[idx], animals[idx]

    # 进化循环
    for t in range(100):
        fitness = get_fitness(animals)
        if np.sum(fitness) == 0:
            print(f"  Warning: All animals have zero fitness.")
            break
        selected_animals = select_animal(animals, fitness)
        animals = crossover_and_variation(selected_animals, cross_rate)
        iou, point = evaluation(animals)
        if iou > best_val:
            best_val = iou
            best_point = point
            print(f"  Gen {t}: New Best IoU = {best_val:.4f}")

    print(f"  Final Optimized IoU: {best_val:.4f}")

    # 生成结果图
    key_point2 = np.zeros((256, 256), np.uint8)

    # 绘制边缘
    if len(edgex) > 0:
        for i in range(len(edgex)):
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    r, c = edgex[i] + dx, edgey[i] + dy
                    if 0 <= r < 256 and 0 <= c < 256:
                        key_point2[r, c] = label_ori[r, c]

    # 绘制优化后的点
    if len(best_point) > 0:
        for idx in best_point:
            x, y = contours[idx]
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    r, c = y + dy, x + dx
                    if 0 <= r < 256 and 0 <= c < 256:
                        key_point2[r, c] = label_ori[r, c]

    # 如果结果全黑，回退到简单轮廓
    if np.sum(key_point2) == 0:
        print(f"  Warning: Result is black, using fallback.")
        cv2.drawContours(key_point2, [contours.astype(int)], -1, 255, 1)

    # 保存为编号命名
    save_path = os.path.join(output_dir, f"{output_idx}.png")
    cv2.imwrite(save_path, key_point2)
    print(f"  Saved: {save_path}")
    return True


if __name__ == '__main__':
    # 输入输出目录
    input_dir = './data/PH2/train/masks'
    output_dir = './data/PH2/train/points_boundary2'

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有掩码图片并排序
    mask_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    ])

    print(f"Found {len(mask_files)} mask images in {input_dir}")

    # 批量处理，编号从 0 开始
    success_count = 0
    fail_count = 0

    for i, mask_file in enumerate(mask_files):
        mask_path = os.path.join(input_dir, mask_file)

        try:
            result = process_single_image(mask_path, output_dir, i)
            if result:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"  Error processing {mask_file}: {str(e)}")
            fail_count += 1

    print(f"\n{'=' * 50}")
    print(f"Batch processing complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Total: {len(mask_files)}")
    print(f"Output directory: {output_dir}")
    print(f"Files named: 0.png, 1.png, 2.png, ..., {len(mask_files) - 1}.png")