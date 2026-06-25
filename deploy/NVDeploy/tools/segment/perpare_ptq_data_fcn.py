
import cv2
import numpy as np
import argparse
import pathlib
import os
import tqdm

def main(args):
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # FCN-ResNet50-12 归一化参数
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    target_size = (520, 520)  # (width, height)

    # 获取所有图片文件
    img_files = list(pathlib.Path(args.img_dir).glob('*'))

    for img_file in tqdm.tqdm(img_files):
        img_path = str(img_file)
        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            continue
        # 转为RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 调整尺寸
        img = cv2.resize(img, target_size)
        # 转为float32并归一化到[0,1]
        img = img.astype(np.float32) / 255.0
        # HWC->CHW
        img = img.transpose(2, 0, 1)
        # 归一化 (img - mean) / std
        img = (img - mean) / std
        # 保存npy，命名为原文件名去扩展名
        save_path = os.path.join(save_dir, img_file.stem + '.npy')
        np.save(save_path, img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='将图片批量转换为FCN输入格式的npy文件')
    parser.add_argument('--img_dir', type=str, required=True, help='输入图片文件夹')
    parser.add_argument('--save_dir', type=str, default='./fcn_calib_npy', help='输出npy文件夹')
    args = parser.parse_args()
    main(args)