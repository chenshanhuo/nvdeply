import numpy as np
import cv2
from pathlib import Path
import argparse
import os
import json


def visualize_single_npy(npy_file, output_dir, show_stats=True):
    """
    简化版：可视化单个 .npy 文件，只保存反标准化的图片
    
    Args:
        npy_file: 单个 .npy 文件路径
        output_dir: 输出目录
        show_stats: 是否显示统计信息
    """
    npy_path = Path(npy_file)
    if not npy_path.exists():
        print(f"❌ 文件不存在: {npy_path}")
        return None

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 反标准化参数（与转换脚本一致）
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    print("=" * 60)
    print(f"📁 文件: {npy_path.name}")
    print("=" * 60)

    try:
        # 1. 加载 .npy 文件
        data = np.load(npy_path)
        print(f"📊 加载成功，数据类型: {data.dtype}，占用内存: {data.nbytes / 1024:.1f} KB")

        # 2. 验证形状
        expected_shapes = [(1, 3, 520, 520), (3, 520, 520)]
        if data.shape not in expected_shapes:
            print(f"⚠️  警告: 形状为 {data.shape}，期望 {expected_shapes}")
            print("   尝试自动调整...")
            if data.ndim == 3 and data.shape[0] == 520 and data.shape[1] == 520:
                # 可能是 HWC 格式，转换为 CHW
                data = data.transpose(2, 0, 1)
                if data.shape == (3, 520, 520):
                    data = data[np.newaxis, ...]  # 添加批次维度
                    print(f"   已调整为: {data.shape}")
                else:
                    print(f"❌ 无法自动调整形状")
                    return None
            else:
                print(f"❌ 不支持的形状")
                return None

        # 3. 提取图片数据
        if data.shape[0] == 1:
            img_data = data[0]  # 去掉批次维度 -> (3, 520, 520)
        else:
            img_data = data  # 已经是 (3, 520, 520)

        # 4. 计算统计信息
        stats = {
            "raw_min": float(data.min()),
            "raw_max": float(data.max()),
            "raw_mean": float(data.mean()),
            "raw_std": float(data.std()),
            "shape": str(data.shape),
            "has_nan": bool(np.isnan(data).any()),
            "has_inf": bool(np.isinf(data).any()),
        }

        if show_stats:
            print(f"\n📈 原始数据统计:")
            print(f"   形状: {data.shape}")
            print(f"   范围: [{stats['raw_min']:.6f}, {stats['raw_max']:.6f}]")
            print(f"   均值: {stats['raw_mean']:.6f}，标准差: {stats['raw_std']:.6f}")
            print(f"   包含NaN: {stats['has_nan']}，包含Inf: {stats['has_inf']}")

        # 5. 反标准化
        img_denorm = img_data * std + mean
        img_denorm = np.clip(img_denorm, 0, 1)

        # 6. 转换为图片格式
        img_uint8 = (img_denorm * 255).astype(np.uint8)
        img_hwc = img_uint8.transpose(1, 2, 0)  # (H, W, C)
        img_bgr = cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR)

        # 7. 保存图片
        img_output = output_path / f"{npy_path.stem}.png"
        cv2.imwrite(str(img_output), img_bgr)
        print(f"\n💾 图片已保存: {img_output}")

        # 8. 显示通道信息
        print(f"\n🎨 图片信息:")
        print(f"   尺寸: {img_hwc.shape[1]}x{img_hwc.shape[0]}")
        print(f"   颜色范围: [{img_hwc.min()}, {img_hwc.max()}]")
        
        # 9. 保存元数据
        metadata = {
            "file": str(npy_path.name),
            "original_shape": str(data.shape),
            "processed_shape": str(img_hwc.shape),
            "statistics": stats,
            "normalization": {
                "mean": mean.flatten().tolist(),
                "std": std.flatten().tolist()
            },
            "output_file": str(img_output.name)
        }

        metadata_file = output_path / f"{npy_path.stem}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"📄 元数据已保存: {metadata_file}")

        print(f"\n📁 保存位置: {output_path.absolute()}/")

        return metadata

    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数：支持单个文件或整个目录的可视化"""
    parser = argparse.ArgumentParser(description='简化版：可视化FCN校准.npy文件')
    parser.add_argument('--input', type=str, help='单个.npy文件路径')
    parser.add_argument('--npy_dir', type=str, help='.npy文件目录（批量处理）')
    parser.add_argument('--output_dir', type=str, default='./visualized_output',
                       help='输出目录')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='批量处理时的样本数量（仅当使用--npy_dir时有效）')

    args = parser.parse_args()

    # 检查必要的参数
    if not args.input and not args.npy_dir:
        print("❌ 请指定 --input（单个文件）或 --npy_dir（目录）")
        return

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 单个文件模式
    if args.input:
        print("🔄 单个文件模式")
        metadata = visualize_single_npy(
            npy_file=args.input,
            output_dir=args.output_dir,
            show_stats=True
        )

        if metadata:
            print("\n✅ 处理完成！")
            print(f"输出目录: {output_path.absolute()}")
            
            # 显示元数据中的关键信息
            print("\n📊 转换验证:")
            stats = metadata["statistics"]
            print(f"  形状正确: {stats['shape']} == (1, 3, 520, 520)")
            print(f"  数值范围合理: [{stats['raw_min']:.3f}, {stats['raw_max']:.3f}]")
            print(f"  无异常值: NaN={stats['has_nan']}, Inf={stats['has_inf']}")
        else:
            print("\n❌ 处理失败")

    # 批量模式
    elif args.npy_dir:
        print("🔄 批量处理模式")
        npy_files = list(Path(args.npy_dir).glob("*.npy"))
        if not npy_files:
            print(f"❌ 在 {args.npy_dir} 中未找到 .npy 文件")
            return

        print(f"找到 {len(npy_files)} 个文件，处理前 {args.num_samples} 个")

        successful = 0
        for i, npy_file in enumerate(npy_files[:args.num_samples]):
            print(f"\n{'='*40}")
            print(f"处理文件 {i+1}/{min(args.num_samples, len(npy_files))}")
            metadata = visualize_single_npy(
                npy_file=npy_file,
                output_dir=args.output_dir,
                show_stats=False
            )
            if metadata:
                successful += 1

        print(f"\n{'='*40}")
        print(f"批量处理完成！")
        print(f"成功: {successful}/{min(args.num_samples, len(npy_files))}")
        print(f"输出目录: {output_path.absolute()}")


if __name__ == '__main__':
    ####该文件作用：用于可视化FCN校准过程中生成的.npy文件，支持单个文件和批量处理模式。它会将.npy文件中的图像数据反标准化并保存为可视化的PNG图片，同时生成元数据JSON文件，记录原始形状、统计信息和输出文件路径。
    main()