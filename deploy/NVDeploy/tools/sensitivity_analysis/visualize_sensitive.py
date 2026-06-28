import json
import numpy as np
from pathlib import Path
import argparse
import plotly.graph_objs as go
import plotly.io as pio


def visualize_sensitivity_analysis(json_file, output_dir="./sensitivity_vis"):
    """
    Visualize layer precision sensitivity analysis results from a JSON file.
    """
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 获取 final_outputs_comparison 作为主要数据源
    final_outputs = data.get('final_outputs_comparison', {})
    if not final_outputs:
        print("No outputs found in JSON file")
        return

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 提取数据
    output_names = list(final_outputs.keys())

    # Extract metrics
    metrics_data = {
        'cosine_similarity': [],
        'l2_distance': [],
        'mse': []
    }

    for name in output_names:
        metrics = final_outputs[name]
        metrics_data['cosine_similarity'].append(metrics.get('cosine_similarity', 0))
        metrics_data['l2_distance'].append(metrics.get('l2_distance', 0))
        metrics_data['mse'].append(metrics.get('mse', 0))

    # 找出敏感输出（余弦相似度低于阈值）
    cosine_threshold = data.get('cosine_threshold', 0.95)
    sensitive_mask = [c < cosine_threshold for c in metrics_data['cosine_similarity']]

    # 生成可视化
    for metric, values in metrics_data.items():
        # 只显示有该指标的数据
        valid_indices = [i for i, v in enumerate(values) if v != 0]

        if not valid_indices:
            continue

        hover_text = [
            f"Output: {output_names[i]}<br>{metric}: {values[i]:.6f}"
            for i in valid_indices
        ]

        # 颜色：根据敏感度
        colors = ['red' if sensitive_mask[i] else 'blue' for i in valid_indices]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=[output_names[i].split('/')[-1][:20] for i in valid_indices],
                    y=[values[i] for i in valid_indices],
                    text=hover_text,
                    hoverinfo='text',
                    marker=dict(color=colors),
                )
            ]
        )

        fig.update_layout(
            title=f"{metric.replace('_', ' ').title()} Overview",
            xaxis_title="Output",
            yaxis_title=metric,
            xaxis_tickangle=-45,
            height=600,
            width=1200,
        )

        output_file = f"{output_dir}/sensitivity_{metric}.html"
        pio.write_html(fig, file=output_file, auto_open=False)
        print(f"Generated: {output_file}")

    print(f"\nInteractive sensitivity analysis visualizations saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Visualize layer precision sensitivity analysis results')
    parser.add_argument('--json-file', '-j', type=str, required=True,
                        help='Path to thensitivity an sealysis JSON file')
    parser.add_argument('--output-dir', '-o', type=str, default='./sensitivity_vis',
                        help='Output directory for visualizations (default: ./sensitivity_vis)')
    args = parser.parse_args()

    visualize_sensitivity_analysis(args.json_file, args.output_dir)


if __name__ == "__main__":
    main()