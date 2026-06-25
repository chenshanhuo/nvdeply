import json
import matplotlib.pyplot as plt
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

    sensitive_outputs = data.get('sensitive_outputs', [])
    if not sensitive_outputs:
        print("No sensitive outputs found in JSON file")
        return

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Extract data for visualization
    ranks = [item['rank'] for item in sensitive_outputs]
    # Get last part of output name (e.g., "layer5_output" from "model/layer5_output")
    short_names = [item['output_name'].split('/')[-1] for item in sensitive_outputs]
    full_output_names = [item['output_name'] for item in sensitive_outputs]

    # Extract metrics (cosine similarity, l2 distance, normalized_l2, mse, mae, pearson_correlation, psnr)
    metrics_data = {}
    metric_names = ['cosine_similarity', 'l2_distance', 'normalized_l2', 'mse', 'mae', 'pearson_correlation', 'psnr']
    for metric in metric_names:
        metrics_data[metric] = [item['metrics'][metric] for item in sensitive_outputs]

    # Create multiple visualizations
    # 1. Interactive bar chart for each metric (using Plotly)
    for metric, values in metrics_data.items():
        hover_text = [
            f"Rank: R{r}Layer: {name}{metric}: {val:.4f}"
            for r, name, val in zip(ranks, full_output_names, values)
        ]
        fig = go.Figure(
            data=[
                go.Bar(
                    x=[f"R{r}" for r in ranks],
                    y=values,
                    text=hover_text,
                    hoverinfo='text',
                    marker=dict(color=values, colorscale='Reds'),
                )
            ]
        )
        fig.update_layout(
            title=f"{metric.replace('_', ' ').title()} Overview",
            xaxis_title="Rank",
            yaxis_title=metric,
            xaxis_tickangle=-45,
            height=600,
            width=1200,
        )
        pio.write_html(fig, file=f"{output_dir}/sensitivity_metrics_overview_{metric}.html", auto_open=False)
        print(f"Generated: {output_dir}/sensitivity_metrics_overview_{metric}.html")

    print(f"Interactive sensitivity analysis visualizations saved to {output_dir}/")


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