"""
可视化模块
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from utils import ModelProfile, format_flops, format_number
from trt_analyzer import TRTAnalyzer, ComparisonResult


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class TRTVisualizer:
    """TensorRT性能可视化器"""
    
    def __init__(self, analyzer: TRTAnalyzer, output_dir: str = "./visualization"):
        """
        初始化可视化器
        
        Args:
            analyzer: TRTAnalyzer实例
            output_dir: 图表输出目录
        """
        self.analyzer = analyzer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 配色方案
        self.colors = {
            'primary': '#2196F3',
            'secondary': '#FF9800', 
            'success': '#4CAF50',
            'warning': '#FFC107',
            'danger': '#F44336',
            'info': '#00BCD4',
        }
        
        self.precision_colors = {
            'FP32': '#2196F3',
            'FP16': '#4CAF50',
            'INT8': '#FF9800',
            'INT4': '#F44336',
            'UNKNOWN': '#9E9E9E',
        }
    
    def plot_layer_time_distribution(self, model_name: str, top_k: int = 20,
                                     save: bool = True) -> plt.Figure:
        """
        绘制层耗时分布图
        
        Args:
            model_name: 模型名称
            top_k: 显示前k个层
            save: 是否保存图片
        
        Returns:
            matplotlib Figure对象
        """
        if model_name not in self.analyzer.models:
            print(f"未找到模型: {model_name}")
            return None
        
        profile = self.analyzer.models[model_name]
        top_layers = sorted(profile.layers, key=lambda x: x.avg_time_ms, reverse=True)[:top_k]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        names = [layer.name[:30] + '...' if len(layer.name) > 30 else layer.name 
                 for layer in top_layers]
        times = [layer.avg_time_ms for layer in top_layers]
        colors = [self.precision_colors.get(layer.precision.upper(), '#9E9E9E') 
                  for layer in top_layers]
        
        bars = ax.barh(range(len(names)), times, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_title(f'Layer Time Distribution - {model_name}\n(Top {top_k} Layers)', fontsize=14)
        
        # 添加数值标签
        for i, (bar, time) in enumerate(zip(bars, times)):
            ax.text(bar.get_width() + 0.01 * max(times), bar.get_y() + bar.get_height()/2,
                   f'{time:.3f}ms ({top_layers[i].percentage:.1f}%)',
                   va='center', fontsize=8)
        
        # 添加图例
        legend_patches = [mpatches.Patch(color=color, label=precision) 
                         for precision, color in self.precision_colors.items()
                         if any(l.precision.upper() == precision for l in top_layers)]
        ax.legend(handles=legend_patches, loc='lower right', title='Precision')
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / f"{model_name}_layer_time.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        
        return fig
    
    def plot_precision_distribution(self, model_name: str, save: bool = True) -> plt.Figure:
        """
        绘制精度分布饼图
        
        Args:
            model_name: 模型名称
            save: 是否保存图片
        
        Returns:
            matplotlib Figure对象
        """
        distribution = self.analyzer.get_precision_distribution(model_name)
        if not distribution:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 层数量饼图
        labels = list(distribution.keys())
        sizes = list(distribution.values())
        colors = [self.precision_colors.get(p, '#9E9E9E') for p in labels]
        
        axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                   startangle=90, explode=[0.02]*len(sizes))
        axes[0].set_title(f'Precision Distribution by Layer Count\n{model_name}', fontsize=12)
        
        # 耗时分布饼图
        profile = self.analyzer.models[model_name]
        time_by_precision = {}
        for layer in profile.layers:
            p = layer.precision.upper()
            time_by_precision[p] = time_by_precision.get(p, 0) + layer.avg_time_ms
        
        labels2 = list(time_by_precision.keys())
        sizes2 = list(time_by_precision.values())
        colors2 = [self.precision_colors.get(p, '#9E9E9E') for p in labels2]
        
        axes[1].pie(sizes2, labels=labels2, colors=colors2, autopct='%1.1f%%',
                   startangle=90, explode=[0.02]*len(sizes2))
        axes[1].set_title(f'Precision Distribution by Time\n{model_name}', fontsize=12)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / f"{model_name}_precision_dist.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        
        return fig
    
    def plot_layer_type_analysis(self, model_name: str, save: bool = True) -> plt.Figure:
        """
        绘制层类型分析图
        
        Args:
            model_name: 模型名称
            save: 是否保存图片
        
        Returns:
            matplotlib Figure对象
        """
        analysis = self.analyzer.get_layer_type_analysis(model_name)
        if not analysis:
            return None
        
        # 按耗时排序
        sorted_types = sorted(analysis.items(), key=lambda x: x[1]['total_time_ms'], reverse=True)[:15]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # 耗时柱状图
        types = [t[0][:20] for t in sorted_types]
        times = [t[1]['total_time_ms'] for t in sorted_types]
        counts = [t[1]['count'] for t in sorted_types]
        
        x = np.arange(len(types))
        width = 0.35
        
        bars1 = axes[0].bar(x, times, width, label='Total Time (ms)', color=self.colors['primary'])
        axes[0].set_ylabel('Time (ms)', fontsize=11)
        axes[0].set_xlabel('Layer Type', fontsize=11)
        axes[0].set_title(f'Time by Layer Type - {model_name}', fontsize=12)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(types, rotation=45, ha='right', fontsize=9)
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            axes[0].annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        # 层数量和FLOPs
        flops = [t[1]['total_flops'] / 1e9 for t in sorted_types]  # 转换为GFLOPs
        
        ax2 = axes[1].twinx()
        bars2 = axes[1].bar(x - width/2, counts, width, label='Layer Count', color=self.colors['success'])
        bars3 = ax2.bar(x + width/2, flops, width, label='GFLOPs', color=self.colors['warning'])
        
        axes[1].set_ylabel('Layer Count', fontsize=11, color=self.colors['success'])
        ax2.set_ylabel('GFLOPs', fontsize=11, color=self.colors['warning'])
        axes[1].set_xlabel('Layer Type', fontsize=11)
        axes[1].set_title(f'Layer Count & FLOPs by Type - {model_name}', fontsize=12)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(types, rotation=45, ha='right', fontsize=9)
        
        # 合并图例
        lines1, labels1 = axes[1].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[1].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / f"{model_name}_layer_type.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        
        return fig
    
    def plot_comparison(self, comparison: ComparisonResult, save: bool = True) -> plt.Figure:
        """
        绘制模型对比图
        
        Args:
            comparison: ComparisonResult对象
            save: 是否保存图片
        
        Returns:
            matplotlib Figure对象
        """
        baseline = comparison.baseline
        optimized = comparison.optimized
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. 延迟对比
        ax1 = axes[0, 0]
        metrics = ['Mean\nLatency', 'Median\nLatency', '99%\nLatency']
        baseline_vals = [baseline.latency_mean_ms, baseline.latency_median_ms, baseline.latency_99_ms]
        optimized_vals = [optimized.latency_mean_ms, optimized.latency_median_ms, optimized.latency_99_ms]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, baseline_vals, width, label='Baseline', color=self.colors['primary'])
        bars2 = ax1.bar(x + width/2, optimized_vals, width, label='Optimized', color=self.colors['success'])
        
        ax1.set_ylabel('Latency (ms)', fontsize=11)
        ax1.set_title('Latency Comparison', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.legend()
        
        # 添加百分比变化标签
        for i, (b, o) in enumerate(zip(baseline_vals, optimized_vals)):
            change = (o - b) / b * 100
            color = self.colors['success'] if change < 0 else self.colors['danger']
            ax1.annotate(f'{change:+.1f}%', xy=(x[i], max(b, o)), xytext=(0, 5),
                        textcoords="offset points", ha='center', fontsize=10, color=color)
        
        # 2. 吞吐量对比
        ax2 = axes[0, 1]
        throughputs = [baseline.throughput, optimized.throughput]
        bars = ax2.bar(['Baseline', 'Optimized'], throughputs, 
                      color=[self.colors['primary'], self.colors['success']])
        ax2.set_ylabel('Throughput (FPS)', fontsize=11)
        ax2.set_title(f'Throughput Comparison\n(Speedup: {comparison.speedup:.2f}x)', fontsize=12)
        
        for bar, val in zip(bars, throughputs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}', ha='center', fontsize=11)
        
        # 3. FLOPs对比
        ax3 = axes[1, 0]
        flops = [baseline.total_flops / 1e9, optimized.total_flops / 1e9]
        actual_flops = [baseline.actual_flops / 1e12, optimized.actual_flops / 1e12]
        
        x = np.arange(2)
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, flops, width, label='Theoretical (GFLOPs)', color=self.colors['info'])
        bars2 = ax3.bar(x + width/2, [f * 1000 for f in actual_flops], width, 
                       label='Actual (GFLOPS)', color=self.colors['warning'])
        
        ax3.set_ylabel('GFLOPs / GFLOPS', fontsize=11)
        ax3.set_title(f'FLOPs Comparison\n(Reduction: {comparison.flops_reduction*100:.1f}%)', fontsize=12)
        ax3.set_xticks(x)
        ax3.set_xticklabels(['Baseline', 'Optimized'])
        ax3.legend()
        
        # 4. 层数量和精度对比
        ax4 = axes[1, 1]
        
        # 获取精度分布
        baseline_precision = self.analyzer.get_precision_distribution(baseline.model_name)
        optimized_precision = self.analyzer.get_precision_distribution(optimized.model_name)
        
        all_precisions = sorted(set(baseline_precision.keys()) | set(optimized_precision.keys()))
        
        x = np.arange(len(all_precisions))
        width = 0.35
        
        baseline_counts = [baseline_precision.get(p, 0) for p in all_precisions]
        optimized_counts = [optimized_precision.get(p, 0) for p in all_precisions]
        
        bars1 = ax4.bar(x - width/2, baseline_counts, width, label='Baseline', 
                       color=self.colors['primary'])
        bars2 = ax4.bar(x + width/2, optimized_counts, width, label='Optimized',
                       color=self.colors['success'])
        
        ax4.set_ylabel('Layer Count', fontsize=11)
        ax4.set_title('Precision Distribution Comparison', fontsize=12)
        ax4.set_xticks(x)
        ax4.set_xticklabels(all_precisions)
        ax4.legend()
        
        plt.suptitle(f'Model Comparison: {baseline.model_name} vs {optimized.model_name}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / f"comparison_{baseline.model_name}_vs_{optimized.model_name}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        
        return fig
    
    def plot_flops_efficiency(self, model_name: str, save: bool = True) -> plt.Figure:
        """
        绘制FLOPs效率分析图
        
        Args:
            model_name: 模型名称
            save: 是否保存图片
        
        Returns:
            matplotlib Figure对象
        """
        if model_name not in self.analyzer.models:
            return None
        
        profile = self.analyzer.models[model_name]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. 每层FLOPs效率 (FLOPS = FLOPs / time)
        layers = sorted(profile.layers, key=lambda x: x.avg_time_ms, reverse=True)[:20]
        
        names = [l.name[:25] + '...' if len(l.name) > 25 else l.name for l in layers]
        efficiencies = []
        for layer in layers:
            if layer.avg_time_ms > 0:
                eff = layer.flops_estimate / (layer.avg_time_ms / 1000) / 1e9  # GFLOPS
            else:
                eff = 0
            efficiencies.append(eff)
        
        colors = [self.precision_colors.get(l.precision.upper(), '#9E9E9E') for l in layers]
        
        bars = axes[0].barh(range(len(names)), efficiencies, color=colors)
        axes[0].set_yticks(range(len(names)))
        axes[0].set_yticklabels(names, fontsize=9)
        axes[0].invert_yaxis()
        axes[0].set_xlabel('Efficiency (GFLOPS)', fontsize=11)
        axes[0].set_title(f'Layer Efficiency - {model_name}', fontsize=12)
        
        # 2. FLOPs vs Time 散点图
        all_flops = [l.flops_estimate / 1e6 for l in profile.layers]  # MFLOPs
        all_times = [l.avg_time_ms for l in profile.layers]
        all_colors = [self.precision_colors.get(l.precision.upper(), '#9E9E9E') for l in profile.layers]
        
        axes[1].scatter(all_flops, all_times, c=all_colors, alpha=0.6, s=50)
        axes[1].set_xlabel('FLOPs (MFLOPs)', fontsize=11)
        axes[1].set_ylabel('Time (ms)', fontsize=11)
        axes[1].set_title(f'FLOPs vs Time - {model_name}', fontsize=12)
        
        # 添加图例
        legend_patches = [mpatches.Patch(color=color, label=precision) 
                         for precision, color in self.precision_colors.items()
                         if any(l.precision.upper() == precision for l in profile.layers)]
        axes[1].legend(handles=legend_patches, loc='upper left', title='Precision')
        
        # 添加趋势线
        if len(all_flops) > 1:
            z = np.polyfit(all_flops, all_times, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(all_flops), max(all_flops), 100)
            axes[1].plot(x_line, p(x_line), "r--", alpha=0.5, label='Trend')
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / f"{model_name}_flops_efficiency.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存: {save_path}")
        
        return fig
    
    def plot_interactive_comparison(self, comparison: ComparisonResult, 
                                   save: bool = True) -> Optional['go.Figure']:
        """
        绘制交互式对比图 (使用Plotly)
        
        Args:
            comparison: ComparisonResult对象
            save: 是否保存HTML文件
        
        Returns:
            Plotly Figure对象
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly未安装，无法生成交互式图表")
            return None
        
        baseline = comparison.baseline
        optimized = comparison.optimized
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Latency Comparison',
                'Throughput & Speedup',
                'FLOPs Analysis', 
                'Layer Time Distribution'
            ),
            specs=[
                [{"type": "bar"}, {"type": "indicator"}],
                [{"type": "bar"}, {"type": "bar"}]
            ]
        )
        
        # 1. 延迟对比
        metrics = ['Mean Latency', 'Median Latency', '99% Latency']
        baseline_vals = [baseline.latency_mean_ms, baseline.latency_median_ms, baseline.latency_99_ms]
        optimized_vals = [optimized.latency_mean_ms, optimized.latency_median_ms, optimized.latency_99_ms]
        
        fig.add_trace(go.Bar(name='Baseline', x=metrics, y=baseline_vals, 
                            marker_color=self.colors['primary']), row=1, col=1)
        fig.add_trace(go.Bar(name='Optimized', x=metrics, y=optimized_vals,
                            marker_color=self.colors['success']), row=1, col=1)
        
        # 2. 加速比指示器
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=comparison.speedup,
            title={'text': "Speedup"},
            delta={'reference': 1.0},
            gauge={
                'axis': {'range': [0, max(3, comparison.speedup * 1.2)]},
                'bar': {'color': self.colors['success']},
                'steps': [
                    {'range': [0, 1], 'color': "lightgray"},
                    {'range': [1, 2], 'color': "lightgreen"},
                    {'range': [2, 3], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 1.0
                }
            }
        ), row=1, col=2)
        
        # 3. FLOPs对比
        fig.add_trace(go.Bar(
            name='Theoretical FLOPs (G)',
            x=['Baseline', 'Optimized'],
            y=[baseline.total_flops / 1e9, optimized.total_flops / 1e9],
            marker_color=self.colors['info']
        ), row=2, col=1)
        
        fig.add_trace(go.Bar(
            name='Actual FLOPS (T)',
            x=['Baseline', 'Optimized'],
            y=[baseline.actual_flops / 1e12, optimized.actual_flops / 1e12],
            marker_color=self.colors['warning']
        ), row=2, col=1)
        
        # 4. 层耗时对比 (Top 10)
        baseline_top = sorted(baseline.layers, key=lambda x: x.avg_time_ms, reverse=True)[:10]
        optimized_top = sorted(optimized.layers, key=lambda x: x.avg_time_ms, reverse=True)[:10]
        
        baseline_names = [l.name[:20] for l in baseline_top]
        baseline_times = [l.avg_time_ms for l in baseline_top]
        
        optimized_names = [l.name[:20] for l in optimized_top]
        optimized_times = [l.avg_time_ms for l in optimized_top]
        
        fig.add_trace(go.Bar(
            name='Baseline Top Layers',
            x=baseline_names,
            y=baseline_times,
            marker_color=self.colors['primary']
        ), row=2, col=2)
        
        fig.update_layout(
            title=f'Model Comparison: {baseline.model_name} vs {optimized.model_name}',
            height=800,
            showlegend=True,
            barmode='group'
        )
        
        if save:
            save_path = self.output_dir / f"interactive_comparison_{baseline.model_name}_vs_{optimized.model_name}.html"
            fig.write_html(str(save_path))
            print(f"交互式图表已保存: {save_path}")
        
        return fig
    
    def generate_dashboard(self, model_names: List[str], save: bool = True) -> Optional['go.Figure']:
        """
        生成多模型对比仪表板
        
        Args:
            model_names: 模型名称列表
            save: 是否保存
        
        Returns:
            Plotly Figure对象
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly未安装，无法生成仪表板")
            return None
        
        profiles = [self.analyzer.models[name] for name in model_names 
                   if name in self.analyzer.models]
        
        if not profiles:
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Latency Comparison',
                'Throughput Comparison',
                'Theoretical FLOPs',
                'Actual FLOPS'
            )
        )
        
        colors = px.colors.qualitative.Set1[:len(profiles)]
        
        # 延迟
        for i, profile in enumerate(profiles):
            fig.add_trace(go.Bar(
                name=profile.model_name,
                x=['Mean', 'Median', '99%'],
                y=[profile.latency_mean_ms, profile.latency_median_ms, profile.latency_99_ms],
                marker_color=colors[i],
                showlegend=True
            ), row=1, col=1)
        
        # 吞吐量
        fig.add_trace(go.Bar(
            x=[p.model_name for p in profiles],
            y=[p.throughput for p in profiles],
            marker_color=colors,
            showlegend=False
        ), row=1, col=2)
        
        # 理论FLOPs
        fig.add_trace(go.Bar(
            x=[p.model_name for p in profiles],
            y=[p.total_flops / 1e9 for p in profiles],
            marker_color=colors,
            showlegend=False
        ), row=2, col=1)
        
        # 实际FLOPS
        fig.add_trace(go.Bar(
            x=[p.model_name for p in profiles],
            y=[p.actual_flops / 1e12 for p in profiles],
            marker_color=colors,
            showlegend=False
        ), row=2, col=2)
        
        fig.update_layout(
            title='Model Performance Dashboard',
            height=800,
            barmode='group'
        )
        
        fig.update_yaxes(title_text="Latency (ms)", row=1, col=1)
        fig.update_yaxes(title_text="FPS", row=1, col=2)
        fig.update_yaxes(title_text="GFLOPs", row=2, col=1)
        fig.update_yaxes(title_text="TFLOPS", row=2, col=2)
        
        if save:
            save_path = self.output_dir / "model_dashboard.html"
            fig.write_html(str(save_path))
            print(f"仪表板已保存: {save_path}")
        
        return fig