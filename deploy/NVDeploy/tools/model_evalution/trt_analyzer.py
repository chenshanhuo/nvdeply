"""
TensorRT 性能分析核心模块
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from utils import (
    ModelProfile, LayerProfile, PrecisionType,
    parse_trtexec_json, run_trtexec, format_flops, format_number
)


@dataclass
class ComparisonResult:
    """模型对比结果"""
    baseline: ModelProfile
    optimized: ModelProfile
    
    @property
    def speedup(self) -> float:
        """加速比"""
        if self.optimized.total_time_ms > 0:
            return self.baseline.total_time_ms / self.optimized.total_time_ms
        return 0.0
    
    @property
    def flops_reduction(self) -> float:
        """FLOPs减少比例"""
        if self.baseline.total_flops > 0:
            return 1 - (self.optimized.total_flops / self.baseline.total_flops)
        return 0.0
    
    @property
    def throughput_improvement(self) -> float:
        """吞吐量提升比例"""
        if self.baseline.throughput > 0:
            return (self.optimized.throughput - self.baseline.throughput) / self.baseline.throughput
        return 0.0
    
    @property
    def actual_flops_improvement(self) -> float:
        """实际FLOPS提升比例"""
        if self.baseline.actual_flops > 0:
            return (self.optimized.actual_flops - self.baseline.actual_flops) / self.baseline.actual_flops
        return 0.0


class TRTAnalyzer:
    """TensorRT模型性能分析器"""
    
    def __init__(self, output_dir: str = "./analysis_output"):
        """
        初始化分析器
        
        Args:
            output_dir: 分析结果输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, ModelProfile] = {}
    
    def load_profile_from_json(self, json_path: str, model_name: str = "",
                               layer_info_path: str = None) -> Optional[ModelProfile]:
        """
        从JSON文件加载模型profile
        
        Args:
            json_path: trtexec输出的Profile JSON文件路径
            model_name: 模型名称
            layer_info_path: trtexec输出的LayerInfo JSON文件路径 (可选)
        
        Returns:
            ModelProfile对象
        """
        from utils import parse_trtexec_json
        
        profile = parse_trtexec_json(json_path, model_name, layer_info_path)
        if profile:
            self.models[profile.model_name] = profile
        return profile
    
    def profile_engine(self, engine_path: str, model_name: str = "",
                       warmup: int = 100, iterations: int = 1000) -> Optional[ModelProfile]:
        """
        直接对TensorRT engine进行性能分析
        
        Args:
            engine_path: engine文件路径
            model_name: 模型名称
            warmup: 预热次数
            iterations: 测试迭代次数
        
        Returns:
            ModelProfile对象
        """
        if not os.path.exists(engine_path):
            print(f"Engine文件不存在: {engine_path}")
            return None
        
        model_name = model_name or Path(engine_path).stem
        json_output = self.output_dir / f"{model_name}_profile.json"
        
        print(f"正在分析模型: {model_name}")
        if run_trtexec(engine_path, str(json_output), warmup, iterations):
            return self.load_profile_from_json(str(json_output), model_name)
        return None
    
    def compare_models(self, baseline_name: str, optimized_name: str) -> Optional[ComparisonResult]:
        """
        比较两个模型的性能
        
        Args:
            baseline_name: 基准模型名称
            optimized_name: 优化后模型名称
        
        Returns:
            ComparisonResult对象
        """
        if baseline_name not in self.models:
            print(f"未找到基准模型: {baseline_name}")
            return None
        if optimized_name not in self.models:
            print(f"未找到优化模型: {optimized_name}")
            return None
        
        return ComparisonResult(
            baseline=self.models[baseline_name],
            optimized=self.models[optimized_name]
        )
    
    def get_layer_analysis(self, model_name: str, top_k: int = 10) -> List[LayerProfile]:
        """
        获取模型中耗时最长的层
        
        Args:
            model_name: 模型名称
            top_k: 返回前k个耗时最长的层
        
        Returns:
            LayerProfile列表
        """
        if model_name not in self.models:
            return []
        
        profile = self.models[model_name]
        sorted_layers = sorted(profile.layers, key=lambda x: x.avg_time_ms, reverse=True)
        return sorted_layers[:top_k]
    
    def get_precision_distribution(self, model_name: str) -> Dict[str, int]:
        """
        获取模型各精度层的分布
        
        Args:
            model_name: 模型名称
        
        Returns:
            精度分布字典
        """
        if model_name not in self.models:
            return {}
        
        profile = self.models[model_name]
        distribution = {}
        for layer in profile.layers:
            precision = layer.precision.upper()
            distribution[precision] = distribution.get(precision, 0) + 1
        return distribution
    
    def get_layer_type_analysis(self, model_name: str) -> Dict[str, Dict]:
        """
        按层类型分析性能
        
        Args:
            model_name: 模型名称
        
        Returns:
            层类型分析字典
        """
        if model_name not in self.models:
            return {}
        
        profile = self.models[model_name]
        analysis = {}
        
        for layer in profile.layers:
            layer_type = layer.layer_type
            if layer_type not in analysis:
                analysis[layer_type] = {
                    'count': 0,
                    'total_time_ms': 0.0,
                    'total_flops': 0,
                    'layers': []
                }
            
            analysis[layer_type]['count'] += 1
            analysis[layer_type]['total_time_ms'] += layer.avg_time_ms
            analysis[layer_type]['total_flops'] += layer.flops_estimate
            analysis[layer_type]['layers'].append(layer.name)
        
        # 计算百分比
        total_time = profile.total_time_ms
        for layer_type in analysis:
            analysis[layer_type]['percentage'] = (
                analysis[layer_type]['total_time_ms'] / total_time * 100
                if total_time > 0 else 0
            )
        
        return analysis
    
    def generate_report(self, model_name: str) -> str:
        """
        生成模型分析报告
        
        Args:
            model_name: 模型名称
        
        Returns:
            报告文本
        """
        if model_name not in self.models:
            return f"未找到模型: {model_name}"
        
        profile = self.models[model_name]
        
        report = []
        report.append("=" * 60)
        report.append(f"模型性能分析报告: {model_name}")
        report.append("=" * 60)
        report.append("")
        
        # 基本信息
        report.append("【基本信息】")
        report.append(f"  模型路径: {profile.engine_path}")
        report.append(f"  层数量: {len(profile.layers)}")
        report.append(f"  主要精度: {profile.precision}")
        report.append("")
        
        # 性能指标
        report.append("【性能指标】")
        report.append(f"  平均延迟: {profile.latency_mean_ms:.3f} ms")
        report.append(f"  中位延迟: {profile.latency_median_ms:.3f} ms")
        report.append(f"  99%延迟: {profile.latency_99_ms:.3f} ms")
        report.append(f"  吞吐量: {profile.throughput:.2f} FPS")
        report.append("")
        
        # FLOPs分析
        report.append("【FLOPs分析】")
        report.append(f"  理论FLOPs: {format_number(profile.total_flops)}")
        report.append(f"  实际FLOPS: {format_flops(profile.actual_flops)}")
        report.append("")
        
        # 耗时最长的层
        report.append("【耗时TOP10层】")
        top_layers = self.get_layer_analysis(model_name, 10)
        for i, layer in enumerate(top_layers, 1):
            report.append(f"  {i}. {layer.name}")
            report.append(f"     类型: {layer.layer_type}, 精度: {layer.precision}")
            report.append(f"     耗时: {layer.avg_time_ms:.3f} ms ({layer.percentage:.1f}%)")
        report.append("")
        
        # 层类型分析
        report.append("【层类型分析】")
        type_analysis = self.get_layer_type_analysis(model_name)
        sorted_types = sorted(type_analysis.items(), 
                             key=lambda x: x[1]['total_time_ms'], reverse=True)
        for layer_type, info in sorted_types[:10]:
            report.append(f"  {layer_type}:")
            report.append(f"    数量: {info['count']}, 耗时: {info['total_time_ms']:.3f} ms ({info['percentage']:.1f}%)")
        report.append("")
        
        # 精度分布
        report.append("【精度分布】")
        precision_dist = self.get_precision_distribution(model_name)
        for precision, count in sorted(precision_dist.items()):
            report.append(f"  {precision}: {count} 层")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def generate_comparison_report(self, comparison: ComparisonResult) -> str:
        """
        生成对比分析报告
        
        Args:
            comparison: ComparisonResult对象
        
        Returns:
            报告文本
        """
        baseline = comparison.baseline
        optimized = comparison.optimized
        
        report = []
        report.append("=" * 70)
        report.append("模型优化对比报告")
        report.append("=" * 70)
        report.append("")
        
        # 模型信息
        report.append("【模型信息】")
        report.append(f"  基准模型: {baseline.model_name}")
        report.append(f"  优化模型: {optimized.model_name}")
        report.append("")
        
        # 性能对比
        report.append("【性能对比】")
        report.append(f"{'指标':<20} {'基准模型':<15} {'优化模型':<15} {'变化':<15}")
        report.append("-" * 65)
        
        # 延迟
        latency_change = (optimized.latency_mean_ms - baseline.latency_mean_ms) / baseline.latency_mean_ms * 100
        report.append(f"{'平均延迟(ms)':<20} {baseline.latency_mean_ms:<15.3f} {optimized.latency_mean_ms:<15.3f} {latency_change:+.1f}%")
        
        # 吞吐量
        throughput_change = comparison.throughput_improvement * 100
        report.append(f"{'吞吐量(FPS)':<20} {baseline.throughput:<15.2f} {optimized.throughput:<15.2f} {throughput_change:+.1f}%")
        
        # 加速比
        report.append(f"{'加速比':<20} {'1.0x':<15} {comparison.speedup:<15.2f}x")
        
        report.append("")
        
        # FLOPs对比
        report.append("【FLOPs对比】")
        report.append(f"{'指标':<20} {'基准模型':<20} {'优化模型':<20} {'变化':<15}")
        report.append("-" * 75)
        
        flops_change = -comparison.flops_reduction * 100
        report.append(f"{'理论FLOPs':<20} {format_number(baseline.total_flops):<20} {format_number(optimized.total_flops):<20} {flops_change:+.1f}%")
        
        actual_flops_change = comparison.actual_flops_improvement * 100
        report.append(f"{'实际FLOPS':<20} {format_flops(baseline.actual_flops):<20} {format_flops(optimized.actual_flops):<20} {actual_flops_change:+.1f}%")
        
        report.append("")
        
        # 层数量对比
        report.append("【层数量对比】")
        report.append(f"  基准模型层数: {len(baseline.layers)}")
        report.append(f"  优化模型层数: {len(optimized.layers)}")
        layer_reduction = (len(baseline.layers) - len(optimized.layers)) / len(baseline.layers) * 100
        report.append(f"  层数减少: {layer_reduction:.1f}%")
        
        report.append("")
        
        # 精度变化
        report.append("【精度分布变化】")
        baseline_precision = {}
        for layer in baseline.layers:
            p = layer.precision.upper()
            baseline_precision[p] = baseline_precision.get(p, 0) + 1
        
        optimized_precision = {}
        for layer in optimized.layers:
            p = layer.precision.upper()
            optimized_precision[p] = optimized_precision.get(p, 0) + 1
        
        all_precisions = set(baseline_precision.keys()) | set(optimized_precision.keys())
        for precision in sorted(all_precisions):
            base_count = baseline_precision.get(precision, 0)
            opt_count = optimized_precision.get(precision, 0)
            report.append(f"  {precision}: {base_count} -> {opt_count}")
        
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def export_to_json(self, model_name: str, output_path: str = None) -> str:
        """
        导出分析结果为JSON
        
        Args:
            model_name: 模型名称
            output_path: 输出路径
        
        Returns:
            输出文件路径
        """
        if model_name not in self.models:
            return ""
        
        profile = self.models[model_name]
        output_path = output_path or str(self.output_dir / f"{model_name}_analysis.json")
        
        data = {
            'model_name': profile.model_name,
            'engine_path': profile.engine_path,
            'performance': {
                'total_time_ms': profile.total_time_ms,
                'throughput_qps': profile.throughput,
                'latency_mean_ms': profile.latency_mean_ms,
                'latency_median_ms': profile.latency_median_ms,
                'latency_99_ms': profile.latency_99_ms,
            },
            'flops': {
                'theoretical_flops': profile.total_flops,
                'actual_flops': profile.actual_flops,
            },
            'layers': [
                {
                    'name': layer.name,
                    'type': layer.layer_type,
                    'precision': layer.precision,
                    'avg_time_ms': layer.avg_time_ms,
                    'percentage': layer.percentage,
                    'flops_estimate': layer.flops_estimate,
                }
                for layer in profile.layers
            ],
            'precision_distribution': self.get_precision_distribution(model_name),
            'layer_type_analysis': {
                k: {
                    'count': v['count'],
                    'total_time_ms': v['total_time_ms'],
                    'percentage': v['percentage'],
                    'total_flops': v['total_flops'],
                }
                for k, v in self.get_layer_type_analysis(model_name).items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return output_path
