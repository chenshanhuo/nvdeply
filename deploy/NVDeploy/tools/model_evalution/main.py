#!/usr/bin/env python3
"""
TensorRT 模型性能可视化分析工具 - 主程序
"""
import argparse
import sys
import os
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from trt_analyzer import TRTAnalyzer, ComparisonResult
from visualizer import TRTVisualizer
from utils import format_flops, format_number


console = Console()


def print_banner():
    """打印横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║       TensorRT Model Performance Analysis Tool               ║
║          模型量化/剪枝性能评估可视化工具                     ║
╚══════════════════════════════════════════════════════════════╝
    """
    console.print(Panel(banner, style="bold blue"))


def print_model_summary(analyzer: TRTAnalyzer, model_name: str):
    """打印模型摘要"""
    if model_name not in analyzer.models:
        console.print(f"[red]未找到模型: {model_name}[/red]")
        return
    
    profile = analyzer.models[model_name]
    
    # 基本信息表
    table = Table(title=f"📊 模型性能摘要: {model_name}", show_header=True)
    table.add_column("指标", style="cyan")
    table.add_column("值", style="green")
    
    table.add_row("平均延迟", f"{profile.latency_mean_ms:.3f} ms")
    table.add_row("中位延迟", f"{profile.latency_median_ms:.3f} ms")
    table.add_row("99%延迟", f"{profile.latency_99_ms:.3f} ms")
    table.add_row("吞吐量", f"{profile.throughput:.2f} FPS")
    table.add_row("层数量", str(len(profile.layers)))
    table.add_row("理论FLOPs", format_number(profile.total_flops))
    table.add_row("实际FLOPS", format_flops(profile.actual_flops))
    
    console.print(table)


def print_comparison_summary(comparison: ComparisonResult):
    """打印对比摘要"""
    baseline = comparison.baseline
    optimized = comparison.optimized
    
    table = Table(title="📈 模型对比摘要", show_header=True)
    table.add_column("指标", style="cyan")
    table.add_column("基准模型", style="yellow")
    table.add_column("优化模型", style="green")
    table.add_column("变化", style="magenta")
    
    # 延迟 - 添加除零保护
    if baseline.latency_mean_ms > 0:
        latency_change = (optimized.latency_mean_ms - baseline.latency_mean_ms) / baseline.latency_mean_ms * 100
    else:
        latency_change = 0.0
    latency_style = "green" if latency_change < 0 else "red"
    table.add_row(
        "平均延迟",
        f"{baseline.latency_mean_ms:.3f} ms",
        f"{optimized.latency_mean_ms:.3f} ms",
        f"[{latency_style}]{latency_change:+.1f}%[/{latency_style}]"
    )
    
    # 吞吐量 - 添加除零保护
    throughput_change = comparison.throughput_improvement * 100 if comparison.throughput_improvement else 0.0
    throughput_style = "green" if throughput_change > 0 else "red"
    table.add_row(
        "吞吐量",
        f"{baseline.throughput:.2f} FPS",
        f"{optimized.throughput:.2f} FPS",
        f"[{throughput_style}]{throughput_change:+.1f}%[/{throughput_style}]"
    )
    
    # 加速比 - 添加保护
    speedup = comparison.speedup if comparison.speedup else 1.0
    speedup_change = (speedup - 1) * 100
    table.add_row(
        "加速比",
        "1.0x",
        f"{speedup:.2f}x",
        f"[green]{speedup_change:+.1f}%[/green]" if speedup_change >= 0 else f"[red]{speedup_change:+.1f}%[/red]"
    )
    
    # FLOPs - 添加除零保护
    flops_change = -comparison.flops_reduction * 100 if comparison.flops_reduction else 0.0
    table.add_row(
        "理论FLOPs",
        format_number(baseline.total_flops),
        format_number(optimized.total_flops),
        f"[green]{flops_change:+.1f}%[/green]" if flops_change <= 0 else f"[red]{flops_change:+.1f}%[/red]"
    )
    
    # FLOPS - 添加除零保护
    if baseline.actual_flops > 0:
        actual_flops_change = (optimized.actual_flops - baseline.actual_flops) / baseline.actual_flops * 100
    else:
        actual_flops_change = 0.0
    table.add_row(
        "实际FLOPS",
        format_number(baseline.actual_flops),
        format_number(optimized.actual_flops),
        f"[green]{actual_flops_change:+.1f}%[/green]" if actual_flops_change >= 0 else f"[red]{actual_flops_change:+.1f}%[/red]"
    )
    
    # 层数量 - 添加除零保护
    if len(baseline.layers) > 0:
        layer_change = (len(optimized.layers) - len(baseline.layers)) / len(baseline.layers) * 100
    else:
        layer_change = 0.0
    table.add_row(
        "层数量",
        str(len(baseline.layers)),
        str(len(optimized.layers)),
        f"[green]{layer_change:+.1f}%[/green]" if layer_change <= 0 else f"[red]{layer_change:+.1f}%[/red]"
    )
    
    console.print(table)


def analyze_single_model(args):
    """分析单个模型"""
    analyzer = TRTAnalyzer(output_dir=args.output)
    visualizer = TRTVisualizer(analyzer, output_dir=args.output)
    
    console.print(f"\n[bold]正在加载模型: {args.json}[/bold]")
    
    # 支持layers文件
    layers_path = getattr(args, 'layers', None)#从参数中获取layers文件路径，如果没有提供则为None
    profile = analyzer.load_profile_from_json(args.json, args.name, layers_path)
    
    if not profile:
        console.print("[red]加载模型失败[/red]")
        return
    
    print_model_summary(analyzer, profile.model_name)
    
    # 生成报告
    if args.report:
        report = analyzer.generate_report(profile.model_name)
        report_path = Path(args.output) / f"{profile.model_name}_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        console.print(f"\n[green]报告已保存: {report_path}[/green]")
    
    # 导出JSON
    if args.export_json:
        json_path = analyzer.export_to_json(profile.model_name)
        console.print(f"[green]JSON已导出: {json_path}[/green]")
    
    # 生成可视化
    if not args.no_viz:
        console.print("\n[bold]生成可视化图表...[/bold]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("生成图表", total=4)
            
            visualizer.plot_layer_time_distribution(profile.model_name)
            progress.advance(task)
            
            visualizer.plot_precision_distribution(profile.model_name)
            progress.advance(task)
            
            visualizer.plot_layer_type_analysis(profile.model_name)
            progress.advance(task)
            
            visualizer.plot_flops_efficiency(profile.model_name)
            progress.advance(task)
        
        console.print(f"[green]图表已保存到: {args.output}[/green]")


def compare_models(args):
    """对比两个模型"""
    analyzer = TRTAnalyzer(output_dir=args.output)
    visualizer = TRTVisualizer(analyzer, output_dir=args.output)
    
    # 获取layers文件路径
    baseline_layers = getattr(args, 'baseline_layers', None)
    optimized_layers = getattr(args, 'optimized_layers', None)
    
    console.print(f"\n[bold]加载基准模型: {args.baseline}[/bold]")
    if baseline_layers:
        console.print(f"[dim]  层信息文件: {baseline_layers}[/dim]")
    baseline = analyzer.load_profile_from_json(
        args.baseline, 
        args.baseline_name or "baseline",
        baseline_layers
    )
    
    console.print(f"[bold]加载优化模型: {args.optimized}[/bold]")
    if optimized_layers:
        console.print(f"[dim]  层信息文件: {optimized_layers}[/dim]")
    optimized = analyzer.load_profile_from_json(
        args.optimized, 
        args.optimized_name or "optimized",
        optimized_layers
    )
    
    if not baseline or not optimized:
        console.print("[red]加载模型失败[/red]")
        return
    
    comparison = analyzer.compare_models(baseline.model_name, optimized.model_name)
    
    if not comparison:
        console.print("[red]对比失败[/red]")
        return
    
    print_comparison_summary(comparison)
    
    # 生成对比报告
    if args.report:
        report = analyzer.generate_comparison_report(comparison)
        report_path = Path(args.output) / f"comparison_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        console.print(f"\n[green]对比报告已保存: {report_path}[/green]")
    
    # 生成可视化
    if not args.no_viz:
        console.print("\n[bold]生成对比图表...[/bold]")
        
        visualizer.plot_comparison(comparison)
        
        if args.interactive:
            visualizer.plot_interactive_comparison(comparison)
        
        console.print(f"[green]图表已保存到: {args.output}[/green]")


def run_trtexec_analysis(args):
    """直接运行trtexec进行分析"""
    analyzer = TRTAnalyzer(output_dir=args.output)
    visualizer = TRTVisualizer(analyzer, output_dir=args.output)
    
    console.print(f"\n[bold]对Engine进行性能分析: {args.engine}[/bold]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("运行trtexec分析...", total=None)
        
        profile = analyzer.profile_engine(
            args.engine, 
            args.name,
            warmup=args.warmup,
            iterations=args.iterations
        )
    
    if not profile:
        console.print("[red]分析失败[/red]")
        return
    
    print_model_summary(analyzer, profile.model_name)
    
    if not args.no_viz:
        visualizer.plot_layer_time_distribution(profile.model_name)
        visualizer.plot_precision_distribution(profile.model_name)
        visualizer.plot_layer_type_analysis(profile.model_name)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="TensorRT 模型性能可视化分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        示例用法:
        # 分析单个模型 (从JSON)
        python main.py analyze --json profile.json --name my_model
        
        # 分析单个模型 (带layers信息)
        python main.py analyze --json profile.json --layers layers.json --name my_model
        
        # 对比两个模型
        python main.py compare --baseline fp16.json --optimized int8.json
        
        # 对比两个模型 (带layers信息)
        python main.py compare \\
            --baseline fp32_profile.json --baseline-layers fp32_layers.json \\
            --optimized int8_profile.json --optimized-layers int8_layers.json
        
        # 直接对engine进行分析
        python main.py profile --engine model.engine --name my_model

        # 生成示例数据用于测试
        python main.py demo
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 分析单个模型 相关参数
    analyze_parser = subparsers.add_parser('analyze', help='分析单个模型')
    analyze_parser.add_argument('--json', '-j', required=True, help='trtexec输出的Profile JSON文件')
    analyze_parser.add_argument('--layers', '-l', default=None, help='trtexec输出的LayerInfo JSON文件 (可选)')
    analyze_parser.add_argument('--name', '-n', default='', help='模型名称')
    analyze_parser.add_argument('--output', '-o', default='./analysis_output', help='输出目录')
    analyze_parser.add_argument('--report', '-r', action='store_true', help='生成文本报告')
    analyze_parser.add_argument('--export-json', action='store_true', help='导出分析结果为JSON')
    analyze_parser.add_argument('--no-viz', action='store_true', help='不生成可视化图表')
    
    # 对比两个模型 相关参数
    compare_parser = subparsers.add_parser('compare', help='对比两个模型')
    compare_parser.add_argument('--baseline', '-b', required=True, help='基准模型Profile JSON文件')
    compare_parser.add_argument('--baseline-layers', default=None, help='基准模型LayerInfo JSON文件 (可选)')
    compare_parser.add_argument('--optimized', '-opt', required=True, help='优化模型Profile JSON文件')
    compare_parser.add_argument('--optimized-layers', default=None, help='优化模型LayerInfo JSON文件 (可选)')
    compare_parser.add_argument('--baseline-name', default='', help='基准模型名称')
    compare_parser.add_argument('--optimized-name', default='', help='优化模型名称')
    compare_parser.add_argument('--output', '-o', default='./analysis_output', help='输出目录')
    compare_parser.add_argument('--report', '-r', action='store_true', help='生成对比报告')
    compare_parser.add_argument('--interactive', '-i', action='store_true', help='生成交互式图表')
    compare_parser.add_argument('--no-viz', action='store_true', help='不生成可视化图表')
    
    # 直接分析engine 相关参数
    profile_parser = subparsers.add_parser('profile', help='直接对TensorRT engine进行分析')
    profile_parser.add_argument('--engine', '-e', required=True, help='TensorRT engine文件')
    profile_parser.add_argument('--name', '-n', default='', help='模型名称')
    profile_parser.add_argument('--output', '-o', default='./analysis_output', help='输出目录')
    profile_parser.add_argument('--warmup', type=int, default=100, help='预热迭代次数')
    profile_parser.add_argument('--iterations', type=int, default=1000, help='测试迭代次数')
    profile_parser.add_argument('--no-viz', action='store_true', help='不生成可视化图表')
    
   
    args = parser.parse_args()
    
    print_banner()
    
    if args.command == 'analyze':
        analyze_single_model(args)
    elif args.command == 'compare':
        compare_models(args)
    elif args.command == 'profile':
        run_trtexec_analysis(args)
    else:
        parser.print_help()



if __name__ == "__main__":
    main()  