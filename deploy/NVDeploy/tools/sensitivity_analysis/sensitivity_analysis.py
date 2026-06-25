#!/usr/bin/env python3
"""
Layer-wise sensitivity analysis between two TensorRT engines.
配置优化：默认只使用余弦相似度，可选启用其他指标
"""

import numpy as np
import json
import argparse
from collections import OrderedDict

from polygraphy.backend.trt import TrtRunner, EngineFromBytes
from polygraphy.backend.common import BytesFromPath
from polygraphy.comparator import Comparator
import os


class EngineSensitivityAnalyzer:
    def __init__(self, engine1_path, engine2_path, config=None):
        self.engine1_path = engine1_path
        self.engine2_path = engine2_path
        self.config = config or {
            'use_cosine': True,
            'use_l2': False,
            'use_mse': False,
            'use_psr': False,
            'use_pearson': False,
            'max_outputs': None,  # None表示不限制
            'skip_perturbation': True  # 跳过扰动分析
        }
    
    def get_engine_layer_outputs(self, engine_path, input_data):
        """Run inference and extract layer-wise outputs from a TensorRT engine using profiling."""
        layer_outputs = OrderedDict()
        try:
            with TrtRunner(EngineFromBytes(BytesFromPath(engine_path))) as runner:
                # Enable profiling to get layer-wise information
                runner.context.profiler = self.create_profiler()

                # Run inference
                outputs = runner.infer(input_data)

                # Get layer information from the engine
                engine = runner.engine
                for layer in engine:
                    layer_name = f"{layer.type.name}.{layer.name}" if layer.name else f"{layer.type.name}"
                    # Try to get layer outputs (this is a simplified approach)
                    # In practice, you might need to modify the engine or use TensorRT's debugging features
                    for j in range(layer.num_outputs):
                        output_tensor = layer.get_output(j)
                        if output_tensor:
                            tensor_name = f"{layer_name}.output.{j}"
                            # Note: This is a conceptual approach - actual implementation may vary
                            layer_outputs[tensor_name] = None  # Placeholder
        except Exception as e:
            print(f"Error extracting layer outputs from {engine_path}: {e}")
        return layer_outputs

    def create_profiler(self):
        """Create a simple profiler for TensorRT."""
        class SimpleProfiler:
            def __init__(self):
                self.layer_times = {}

            def report_layer_time(self, layer_name, time_ms):
                self.layer_times[layer_name] = time_ms

        return SimpleProfiler()

    def create_realistic_input_data(self, engine_path):
        """Create realistic input data based on engine requirements."""
        import tensorrt as trt

        # Load engine to get input specifications
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(engine_data)

        input_data = {}
        print("Detected input shapes:", end=" ")
        for i in range(engine.num_bindings):
            if engine.binding_is_input(i):
                name = engine.get_binding_name(i)
                shape = tuple(engine.get_binding_shape(i))
                dtype = engine.get_binding_dtype(i)
                print(f"{name}: {shape}, {dtype}", end="; ")
                if name == "images":
                    # Create realistic image data (normalized)
                    data = np.random.uniform(0, 1, shape).astype(np.float32)
                elif "target_sizes" in name:
                    # Create realistic target sizes as int32
                    data = np.array([[640, 640]], dtype=np.int32)
                else:
                    # Default: create data based on detected dtype
                    if dtype == trt.DataType.INT32:
                        data = np.random.randint(0, 1000, shape).astype(np.int32)
                    elif dtype == trt.DataType.FLOAT:
                        data = np.random.rand(*shape).astype(np.float32)
                    else:
                        data = np.random.randn(*shape).astype(np.float32)
                input_data[name] = data
        print()
        return input_data

    def compare_engines_outputs(self, input_data):
        """Compare final outputs between two engines with configurable metrics."""
        print("Comparing final outputs between engines...")

        # Get outputs from both engines
        with TrtRunner(EngineFromBytes(BytesFromPath(self.engine1_path))) as runner1:
            outputs1 = runner1.infer(input_data)
        with TrtRunner(EngineFromBytes(BytesFromPath(self.engine2_path))) as runner2:
            outputs2 = runner2.infer(input_data)

        # 限制输出数量
        if self.config['max_outputs'] and len(outputs1) > self.config['max_outputs']:
            print(f"Limiting comparison to {self.config['max_outputs']} outputs")
            output_names = list(outputs1.keys())[:self.config['max_outputs']]
        else:
            output_names = list(outputs1.keys())
        
        # Compare outputs
        comparison_results = {}
        for output_name in output_names:
            if output_name in outputs2:
                metrics = self.calculate_metrics(outputs1[output_name], outputs2[output_name])
                comparison_results[output_name] = metrics
                
                # 只打印余弦相似度低的输出
                if metrics['cosine_similarity'] < 0.99 or not self.config['use_cosine']:
                    print(f"Output: {output_name}")
                    if self.config['use_cosine']:
                        print(f"  Cosine Similarity: {metrics['cosine_similarity']:.6f}")
                    if self.config['use_l2']:
                        print(f"  L2 Distance: {metrics['l2_distance']:.6f}")
                    if self.config['use_mse']:
                        print(f"  MSE: {metrics['mse']:.6f}")
                    if self.config['use_psr']:
                        print(f"  PSR: {metrics['psr']:.6f}")
                    if self.config['use_pearson']:
                        print(f"  Pearson Correlation: {metrics['pearson_correlation']:.6f}")

        return comparison_results

    def calculate_metrics(self, array1, array2):
        """Calculate configured similarity metrics between two arrays."""
        # Flatten arrays
        a1_flat = array1.flatten().astype(np.float32)
        a2_flat = array2.flatten().astype(np.float32)

        # Handle shape mismatch by taking minimum size
        min_size = min(a1_flat.size, a2_flat.size)
        a1_flat = a1_flat[:min_size]
        a2_flat = a2_flat[:min_size]

        # Initialize metrics with default values
        metrics = {}
        
        # 只计算需要的指标
        if self.config['use_cosine']:
            metrics['cosine_similarity'] = self._calculate_cosine_similarity(a1_flat, a2_flat, min_size)
        
        if self.config['use_l2']:
            metrics['l2_distance'] = self._calculate_l2_distance(a1_flat, a2_flat, min_size)
        
        if self.config['use_mse']:
            metrics['mse'] = self._calculate_mse(a1_flat, a2_flat, min_size)
        
        if self.config['use_psr']:
            metrics['psr'] = self._calculate_psr(a1_flat, a2_flat, min_size)
        
        if self.config['use_pearson']:
            metrics['pearson_correlation'] = self._calculate_pearson_correlation(a1_flat, a2_flat, min_size)
        
        return metrics

    def _calculate_cosine_similarity(self, a1_flat, a2_flat, min_size):
        """Calculate cosine similarity."""
        if min_size < 100:
            return 0.0
        
        dot_product = np.dot(a1_flat, a2_flat)
        norm1 = np.linalg.norm(a1_flat)
        norm2 = np.linalg.norm(a2_flat)
        if norm1 > 0 and norm2 > 0:
            return float(dot_product / (norm1 * norm2))
        else:
            return 0.0

    def _calculate_l2_distance(self, a1_flat, a2_flat, min_size):
        """Calculate L2 distance."""
        if min_size < 100:
            return float('inf')
        
        norm1 = np.linalg.norm(a1_flat)
        norm2 = np.linalg.norm(a2_flat)
        l2_dist = np.linalg.norm(a1_flat - a2_flat) / (norm1 + norm2 + 1e-8)
        return float(l2_dist)

    def _calculate_mse(self, a1_flat, a2_flat, min_size):
        """Calculate mean squared error."""
        if min_size < 100:
            return float('inf')
        
        mse = float(np.mean((a1_flat - a2_flat) ** 2))
        return mse

    def _calculate_psr(self, a1_flat, a2_flat, min_size):
        """Calculate peak signal-to-noise ratio."""
        if min_size < 100:
            return float('inf')
        
        mse = float(np.mean((a1_flat - a2_flat) ** 2))
        max_val = max(np.max(a1_flat), np.max(a2_flat))
        if mse > 0:
            psr = 10 * np.log10((max_val ** 2) / mse)
        else:
            psr = float('inf')
        return float(psr)

    def _calculate_pearson_correlation(self, a1_flat, a2_flat, min_size):
        """Calculate Pearson correlation coefficient."""
        if min_size < 2:
            return 0.0
        
        try:
            corr_matrix = np.corrcoef(a1_flat, a2_flat)
            pearson_corr = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
        except:
            pearson_corr = 0.0
        return pearson_corr

    def analyze_sensitivity(self, input_data=None, cosine_threshold=0.95, num_perturbations=10):
        """Perform complete sensitivity analysis between two engines."""
        print("Starting engine sensitivity analysis...")
        print(f"Configuration: {self.config}")

        # Prepare input data
        if input_data is None:
            print("Creating realistic input data based on engine requirements...")
            input_data = self.create_realistic_input_data(self.engine1_path)
        else:
            print(f"Using input data with shapes: {[(k, v.shape) for k, v in input_data.items()]}")

        # Compare final outputs
        print("\n=== Final Outputs Comparison ===")
        final_metrics = self.compare_engines_outputs(input_data)

        # Skip perturbation analysis if configured
        sensitivity_analysis = {}
        if not self.config['skip_perturbation']:
            print("\n=== Layer Sensitivity Analysis via Input Perturbation ===")
            sensitivity_analysis = self.analyze_layer_sensitivity_via_perturbation(input_data, num_perturbations)
        else:
            print("\n=== Skipping perturbation analysis (configured to skip) ===")

        # Find sensitive outputs (where cosine similarity is below threshold)
        sensitive_outputs = []
        for output_name, metrics in final_metrics.items():
            if 'cosine_similarity' in metrics and metrics['cosine_similarity'] < cosine_threshold:
                sensitive_outputs.append({
                    'name': output_name,
                    'cosine_similarity': metrics['cosine_similarity']
                })

        # Print summary
        print(f"\nFound {len(sensitive_outputs)} sensitive outputs out of {len(final_metrics)} analyzed.")
        
        if sensitive_outputs and self.config['use_cosine']:
            print("Sensitive outputs (cosine similarity < threshold):")
            sensitive_outputs.sort(key=lambda x: x['cosine_similarity'])
            for i, output in enumerate(sensitive_outputs[:10]):  # Show top 10
                print(f"  {i+1}. {output['name']}: {output['cosine_similarity']:.6f}")

        # Save detailed results
        results = {
            'engine1_path': self.engine1_path,
            'engine2_path': self.engine2_path,
            'final_outputs_comparison': final_metrics,
            'layer_sensitivity_analysis': sensitivity_analysis,
            'sensitive_outputs': sensitive_outputs,
            'cosine_threshold': cosine_threshold,
            'num_perturbations': num_perturbations,
            'config': self.config
        }

        with open('layer_precision_sensitivity_analysis.json', 'w') as f:
            json.dump(results, f, indent=2)

        print("\nAnalysis complete! Results saved to 'layer_precision_sensitivity_analysis.json'.")
        return results


def main():
    parser = argparse.ArgumentParser(description='Engine-to-engine sensitivity analysis')
    parser.add_argument('--engine1', required=True, help='Path to first TensorRT engine')
    parser.add_argument('--engine2', required=True, help='Path to second TensorRT engine')
    parser.add_argument('--cosine-threshold', type=float, default=0.95,
                        help='Cosine similarity threshold for sensitivity (default: 0.95)')
    parser.add_argument('--num-perturbations', type=int, default=10,
                        help='Number of input perturbations for sensitivity analysis (default: 10)')
    
    # 配置参数
    parser.add_argument('--use-cosine', action='store_true', default=True,
                        help='Use cosine similarity (default: True)')
    parser.add_argument('--no-cosine', action='store_false', dest='use_cosine',
                        help='Disable cosine similarity')
    
    parser.add_argument('--use-l2', action='store_true', default=False,
                        help='Use L2 distance (default: False)')
    parser.add_argument('--use-mse', action='store_true', default=False,
                        help='Use MSE (default: False)')
    parser.add_argument('--use-psr', action='store_true', default=False,
                        help='Use PSR (default: False)')
    parser.add_argument('--use-pearson', action='store_true', default=False,
                        help='Use Pearson correlation (default: False)')
    
    parser.add_argument('--max-outputs', type=int, default=None,
                        help='Maximum number of outputs to compare (default: all)')
    parser.add_argument('--skip-perturbation', action='store_true', default=True,
                        help='Skip perturbation analysis (default: True)')
    parser.add_argument('--enable-perturbation', action='store_false', dest='skip_perturbation',
                        help='Enable perturbation analysis')
    
    args = parser.parse_args()

    # Check if engine files exist
    if not os.path.exists(args.engine1):
        print(f"Error: Engine file '{args.engine1}' not found!")
        return
    if not os.path.exists(args.engine2):
        print(f"Error: Engine file '{args.engine2}' not found!")
        return

    # Create configuration
    config = {
        'use_cosine': args.use_cosine,
        'use_l2': args.use_l2,
        'use_mse': args.use_mse,
        'use_psr': args.use_psr,
        'use_pearson': args.use_pearson,
        'max_outputs': args.max_outputs,
        'skip_perturbation': args.skip_perturbation
    }

    # Print configuration
    print("="*60)
    print("Sensitivity Analysis Configuration")
    print("="*60)
    print(f"Engine 1: {args.engine1}")
    print(f"Engine 2: {args.engine2}")
    print(f"Metrics: ", end="")
    metrics_enabled = []
    if config['use_cosine']: metrics_enabled.append("Cosine")
    if config['use_l2']: metrics_enabled.append("L2")
    if config['use_mse']: metrics_enabled.append("MSE")
    if config['use_psr']: metrics_enabled.append("PSR")
    if config['use_pearson']: metrics_enabled.append("Pearson")
    print(", ".join(metrics_enabled) if metrics_enabled else "None")
    print(f"Max outputs: {config['max_outputs'] or 'Unlimited'}")
    print(f"Perturbation analysis: {'Disabled' if config['skip_perturbation'] else 'Enabled'}")
    print(f"Cosine threshold: {args.cosine_threshold}")
    print("="*60)

    analyzer = EngineSensitivityAnalyzer(args.engine1, args.engine2, config)
    results = analyzer.analyze_sensitivity(
        cosine_threshold=args.cosine_threshold,
        num_perturbations=args.num_perturbations
    )

    return results


if __name__ == "__main__":
    main()