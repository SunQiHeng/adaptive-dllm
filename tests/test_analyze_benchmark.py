"""
Analyze and visualize benchmark results.
"""

import json
import argparse
import numpy as np
from typing import Dict, List


def load_results(filepath: str) -> Dict:
    """Load benchmark results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def print_detailed_results(data: Dict):
    """Print detailed results for each method."""
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)
    
    config = data['config']
    print(f"\nConfiguration:")
    print(f"  Model: {config['model_path']}")
    print(f"  Samples: {config['num_samples']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Sequence length: {config['seq_len']}")
    print(f"  Steps: {config['steps']}")
    
    results = data['results']
    
    for i, result in enumerate(results, 1):
        print(f"\n{'─'*80}")
        print(f"Method {i}: {result['method']}")
        print(f"{'─'*80}")
        print(f"  Total time:           {result['total_time']:.3f}s")
        print(f"  Avg time per batch:   {result['avg_time_per_batch']:.3f}s ± {result['std_time_per_batch']:.3f}s")
        print(f"  Avg time per sample:  {result['avg_time_per_sample']:.3f}s")
        print(f"  Throughput:           {result['tokens_per_sec']:.2f} tokens/sec")
        print(f"  Number of batches:    {result['num_batches']}")
        
        # Show per-batch times
        if 'batch_times' in result:
            batch_times = result['batch_times']
            print(f"\n  Per-batch times:")
            for j, t in enumerate(batch_times, 1):
                print(f"    Batch {j}: {t:.3f}s")


def print_comparison(data: Dict):
    """Print comparison table."""
    results = data['results']
    
    if len(results) < 2:
        print("\nNeed at least 2 methods to compare.")
        return
    
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    
    # Use first method as baseline
    baseline = results[0]
    baseline_time = baseline['avg_time_per_sample']
    baseline_throughput = baseline['tokens_per_sec']
    
    print(f"\nBaseline: {baseline['method']}")
    print(f"  Time per sample: {baseline_time:.3f}s")
    print(f"  Throughput: {baseline_throughput:.2f} tokens/sec")
    
    print(f"\n{'Method':<45} {'Time/Sample':<15} {'Speedup':<12} {'Throughput':<15} {'Improvement'}")
    print("─"*110)
    
    for result in results:
        time_per_sample = result['avg_time_per_sample']
        throughput = result['tokens_per_sec']
        speedup = baseline_time / time_per_sample
        throughput_improvement = (throughput - baseline_throughput) / baseline_throughput * 100
        
        method_name = result['method']
        if len(method_name) > 44:
            method_name = method_name[:41] + "..."
        
        print(f"{method_name:<45} "
              f"{time_per_sample:>7.3f}s        "
              f"{speedup:>6.2f}x       "
              f"{throughput:>8.2f} tok/s  "
              f"{throughput_improvement:>+6.1f}%")


def print_statistics(data: Dict):
    """Print statistical analysis."""
    results = data['results']
    
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)
    
    if len(results) < 2:
        return
    
    times = [r['avg_time_per_sample'] for r in results]
    throughputs = [r['tokens_per_sec'] for r in results]
    
    print(f"\nTime per sample:")
    print(f"  Min:    {min(times):.3f}s ({results[np.argmin(times)]['method']})")
    print(f"  Max:    {max(times):.3f}s ({results[np.argmax(times)]['method']})")
    print(f"  Range:  {max(times) - min(times):.3f}s")
    print(f"  Ratio:  {max(times) / min(times):.2f}x")
    
    print(f"\nThroughput (tokens/sec):")
    print(f"  Min:    {min(throughputs):.2f} tok/s ({results[np.argmin(throughputs)]['method']})")
    print(f"  Max:    {max(throughputs):.2f} tok/s ({results[np.argmax(throughputs)]['method']})")
    print(f"  Range:  {max(throughputs) - min(throughputs):.2f} tok/s")
    print(f"  Ratio:  {max(throughputs) / min(throughputs):.2f}x")


def print_efficiency_analysis(data: Dict):
    """Analyze efficiency of sparse methods."""
    results = data['results']
    
    print("\n" + "="*80)
    print("EFFICIENCY ANALYSIS")
    print("="*80)
    
    # Find dense method
    dense_result = None
    sparse_results = []
    
    for result in results:
        method_lower = result['method'].lower()
        if 'dense' in method_lower:
            dense_result = result
        elif 'sparse' in method_lower or 'adaptive' in method_lower:
            sparse_results.append(result)
    
    if dense_result and sparse_results:
        dense_time = dense_result['avg_time_per_sample']
        dense_throughput = dense_result['tokens_per_sec']
        
        print(f"\nDense baseline:")
        print(f"  Time per sample: {dense_time:.3f}s")
        print(f"  Throughput: {dense_throughput:.2f} tokens/sec")
        
        print(f"\nSparse methods efficiency:")
        print(f"{'Method':<45} {'Time Saved':<15} {'Speedup':<12} {'Efficiency'}")
        print("─"*90)
        
        for result in sparse_results:
            time_saved = dense_time - result['avg_time_per_sample']
            speedup = dense_time / result['avg_time_per_sample']
            time_saved_pct = time_saved / dense_time * 100
            
            # Assume 30% keep ratio → 70% sparsity
            # Ideal speedup would be proportional to sparsity
            # Efficiency = actual_speedup / theoretical_speedup
            theoretical_speedup = 1.0 / 0.3  # If 30% keep, ideal is 3.33x
            efficiency = (speedup - 1.0) / (theoretical_speedup - 1.0) * 100
            
            method_name = result['method']
            if len(method_name) > 44:
                method_name = method_name[:41] + "..."
            
            print(f"{method_name:<45} "
                  f"{time_saved:>+6.3f}s ({time_saved_pct:>+5.1f}%)  "
                  f"{speedup:>5.2f}x       "
                  f"{efficiency:>5.1f}%")
        
        print(f"\nNote: Efficiency = (actual_speedup - 1) / (theoretical_speedup - 1)")
        print(f"      Theoretical speedup ≈ 1/keep_ratio = 1/0.3 ≈ 3.33x for 30% keep")


def save_markdown_report(data: Dict, output_file: str):
    """Save results as markdown report."""
    results = data['results']
    config = data['config']
    
    with open(output_file, 'w') as f:
        f.write("# Attention Methods Benchmark Results\n\n")
        
        f.write("## Configuration\n\n")
        f.write(f"- Model: `{config['model_path']}`\n")
        f.write(f"- Samples: {config['num_samples']}\n")
        f.write(f"- Batch size: {config['batch_size']}\n")
        f.write(f"- Sequence length: {config['seq_len']}\n")
        f.write(f"- Steps: {config['steps']}\n")
        f.write(f"- Seed: {config['seed']}\n\n")
        
        f.write("## Results Summary\n\n")
        f.write("| Method | Time/Sample | Throughput | Speedup |\n")
        f.write("|--------|-------------|------------|---------|\n")
        
        baseline_time = results[0]['avg_time_per_sample']
        for result in results:
            speedup = baseline_time / result['avg_time_per_sample']
            f.write(f"| {result['method']} | {result['avg_time_per_sample']:.3f}s | "
                   f"{result['tokens_per_sec']:.2f} tok/s | {speedup:.2f}x |\n")
        
        f.write("\n## Detailed Results\n\n")
        for i, result in enumerate(results, 1):
            f.write(f"### {i}. {result['method']}\n\n")
            f.write(f"- Total time: {result['total_time']:.3f}s\n")
            f.write(f"- Avg time per batch: {result['avg_time_per_batch']:.3f}s "
                   f"± {result['std_time_per_batch']:.3f}s\n")
            f.write(f"- Avg time per sample: {result['avg_time_per_sample']:.3f}s\n")
            f.write(f"- Throughput: {result['tokens_per_sec']:.2f} tokens/sec\n")
            f.write(f"- Number of batches: {result['num_batches']}\n\n")
    
    print(f"\n✓ Markdown report saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze benchmark results')
    parser.add_argument('--input', type=str, default='benchmark_results_all.json',
                       help='Input JSON file with benchmark results')
    parser.add_argument('--markdown', type=str, default='benchmark_report.md',
                       help='Output markdown report file')
    parser.add_argument('--no-detailed', action='store_true',
                       help='Skip detailed per-batch results')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.input}...")
    data = load_results(args.input)
    
    # Print analyses
    if not args.no_detailed:
        print_detailed_results(data)
    
    print_comparison(data)
    print_statistics(data)
    print_efficiency_analysis(data)
    
    # Save markdown report
    save_markdown_report(data, args.markdown)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()

