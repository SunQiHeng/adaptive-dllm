#!/usr/bin/env python3
"""
Compare results across three attention types for a single task
"""
import json
import sys
from pathlib import Path


def load_result(result_path):
    """Load result from JSON file"""
    try:
        with open(result_path, 'r') as f:
            data = json.load(f)
        return data.get('results', {})
    except Exception as e:
        print(f"Error loading {result_path}: {e}")
        return None


def extract_main_metric(task_results, task_name):
    """Extract the main metric from task results"""
    if not task_results:
        return None
    
    # Common metric mappings
    metric_priority = [
        'acc', 'acc_norm', 'exact_match', 'pass@1', 'f1', 
        'bleu', 'rouge', 'em', 'accuracy'
    ]
    
    # First try to find task-specific metrics
    for key, value in task_results.items():
        if isinstance(value, dict):
            for metric in metric_priority:
                if metric in value:
                    return {
                        'name': f"{key}.{metric}",
                        'value': value[metric]
                    }
    
    # If not found, try direct metrics
    for metric in metric_priority:
        if metric in task_results:
            return {
                'name': metric,
                'value': task_results[metric]
            }
    
    # Return first numeric value found
    for key, value in task_results.items():
        if isinstance(value, (int, float)):
            return {
                'name': key,
                'value': value
            }
    
    return None


def compare_task_results(task_name, base_dir="results/compare"):
    """Compare results for a specific task"""
    base_path = Path(base_dir) / task_name
    
    if not base_path.exists():
        print(f"Error: Task directory not found: {base_path}")
        print(f"Make sure you've run: sbatch run_compare_all.slurm {task_name}")
        return
    
    print("=" * 80)
    print(f"Comparison Results: {task_name.upper()}")
    print("=" * 80)
    print()
    
    model_types = ['standard', 'sparse', 'adaptive']
    results = {}
    all_metrics = set()
    
    # Load all results
    for model_type in model_types:
        result_file = base_path / model_type / "results.json"
        if result_file.exists():
            result_data = load_result(result_file)
            results[model_type] = result_data
            
            # Collect all metric names
            if result_data:
                for task_key, task_metrics in result_data.items():
                    if isinstance(task_metrics, dict):
                        for metric_name in task_metrics.keys():
                            all_metrics.add(f"{task_key}.{metric_name}")
        else:
            print(f"Warning: Results not found for {model_type}")
            results[model_type] = None
    
    if not any(results.values()):
        print("No results found!")
        return
    
    # Print main metric comparison
    print("Main Metrics:")
    print("-" * 80)
    print(f"{'Model Type':<15} {'Metric':<30} {'Value':<15} {'Relative':<15}")
    print("-" * 80)
    
    main_metrics = {}
    for model_type in model_types:
        if results[model_type]:
            metric_info = extract_main_metric(results[model_type], task_name)
            if metric_info:
                main_metrics[model_type] = metric_info
    
    # Calculate baseline (standard)
    baseline_value = main_metrics.get('standard', {}).get('value')
    
    for model_type in model_types:
        if model_type in main_metrics:
            info = main_metrics[model_type]
            value = info['value']
            
            if baseline_value is not None and baseline_value != 0:
                relative = (value - baseline_value) / baseline_value * 100
                relative_str = f"{relative:+.2f}%"
            else:
                relative_str = "-"
            
            print(f"{model_type:<15} {info['name']:<30} {value:<15.4f} {relative_str:<15}")
        else:
            print(f"{model_type:<15} {'N/A':<30} {'-':<15} {'-':<15}")
    
    print()
    
    # Print detailed comparison
    print("Detailed Metrics:")
    print("-" * 80)
    
    # Get all unique task keys
    all_task_keys = set()
    for model_type, result_data in results.items():
        if result_data:
            all_task_keys.update(result_data.keys())
    
    for task_key in sorted(all_task_keys):
        print(f"\n{task_key}:")
        print(f"  {'Metric':<35} {'Standard':<12} {'Sparse':<12} {'Adaptive':<12}")
        print(f"  {'-'*71}")
        
        # Collect all metrics for this task
        task_metrics = set()
        for model_type in model_types:
            if results[model_type] and task_key in results[model_type]:
                task_data = results[model_type][task_key]
                if isinstance(task_data, dict):
                    task_metrics.update(task_data.keys())
        
        for metric_name in sorted(task_metrics):
            row = f"  {metric_name:<35}"
            
            for model_type in model_types:
                if results[model_type] and task_key in results[model_type]:
                    task_data = results[model_type][task_key]
                    if isinstance(task_data, dict) and metric_name in task_data:
                        value = task_data[metric_name]
                        if isinstance(value, float):
                            row += f"{value:<12.4f}"
                        else:
                            row += f"{str(value):<12}"
                    else:
                        row += f"{'-':<12}"
                else:
                    row += f"{'-':<12}"
            
            print(row)
    
    print()
    print("=" * 80)
    
    # Summary statistics
    if len(main_metrics) == 3:
        standard_val = main_metrics['standard']['value']
        sparse_val = main_metrics['sparse']['value']
        adaptive_val = main_metrics['adaptive']['value']
        
        print("\nSummary:")
        print(f"  Standard (baseline): {standard_val:.4f}")
        print(f"  Sparse vs Standard: {(sparse_val - standard_val) / standard_val * 100:+.2f}%")
        print(f"  Adaptive vs Standard: {(adaptive_val - standard_val) / standard_val * 100:+.2f}%")
        print(f"  Adaptive vs Sparse: {(adaptive_val - sparse_val) / sparse_val * 100:+.2f}%")
    
    print("=" * 80)


def list_available_tasks(base_dir="results/compare"):
    """List all available task comparisons"""
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"No comparison results found in {base_dir}")
        return []
    
    tasks = [d.name for d in base_path.iterdir() if d.is_dir()]
    return sorted(tasks)


def main():
    # Check for debug flag
    debug_mode = False
    if '--debug' in sys.argv:
        debug_mode = True
        sys.argv.remove('--debug')
    
    if len(sys.argv) < 2:
        print("Usage: python compare_results.py [--debug] <task_name>")
        print()
        
        base_dir = "results/debug" if debug_mode else "results/compare"
        tasks = list_available_tasks(base_dir)
        if tasks:
            print(f"Available tasks ({base_dir}):")
            for task in tasks:
                print(f"  - {task}")
            print()
            print("Example: python compare_results.py gsm8k")
            print("         python compare_results.py --debug gsm8k")
        else:
            print("No comparison results found yet.")
            if debug_mode:
                print("Run: ./run_compare_quick.sh gsm8k 10")
            else:
                print("Run: sbatch run_compare_all.slurm gsm8k")
        
        sys.exit(1)
    
    task_name = sys.argv[1]
    base_dir = "results/debug" if debug_mode else "results/compare"
    compare_task_results(task_name, base_dir)


if __name__ == "__main__":
    main()

