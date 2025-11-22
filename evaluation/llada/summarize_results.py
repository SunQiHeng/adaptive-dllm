#!/usr/bin/env python3
"""
Summarize evaluation results from all model types
"""
import json
import os
from pathlib import Path
from collections import defaultdict


def load_results(results_dir):
    """Load all results from the results directory"""
    results = defaultdict(dict)
    
    for model_type in ['standard', 'sparse', 'adaptive']:
        model_dir = Path(results_dir) / model_type
        if not model_dir.exists():
            continue
        
        for task_dir in model_dir.iterdir():
            if not task_dir.is_dir():
                continue
            
            result_file = task_dir / "results.json"
            if result_file.exists():
                try:
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                    
                    task_name = task_dir.name
                    results[task_name][model_type] = data.get('results', {})
                except Exception as e:
                    print(f"Error loading {result_file}: {e}")
    
    return results


def extract_metrics(task_results):
    """Extract main metrics from task results"""
    metrics = {}
    
    for key, value in task_results.items():
        if isinstance(value, dict):
            for metric_name, metric_value in value.items():
                if isinstance(metric_value, (int, float)):
                    metrics[f"{key}_{metric_name}"] = metric_value
        elif isinstance(value, (int, float)):
            metrics[key] = value
    
    return metrics


def print_summary_table(results):
    """Print a formatted summary table"""
    print("\n" + "="*100)
    print("LLaDA Evaluation Summary")
    print("="*100)
    
    # Get all tasks
    tasks = sorted(results.keys())
    
    for task in tasks:
        print(f"\n{task.upper()}")
        print("-" * 100)
        
        # Get all metrics for this task
        all_metrics = set()
        for model_type in ['standard', 'sparse', 'adaptive']:
            if model_type in results[task]:
                metrics = extract_metrics(results[task][model_type])
                all_metrics.update(metrics.keys())
        
        all_metrics = sorted(all_metrics)
        
        # Print header
        print(f"{'Metric':<40} {'Standard':<20} {'Sparse':<20} {'Adaptive':<20}")
        print("-" * 100)
        
        # Print each metric
        for metric in all_metrics:
            row = f"{metric:<40}"
            
            for model_type in ['standard', 'sparse', 'adaptive']:
                if model_type in results[task]:
                    metrics = extract_metrics(results[task][model_type])
                    value = metrics.get(metric, 'N/A')
                    if isinstance(value, float):
                        row += f"{value:<20.4f}"
                    else:
                        row += f"{str(value):<20}"
                else:
                    row += f"{'N/A':<20}"
            
            print(row)
    
    print("\n" + "="*100)


def save_csv(results, output_file):
    """Save results to CSV"""
    import csv
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Task', 'Metric', 'Standard', 'Sparse', 'Adaptive'])
        
        for task in sorted(results.keys()):
            all_metrics = set()
            for model_type in ['standard', 'sparse', 'adaptive']:
                if model_type in results[task]:
                    metrics = extract_metrics(results[task][model_type])
                    all_metrics.update(metrics.keys())
            
            for metric in sorted(all_metrics):
                row = [task, metric]
                for model_type in ['standard', 'sparse', 'adaptive']:
                    if model_type in results[task]:
                        metrics = extract_metrics(results[task][model_type])
                        value = metrics.get(metric, 'N/A')
                        row.append(value)
                    else:
                        row.append('N/A')
                writer.writerow(row)


def main():
    results_dir = Path(__file__).parent / "results"
    
    print(f"Loading results from: {results_dir}")
    results = load_results(results_dir)
    
    if not results:
        print("No results found!")
        return
    
    # Print summary
    print_summary_table(results)
    
    # Save to CSV
    csv_file = results_dir / "summary.csv"
    save_csv(results, csv_file)
    print(f"\nResults also saved to: {csv_file}")


if __name__ == "__main__":
    main()

