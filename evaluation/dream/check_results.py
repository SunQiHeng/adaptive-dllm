#!/usr/bin/env python3
"""
实时检查评估结果脚本
计算并对比 standard, sparse, adaptive 三种模型的准确率
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def find_latest_result_file(task_dir):
    """找到任务目录下最新的结果文件"""
    if not task_dir.exists():
        return None
    
    json_files = list(task_dir.glob('results_*.json'))
    if not json_files:
        return None
    
    return max(json_files, key=lambda p: p.stat().st_mtime)

def extract_metrics(result_file, task_name):
    """从结果文件中提取主要指标"""
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        if 'results' not in data:
            return None
        
        # 获取任务的主要结果
        task_results = data['results'].get(task_name, {})
        if not task_results:
            return None
        
        metrics = {}
        for key, value in task_results.items():
            # 过滤掉包含 _stderr 的指标和 alias
            if '_stderr' not in key and key != 'alias':
                metrics[key] = value
        
        return metrics
    except Exception as e:
        return {'error': str(e)}

def get_all_tasks():
    """获取所有任务列表"""
    results_dir = Path('/home/qiheng/Projects/adaptive-dllm/evaluation/dream/results')
    tasks = set()
    
    for model_type in ['standard', 'sparse', 'adaptive']:
        model_dir = results_dir / model_type
        if model_dir.exists():
            for task_dir in model_dir.iterdir():
                if task_dir.is_dir():
                    tasks.add(task_dir.name)
    
    return sorted(tasks)

def collect_results():
    """收集所有模型的结果"""
    results_dir = Path('/home/qiheng/Projects/adaptive-dllm/evaluation/dream/results')
    tasks = get_all_tasks()
    
    all_results = defaultdict(lambda: {
        'standard': None,
        'sparse': None,
        'adaptive': None
    })
    
    for task in tasks:
        for model_type in ['standard', 'sparse', 'adaptive']:
            task_dir = results_dir / model_type / task
            result_file = find_latest_result_file(task_dir)
            
            if result_file:
                metrics = extract_metrics(result_file, task)
                if metrics and 'error' not in metrics:
                    all_results[task][model_type] = {
                        'file': result_file.name,
                        'metrics': metrics
                    }
    
    return all_results

def format_metric_value(value):
    """格式化指标值"""
    if isinstance(value, (int, float)):
        if abs(value) < 0.01:
            return f"{value:.2e}"
        else:
            return f"{value:.4f}"
    return str(value)

def print_results_table(all_results):
    """打印结果对比表"""
    print("\n" + "=" * 120)
    print(f"评估结果对比 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 120)
    
    # 表头
    print(f"\n{'任务':<20} {'指标':<20} {'Standard':<15} {'Sparse':<15} {'Adaptive':<15} {'Adaptive vs Sparse':<20}")
    print("-" * 120)
    
    for task in sorted(all_results.keys()):
        task_results = all_results[task]
        
        # 确定要显示的指标（优先显示 acc, acc_norm, exact_match 等）
        all_metrics = set()
        for model_type in ['standard', 'sparse', 'adaptive']:
            if task_results[model_type] and 'metrics' in task_results[model_type]:
                all_metrics.update(task_results[model_type]['metrics'].keys())
        
        # 按优先级排序指标
        priority_metrics = ['acc,none', 'acc_norm,none', 'exact_match,strict-match', 
                           'exact_match,flexible-extract', 'acc']
        sorted_metrics = sorted(all_metrics, key=lambda x: (
            priority_metrics.index(x) if x in priority_metrics else 999,
            x
        ))
        
        for metric in sorted_metrics:
            standard_val = None
            sparse_val = None
            adaptive_val = None
            
            if task_results['standard'] and 'metrics' in task_results['standard']:
                standard_val = task_results['standard']['metrics'].get(metric)
            if task_results['sparse'] and 'metrics' in task_results['sparse']:
                sparse_val = task_results['sparse']['metrics'].get(metric)
            if task_results['adaptive'] and 'metrics' in task_results['adaptive']:
                adaptive_val = task_results['adaptive']['metrics'].get(metric)
            
            # 计算对比
            comparison = ""
            if sparse_val is not None and adaptive_val is not None:
                try:
                    diff = adaptive_val - sparse_val
                    if isinstance(sparse_val, (int, float)) and sparse_val != 0:
                        pct_change = (diff / sparse_val) * 100
                        if diff > 0:
                            comparison = f"✅ +{format_metric_value(diff)} ({pct_change:+.2f}%)"
                        elif diff < 0:
                            comparison = f"❌ {format_metric_value(diff)} ({pct_change:.2f}%)"
                        else:
                            comparison = "= (相同)"
                    else:
                        comparison = f"diff: {format_metric_value(diff)}"
                except:
                    comparison = "N/A"
            
            standard_str = format_metric_value(standard_val) if standard_val is not None else "N/A"
            sparse_str = format_metric_value(sparse_val) if sparse_val is not None else "N/A"
            adaptive_str = format_metric_value(adaptive_val) if adaptive_val is not None else "N/A"
            
            print(f"{task:<20} {metric:<20} {standard_str:<15} {sparse_str:<15} {adaptive_str:<15} {comparison:<20}")
        
        print("-" * 120)
    
    print("\n" + "=" * 120)

def print_summary(all_results):
    """打印摘要统计"""
    print("\n" + "=" * 120)
    print("任务完成情况摘要")
    print("=" * 120)
    
    tasks = sorted(all_results.keys())
    completed = {
        'standard': 0,
        'sparse': 0,
        'adaptive': 0
    }
    
    for task in tasks:
        for model_type in ['standard', 'sparse', 'adaptive']:
            if all_results[task][model_type] is not None:
                completed[model_type] += 1
    
    total_tasks = len(tasks)
    print(f"\n总任务数: {total_tasks}")
    print(f"Standard 完成: {completed['standard']}/{total_tasks} ({completed['standard']/total_tasks*100:.1f}%)")
    print(f"Sparse 完成:    {completed['sparse']}/{total_tasks} ({completed['sparse']/total_tasks*100:.1f}%)")
    print(f"Adaptive 完成:  {completed['adaptive']}/{total_tasks} ({completed['adaptive']/total_tasks*100:.1f}%)")
    print("=" * 120 + "\n")

def main():
    """主函数"""
    results_dir = Path('/home/qiheng/Projects/adaptive-dllm/evaluation/dream/results')
    
    if not results_dir.exists():
        print(f"❌ 结果目录不存在: {results_dir}")
        return
    
    print(f"📂 扫描结果目录: {results_dir}")
    all_results = collect_results()
    
    if not all_results:
        print("⚠️  未找到任何结果文件")
        return
    
    print_summary(all_results)
    print_results_table(all_results)

if __name__ == '__main__':
    main()

