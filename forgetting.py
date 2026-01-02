#!/usr/bin/env python3
"""
Continual Learning Forgetting Calculator

This script calculates forgetting metrics from continual learning experiment logs.
Forgetting is defined as the difference between the maximum accuracy achieved on a task
and the final accuracy on that task.

Usage:
    python forgetting_calculator.py <log_file_path>
    
Or modify the script to use the provided log content directly.
"""

import re
import numpy as np
import argparse
from typing import List, Tuple, Dict


def parse_accuracy_line(line: str) -> Tuple[List[float], List[float]]:
    """
    Parse a line containing accuracy values for Class-IL and Task-IL.
    
    Expected format: "Raw accuracy values: Class-IL [acc1, acc2, ...] | Task-IL [acc1, acc2, ...]"
    
    Args:
        line: Log line containing accuracy values
        
    Returns:
        Tuple of (class_il_accuracies, task_il_accuracies)
    """
    # Extract Class-IL accuracies
    class_il_match = re.search(r'Class-IL \[([\d\., ]+)\]', line)
    task_il_match = re.search(r'Task-IL \[([\d\., ]+)\]', line)
    
    if not class_il_match or not task_il_match:
        return [], []
    
    # Parse the accuracy values
    class_il_str = class_il_match.group(1)
    task_il_str = task_il_match.group(1)
    
    class_il_accs = [float(x.strip()) for x in class_il_str.split(',') if x.strip()]
    task_il_accs = [float(x.strip()) for x in task_il_str.split(',') if x.strip()]
    
    return class_il_accs, task_il_accs


def extract_accuracies_from_log(log_content: str) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Extract all accuracy measurements from the log content.
    
    Args:
        log_content: Complete log file content
        
    Returns:
        Tuple of (class_il_history, task_il_history) where each is a list of accuracy lists
    """
    lines = log_content.split('\n')
    class_il_history = []
    task_il_history = []
    
    for line in lines:
        if 'Raw accuracy values:' in line:
            class_il_accs, task_il_accs = parse_accuracy_line(line)
            if class_il_accs and task_il_accs:
                class_il_history.append(class_il_accs)
                task_il_history.append(task_il_accs)
    
    return class_il_history, task_il_history


def calculate_forgetting(accuracy_history: List[List[float]]) -> Dict[str, float]:
    """
    Calculate forgetting metrics from accuracy history.
    
    Args:
        accuracy_history: List of accuracy lists, where accuracy_history[i] contains
                         accuracies for tasks 0 to i after completing task i
    
    Returns:
        Dictionary containing forgetting metrics
    """
    if not accuracy_history:
        return {}
    
    num_tasks = len(accuracy_history[-1])
    
    # Track maximum accuracy achieved for each task
    max_accuracies = [0.0] * num_tasks
    
    # Track accuracy after each task for each task
    task_accuracies = []
    
    for task_idx, accs in enumerate(accuracy_history):
        # Update maximum accuracies
        for i, acc in enumerate(accs):
            max_accuracies[i] = max(max_accuracies[i], acc)
        
        # Store current accuracies
        task_accuracies.append(accs.copy())
    
    # Calculate forgetting for each task (excluding the last task which can't be forgotten)
    forgetting_per_task = []
    final_accuracies = accuracy_history[-1]
    
    for task_id in range(num_tasks - 1):  # Exclude last task
        forgetting = max_accuracies[task_id] - final_accuracies[task_id]
        forgetting_per_task.append(forgetting)
    
    # Calculate average forgetting
    avg_forgetting = np.mean(forgetting_per_task) if forgetting_per_task else 0.0
    
    # Calculate other metrics
    final_avg_accuracy = np.mean(final_accuracies)
    
    return {
        'forgetting_per_task': forgetting_per_task,
        'average_forgetting': avg_forgetting,
        'max_accuracies': max_accuracies,
        'final_accuracies': final_accuracies,
        'final_average_accuracy': final_avg_accuracy,
        'num_tasks': num_tasks
    }


def print_forgetting_report(class_il_metrics: Dict, task_il_metrics: Dict):
    """Print a detailed forgetting report."""
    
    print("=" * 80)
    print("CONTINUAL LEARNING FORGETTING ANALYSIS")
    print("=" * 80)
    
    print(f"\nNumber of tasks: {class_il_metrics.get('num_tasks', 0)}")
    
    print("\n" + "-" * 50)
    print("CLASS-INCREMENTAL LEARNING (Class-IL)")
    print("-" * 50)
    
    if class_il_metrics:
        print(f"Average Forgetting: {class_il_metrics['average_forgetting']:.3f}%")
        print(f"Final Average Accuracy: {class_il_metrics['final_average_accuracy']:.3f}%")
        
        print("\nForgetting per task:")
        for i, forgetting in enumerate(class_il_metrics['forgetting_per_task']):
            print(f"  Task {i+1}: {forgetting:.3f}%")
        
        print("\nAccuracy comparison:")
        print("Task | Max Acc | Final Acc | Forgetting")
        print("-" * 40)
        for i in range(len(class_il_metrics['max_accuracies'])):
            if i < len(class_il_metrics['forgetting_per_task']):
                forgetting = class_il_metrics['forgetting_per_task'][i]
                print(f"{i+1:4d} | {class_il_metrics['max_accuracies'][i]:7.3f} | "
                      f"{class_il_metrics['final_accuracies'][i]:9.3f} | {forgetting:10.3f}")
            else:
                print(f"{i+1:4d} | {class_il_metrics['max_accuracies'][i]:7.3f} | "
                      f"{class_il_metrics['final_accuracies'][i]:9.3f} | {'N/A':>10}")
    
    print("\n" + "-" * 50)
    print("TASK-INCREMENTAL LEARNING (Task-IL)")
    print("-" * 50)
    
    if task_il_metrics:
        print(f"Average Forgetting: {task_il_metrics['average_forgetting']:.3f}%")
        print(f"Final Average Accuracy: {task_il_metrics['final_average_accuracy']:.3f}%")
        
        print("\nForgetting per task:")
        for i, forgetting in enumerate(task_il_metrics['forgetting_per_task']):
            print(f"  Task {i+1}: {forgetting:.3f}%")
        
        print("\nAccuracy comparison:")
        print("Task | Max Acc | Final Acc | Forgetting")
        print("-" * 40)
        for i in range(len(task_il_metrics['max_accuracies'])):
            if i < len(task_il_metrics['forgetting_per_task']):
                forgetting = task_il_metrics['forgetting_per_task'][i]
                print(f"{i+1:4d} | {task_il_metrics['max_accuracies'][i]:7.3f} | "
                      f"{task_il_metrics['final_accuracies'][i]:9.3f} | {forgetting:10.3f}")
            else:
                print(f"{i+1:4d} | {task_il_metrics['max_accuracies'][i]:7.3f} | "
                      f"{task_il_metrics['final_accuracies'][i]:9.3f} | {'N/A':>10}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Class-IL Average Forgetting: {class_il_metrics.get('average_forgetting', 0):.3f}%")
    print(f"Task-IL Average Forgetting:  {task_il_metrics.get('average_forgetting', 0):.3f}%")
    print(f"Class-IL Final Accuracy:     {class_il_metrics.get('final_average_accuracy', 0):.3f}%")
    print(f"Task-IL Final Accuracy:      {task_il_metrics.get('final_average_accuracy', 0):.3f}%")
    
    # Additional insights
    print("\n" + "=" * 80)
    print("ADDITIONAL INSIGHTS")
    print("=" * 80)
    
    if class_il_metrics and 'forgetting_per_task' in class_il_metrics:
        forgetting_values = class_il_metrics['forgetting_per_task']
        if forgetting_values:
            max_forgetting = max(forgetting_values)
            min_forgetting = min(forgetting_values)
            max_task = forgetting_values.index(max_forgetting) + 1
            min_task = forgetting_values.index(min_forgetting) + 1
            
            print(f"Class-IL - Task with highest forgetting: Task {max_task} ({max_forgetting:.3f}%)")
            print(f"Class-IL - Task with lowest forgetting:  Task {min_task} ({min_forgetting:.3f}%)")
            
            # Calculate how many tasks have significant forgetting (>1%)
            significant_forgetting = [f for f in forgetting_values if f > 1.0]
            print(f"Class-IL - Tasks with >1% forgetting: {len(significant_forgetting)}/{len(forgetting_values)}")
    
    if task_il_metrics and 'forgetting_per_task' in task_il_metrics:
        forgetting_values = task_il_metrics['forgetting_per_task']
        if forgetting_values:
            max_forgetting = max(forgetting_values)
            min_forgetting = min(forgetting_values)
            max_task = forgetting_values.index(max_forgetting) + 1
            min_task = forgetting_values.index(min_forgetting) + 1
            
            print(f"Task-IL - Task with highest forgetting:  Task {max_task} ({max_forgetting:.3f}%)")
            print(f"Task-IL - Task with lowest forgetting:   Task {min_task} ({min_forgetting:.3f}%)")
            
            # Calculate how many tasks have any forgetting
            any_forgetting = [f for f in forgetting_values if f > 0.0]
            print(f"Task-IL - Tasks with any forgetting: {len(any_forgetting)}/{len(forgetting_values)}")
    
    print("\nNote: Forgetting = Maximum accuracy achieved - Final accuracy")
    print("Lower forgetting values indicate better retention of previous knowledge.")


def main():
    # Check if command line argument is provided
    parser = argparse.ArgumentParser(description='Calculate forgetting metrics from continual learning logs')
    parser.add_argument('log_file', nargs='?', default='paste.txt', help='Path to the log file (default: paste.txt)')
    args = parser.parse_args()
    
    try:
        with open(args.log_file, 'r') as f:
            log_content = f.read()
    except FileNotFoundError:
        print(f"Error: Could not find file '{args.log_file}'")
        print("Make sure the file exists or provide the correct path.")
        return
    
    # Extract accuracies from log
    class_il_history, task_il_history = extract_accuracies_from_log(log_content)
    
    print(f"Found {len(class_il_history)} accuracy measurements in the log.")
    
    if not class_il_history:
        print("No accuracy measurements found in the log file.")
        print("Make sure the log contains lines with 'Raw accuracy values:' format.")
        return
    
    # Calculate forgetting metrics
    class_il_metrics = calculate_forgetting(class_il_history)
    task_il_metrics = calculate_forgetting(task_il_history)
    
    # Print detailed report
    print_forgetting_report(class_il_metrics, task_il_metrics)


if __name__ == "__main__":
    main()