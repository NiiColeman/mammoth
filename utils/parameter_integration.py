"""
Parameter Analysis Integration for Mammoth Framework

This module provides functions to integrate parameter efficiency analysis
directly into the Mammoth training pipeline.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from collections import defaultdict
import logging

class ParameterTracker:
    """Track parameter counts and efficiency metrics during training."""
    
    def __init__(self):
        self.param_history = []
        self.task_snapshots = {}
        self.initial_params = None
        
    def format_number(self, num: int) -> str:
        """Format large numbers with appropriate suffixes."""
        if num >= 1_000_000:
            return f"{num / 1_000_000:.2f}M"
        elif num >= 1_000:
            return f"{num / 1_000:.2f}K"
        else:
            return str(num)
    
    def categorize_parameter(self, param_name: str) -> str:
        """Categorize a parameter based on its name."""
        name_lower = param_name.lower()
        
        # Prompt-related parameters
        if any(keyword in name_lower for keyword in ['prompt', 'key']):
            if 'g_prompt' in name_lower or 'g-prompt' in name_lower:
                return 'g_prompt'
            elif 'e_prompt' in name_lower or 'e-prompt' in name_lower:
                return 'e_prompt'
            else:
                return 'prompt'
        
        # Classifier/head parameters
        elif any(keyword in name_lower for keyword in ['head', 'classifier', 'last', 'fc']):
            return 'classifier'
        
        # Backbone transformer layers
        elif any(keyword in name_lower for keyword in ['blocks', 'layers', 'attention', 'mlp', 'norm']):
            return 'backbone_transformer'
        
        # Backbone embedding layers
        elif any(keyword in name_lower for keyword in ['patch_embed', 'pos_embed', 'cls_token']):
            return 'backbone_embedding'
        
        # Other parameters
        else:
            return 'other'
    
  
    
    def analyze_model_parameters(self, model: nn.Module, task_id: Optional[int] = None, 
                                model_name: str = "", verbose: bool = True) -> Dict[str, Any]:
        """
        Comprehensive parameter analysis for continual learning models.
        
        Args:
            model: The model to analyze
            task_id: Current task ID (for logging purposes)
            model_name: Name of the model/method
            verbose: Whether to print detailed analysis
            
        Returns:
            Dictionary with parameter analysis results
        """
        
        # Basic parameter counts
        total_params = 0
        trainable_params = 0
        frozen_params = 0
        
        # Category-wise breakdown
        category_counts = defaultdict(int)
        trainable_category_counts = defaultdict(int)
        
        # Detailed parameter info
        parameter_details = []

        param_iter = None
        has_names = False

        # Attempt to get named parameters
        if hasattr(model, "named_parameters") and callable(getattr(model, "named_parameters", None)):
            try:
                param_iter = model.named_parameters()
                has_names = True
            except Exception as e:
                logging.warning(f"Error iterating named_parameters for model {type(model)}: {e}")
                param_iter = None # Fallback to parameters() if named_parameters() fails

        # If named_parameters failed or was not available, try parameters
        if param_iter is None:
            if hasattr(model, "parameters") and callable(getattr(model, "parameters", None)):
                try:
                    param_iter = [(None, p) for p in model.parameters()]
                    has_names = False
                except Exception as e:
                    logging.warning(f"Error iterating parameters for model {type(model)}: {e}")
                    param_iter = None

        if param_iter is None:
            # No parameter interface found or both failed
            logging.warning(f"Model {type(model)} has no accessible parameters or named_parameters method.")
            return {
                'model_name': model_name,
                'task_id': task_id,
                'total_params': 0,
                'trainable_params': 0,
                'frozen_params': 0,
                'category_counts': {},
                'trainable_category_counts': {},
                'efficiency_metrics': {},
                'parameter_details': [],
            }

        if not has_names:
            # Model does not have named parameters, perform simple count
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params

            logging.info(f"Model {type(model)} has no named parameters. Performing simple parameter count.")
            return {
                'model_name': model_name,
                'task_id': task_id,
                'total_params': total_params,
                'trainable_params': trainable_params,
                'frozen_params': frozen_params,
                'category_counts': {}, # Empty as no categorization is done
                'trainable_category_counts': {}, # Empty
                'efficiency_metrics': {
                    'trainable_ratio': trainable_params / total_params if total_params > 0 else 0,
                    'trainable_percentage': (trainable_params / total_params * 100) if total_params > 0 else 0,
                    'frozen_ratio': frozen_params / total_params if total_params > 0 else 0,
                    'frozen_percentage': (frozen_params / total_params * 100) if total_params > 0 else 0,
                    'parameter_overhead': None
                },
                'parameter_details': [], # Empty as no detailed analysis is done
            }

        for name, param in param_iter:
            param_count = param.numel()
            total_params += param_count

            # Track trainable vs frozen
            if param.requires_grad:
                trainable_params += param_count
            else:
                frozen_params += param_count

            # Determine the parameter name, falling back to 'unnamed' if not available
            # This handles cases where named_parameters() might yield a None name.
            param_name_for_analysis = name if has_names and name else "unnamed"

            # Categorize parameters
            category = self.categorize_parameter(param_name_for_analysis)
            category_counts[category] += param_count

            if param.requires_grad:
                trainable_category_counts[category] += param_count

            # Store detailed info for debugging
            parameter_details.append({
                'name': param_name_for_analysis,
                'shape': list(param.shape),
                'count': param_count,
                'trainable': param.requires_grad,
                'category': category
            })
        
        # Calculate efficiency metrics
        trainable_ratio = trainable_params / total_params if total_params > 0 else 0
        frozen_ratio = frozen_params / total_params if total_params > 0 else 0
        
        # Create analysis result
        analysis = {
            'model_name': model_name,
            'task_id': task_id,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': frozen_params,
            'category_counts': dict(category_counts),
            'trainable_category_counts': dict(trainable_category_counts),
            'efficiency_metrics': {
                'trainable_ratio': trainable_ratio,
                'trainable_percentage': trainable_ratio * 100,
                'frozen_ratio': frozen_ratio,
                'frozen_percentage': frozen_ratio * 100,
                'parameter_overhead': None  # Will be calculated if initial params are available
            },
            'parameter_details': parameter_details
        }
        
        # Calculate parameter overhead if we have initial params
        if self.initial_params is not None:
            initial_total = self.initial_params['total_params']
            overhead = ((total_params - initial_total) / initial_total) * 100
            analysis['efficiency_metrics']['parameter_overhead'] = overhead
        
        # Store snapshot for this task
        if task_id is not None:
            self.task_snapshots[task_id] = analysis
        
        # Print analysis if verbose
        if verbose:
            self.print_parameter_analysis(analysis)
        
        return analysis
    
    def print_parameter_analysis(self, analysis: Dict[str, Any]):
        """Print detailed parameter analysis."""
        model_name = analysis.get('model_name', 'Model')
        task_id = analysis.get('task_id')
        
        # Header
        task_info = f" - Task {task_id}" if task_id is not None else ""
        print(f"\n{'='*60}")
        print(f"📊 Parameter Analysis: {model_name}{task_info}")
        print(f"{'='*60}")
        
        # Basic counts
        print(f"\n📈 Overall Parameter Counts:")
        print(f"  Total Parameters:          {self.format_number(analysis['total_params'])}")
        print(f"  Trainable Parameters:      {self.format_number(analysis['trainable_params'])}")
        print(f"  Frozen Parameters:         {self.format_number(analysis['frozen_params'])}")
        
        # Efficiency metrics
        metrics = analysis['efficiency_metrics']
        print(f"\n⚡ Efficiency Metrics:")
        print(f"  Trainable Ratio:           {metrics['trainable_percentage']:.2f}%")
        print(f"  Frozen Ratio:              {metrics['frozen_percentage']:.2f}%")
        
        if metrics['parameter_overhead'] is not None:
            print(f"  Parameter Overhead:        {metrics['parameter_overhead']:.2f}%")
        
        # Category breakdown
        print(f"\n🔍 Parameter Breakdown by Category:")
        category_counts = analysis['category_counts']
        trainable_counts = analysis['trainable_category_counts']
        
        for category, total_count in category_counts.items():
            trainable_count = trainable_counts.get(category, 0)
            frozen_count = total_count - trainable_count
            
            print(f"  {category:<20}: {self.format_number(total_count):<10} "
                  f"(Trainable: {self.format_number(trainable_count)}, "
                  f"Frozen: {self.format_number(frozen_count)})")
    
    def track_training_progress(self, model: nn.Module, epoch: int, 
                               track_frequency: int = 10) -> Optional[Dict[str, int]]:
        """
        Track parameter counts during training.
        
        Args:
            model: The model being trained
            epoch: Current epoch number
            track_frequency: How often to track (every N epochs)
            
        Returns:
            Parameter counts if tracked, None otherwise
        """
        if epoch % track_frequency == 0:
            trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_count = sum(p.numel() for p in model.parameters())
            
            param_snapshot = {
                'epoch': epoch,
                'total_params': total_count,
                'trainable_params': trainable_count
            }
            
            self.param_history.append(param_snapshot)
            
            logging.info(f"Epoch {epoch}: {self.format_number(trainable_count)} trainable params")
            
            return param_snapshot
        
        return None
    
    def compare_tasks(self, task1_id: int, task2_id: int) -> Dict[str, Any]:
        """Compare parameter counts between two tasks."""
        if task1_id not in self.task_snapshots or task2_id not in self.task_snapshots:
            raise ValueError(f"Task snapshots not found for tasks {task1_id} and/or {task2_id}")
        
        task1 = self.task_snapshots[task1_id]
        task2 = self.task_snapshots[task2_id]
        
        comparison = {
            'task1_id': task1_id,
            'task2_id': task2_id,
            'total_params_change': task2['total_params'] - task1['total_params'],
            'trainable_params_change': task2['trainable_params'] - task1['trainable_params'],
            'category_changes': {}
        }
        
        # Calculate category-wise changes
        for category in set(list(task1['category_counts'].keys()) + list(task2['category_counts'].keys())):
            task1_count = task1['category_counts'].get(category, 0)
            task2_count = task2['category_counts'].get(category, 0)
            comparison['category_changes'][category] = task2_count - task1_count
        
        return comparison
    
    def print_task_comparison(self, comparison: Dict[str, Any]):
        """Print comparison between tasks."""
        print(f"\n{'='*60}")
        print(f"📊 Task Comparison: Task {comparison['task1_id']} → Task {comparison['task2_id']}")
        print(f"{'='*60}")
        
        print(f"\n📈 Parameter Changes:")
        print(f"  Total Parameters:          {comparison['total_params_change']:+,}")
        print(f"  Trainable Parameters:      {comparison['trainable_params_change']:+,}")
        
        if comparison['category_changes']:
            print(f"\n🔍 Changes by Category:")
            for category, change in comparison['category_changes'].items():
                if change != 0:
                    print(f"  {category:<20}: {change:+,}")


def integrate_parameter_tracking():
    """
    Integration instructions and code snippets for the Mammoth framework.
    This function returns the modified training functions.
    """
    
    def modified_train_function():
        """
        Modified version of the train function from utils/training.py
        with integrated parameter tracking.
        """
        return '''
# Add this import at the top of utils/training.py
from parameter_integration import ParameterTracker

def train(model: ContinualModel, dataset: ContinualDataset,
          args: Optional[Namespace] = None) -> None:
    """
    The training process with integrated parameter tracking.
    """
    # ... existing imports and setup code ...
    
    # Initialize parameter tracker
    param_tracker = ParameterTracker()
    
    # ... existing code until model setup ...
    
    model.net.to(model.device)
    torch.cuda.empty_cache()
    
    # ADDITION: Analyze initial model parameters
    logging.info("🔍 Analyzing initial model parameters...")
    initial_analysis = param_tracker.analyze_model_parameters(
        model.net, 
        task_id=None, 
        model_name=f"{model.NAME}",
        verbose=True
    )
    param_tracker.initial_params = initial_analysis
    
    with track_system_stats(logger) as system_tracker:
        results, results_mask_classes = [], []
        
        # ... existing setup code ...
        
        for cur_task in range(start_task, end_task):
            model.net.train()
            train_loader, _ = dataset.get_data_loaders()
            
            # ... existing pre-task setup ...
            
            # ADDITION: Analyze parameters before task
            logging.info(f"📊 Analyzing parameters before Task {cur_task + 1}...")
            pre_task_analysis = param_tracker.analyze_model_parameters(
                model.net, 
                task_id=cur_task, 
                model_name=f"{model.NAME} (Pre-Task {cur_task + 1})",
                verbose=False  # Set to True for detailed output
            )
            
            model.meta_begin_task(dataset)
            
            # ADDITION: Analyze parameters after begin_task
            post_begin_analysis = param_tracker.analyze_model_parameters(
                model.net, 
                task_id=cur_task, 
                model_name=f"{model.NAME} (Post-Begin Task {cur_task + 1})",
                verbose=True
            )
            
            # ADDITION: Compare if there were changes during begin_task
            if (post_begin_analysis['total_params'] != pre_task_analysis['total_params']):
                logging.info(f"⚠️  Parameters changed during begin_task for Task {cur_task + 1}")
                param_change = post_begin_analysis['total_params'] - pre_task_analysis['total_params']
                logging.info(f"   Change: {param_change:+,} parameters")
            
            if not args.inference_only and args.n_epochs > 0:
                # ... existing training setup ...
                
                while True:
                    model.meta_begin_epoch(epoch, dataset)
                    
                    train_pbar.set_description(f"Task {cur_task + 1} - Epoch {epoch + 1}")
                    
                    # ADDITION: Track parameters during training
                    param_tracker.track_training_progress(model.net, epoch, track_frequency=5)
                    
                    train_single_epoch(model, train_loader, args, pbar=train_pbar, epoch=epoch,
                                       system_tracker=system_tracker, scheduler=scheduler)
                    
                    model.meta_end_epoch(epoch, dataset)
                    
                    epoch += 1
                    # ... existing epoch termination conditions ...
                
                train_pbar.close()
            
            model.meta_end_task(dataset)
            
            # ADDITION: Analyze parameters after task completion
            final_task_analysis = param_tracker.analyze_model_parameters(
                model.net, 
                task_id=cur_task, 
                model_name=f"{model.NAME} (Task {cur_task + 1} Complete)",
                verbose=True
            )
            
            # ADDITION: Log parameter efficiency to wandb/logger if available
            if not args.disable_log:
                log_parameter_efficiency(logger, final_task_analysis, args)
            
            # ... existing evaluation and logging code ...
        
        # ADDITION: Final parameter analysis summary
        logging.info("\\n" + "="*80)
        logging.info("📊 FINAL PARAMETER EFFICIENCY SUMMARY")
        logging.info("="*80)
        
        if len(param_tracker.task_snapshots) > 1:
            # Compare first and last tasks
            first_task = min(param_tracker.task_snapshots.keys())
            last_task = max(param_tracker.task_snapshots.keys())
            comparison = param_tracker.compare_tasks(first_task, last_task)
            param_tracker.print_task_comparison(comparison)
        
        # Print parameter history
        if param_tracker.param_history:
            logging.info("\\n📈 Parameter Count History During Training:")
            for snapshot in param_tracker.param_history[-5:]:  # Last 5 snapshots
                logging.info(f"  Epoch {snapshot['epoch']}: {param_tracker.format_number(snapshot['trainable_params'])} trainable")
        
        # ... existing cleanup code ...

def log_parameter_efficiency(logger, analysis: Dict[str, Any], args):
    """Log parameter efficiency metrics to wandb or other loggers."""
    task_id = analysis.get('task_id', 0)
    
    log_data = {
        f'params/total_task_{task_id}': analysis['total_params'],
        f'params/trainable_task_{task_id}': analysis['trainable_params'],
        f'params/frozen_task_{task_id}': analysis['frozen_params'],
        f'params/trainable_ratio_task_{task_id}': analysis['efficiency_metrics']['trainable_ratio'],
    }
    
    # Add category-specific metrics
    for category, count in analysis['trainable_category_counts'].items():
        log_data[f'params/{category}_task_{task_id}'] = count
    
    # Add parameter overhead if available
    if analysis['efficiency_metrics']['parameter_overhead'] is not None:
        log_data[f'params/overhead_task_{task_id}'] = analysis['efficiency_metrics']['parameter_overhead']
    
    # Log to wandb if available
    try:
        import wandb
        if not args.nowand:
            wandb.log(log_data)
    except ImportError:
        pass
    
    # Log to your custom logger
    if hasattr(logger, 'log'):
        logger.log(log_data)
    
    # Also log some key metrics to console
    logging.info(f"📊 Task {task_id} Parameter Efficiency:")
    logging.info(f"   Trainable: {analysis['trainable_params']:,} ({analysis['efficiency_metrics']['trainable_percentage']:.2f}%)")
    if analysis['efficiency_metrics']['parameter_overhead'] is not None:
        logging.info(f"   Overhead: {analysis['efficiency_metrics']['parameter_overhead']:.2f}%")
'''
    
    def modified_continual_model():
        """
        Modified ContinualModel class with parameter tracking capabilities.
        """
        return '''
# Add to models/utils/continual_model.py

class ContinualModel(nn.Module):
    """
    Continual model with integrated parameter tracking.
    """
    
    def __init__(self, backbone, loss, args, transform, dataset=None):
        # ... existing initialization code ...
        
        # Initialize parameter tracker
        self.param_tracker = ParameterTracker()
        
    def observe(self, inputs, labels, not_aug_inputs, epoch=0):
        """
        The observe method with optional parameter tracking.
        """
        # ... existing observe code ...
        
        # Optional: Track parameters every N epochs during training
        if hasattr(self, 'param_tracker') and epoch % 20 == 0:
            self.param_tracker.track_training_progress(self.net, epoch, track_frequency=1)
        
        return loss.item()
    
    def meta_begin_task(self, dataset):
        """
        Begin task with parameter analysis.
        """
        # Track parameters before task-specific initialization
        if hasattr(self, 'param_tracker'):
            self.param_tracker.analyze_model_parameters(
                self.net, 
                task_id=self.current_task,
                model_name=f"{self.NAME} (Begin Task {self.current_task})",
                verbose=False
            )
        
        # ... existing begin_task code ...
    
    def meta_end_task(self, dataset):
        """
        End task with parameter analysis.
        """
        # ... existing end_task code ...
        
        # Track parameters after task completion
        if hasattr(self, 'param_tracker'):
            analysis = self.param_tracker.analyze_model_parameters(
                self.net, 
                task_id=self.current_task,
                model_name=f"{self.NAME} (End Task {self.current_task})",
                verbose=True
            )
            
            # Log key metrics
            logging.info(f"🎯 Task {self.current_task} completed with {analysis['trainable_params']:,} trainable parameters")
'''
    
    return {
        'train_function': modified_train_function(),
        'continual_model': modified_continual_model()
    }


if __name__ == "__main__":
    # Example usage and testing
    print("Parameter Tracking Integration for Mammoth Framework")
    print("="*60)
    print("\nTo integrate this into your Mammoth framework:")
    print("1. Copy this file to your Mammoth project directory")
    print("2. Modify utils/training.py with the provided train function")
    print("3. Modify models/utils/continual_model.py with the enhanced ContinualModel")
    print("4. Run your training as usual - parameter analysis will be automatic")
    
    integration_code = integrate_parameter_tracking()
    print(f"\n✅ Integration code generated successfully!")
    print(f"📝 Check the returned functions for the exact modifications needed.")