"""
Distributed Training Infrastructure for Large Scale ML Models
Supports distributed training, model parallelism, and advanced optimization techniques
"""

import logging
import os
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import multiprocessing as mp
import threading
import time

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    
    # Distributed settings
    backend: str = "nccl"  # nccl, gloo, mpi
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    
    # Model parallelism
    model_parallel_size: int = 1
    data_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    
    # Memory optimization
    gradient_checkpointing: bool = True
    mixed_precision: bool = True
    gradient_clipping: float = 1.0
    
    # Communication
    find_unused_parameters: bool = True
    bucket_cap_mb: int = 25
    
    # Training parameters
    accumulate_grad_batches: int = 1
    max_epochs: int = 100
    val_check_interval: float = 1.0


class DistributedTrainingManager:
    """Manager for distributed training operations."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.is_distributed = config.world_size > 1
        self.device = self._setup_device()
        
    def _setup_device(self):
        """Setup device for training."""
        if self._torch_available():
            import torch
            if torch.cuda.is_available():
                torch.cuda.set_device(self.config.local_rank)
                return torch.device(f'cuda:{self.config.local_rank}')
            else:
                return torch.device('cpu')
        return 'cpu'
    
    def _torch_available(self):
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            return False
    
    def init_distributed(self):
        """Initialize distributed training."""
        if not self._torch_available():
            logger.warning("PyTorch not available, skipping distributed initialization")
            return
        
        try:
            import torch.distributed as dist
            
            if not dist.is_available():
                logger.warning("Distributed training not available")
                return
            
            if not dist.is_initialized():
                # Initialize process group
                dist.init_process_group(
                    backend=self.config.backend,
                    world_size=self.config.world_size,
                    rank=self.config.rank
                )
                
                logger.info(f"Initialized distributed training: "
                           f"rank={self.config.rank}, world_size={self.config.world_size}")
                
        except Exception as e:
            logger.error(f"Failed to initialize distributed training: {e}")
    
    def wrap_model(self, model):
        """Wrap model for distributed training."""
        if not self._torch_available():
            return model
        
        try:
            import torch
            import torch.nn as nn
            from torch.nn.parallel import DistributedDataParallel as DDP
            
            if isinstance(model, nn.Module):
                model = model.to(self.device)
                
                if self.is_distributed:
                    model = DDP(
                        model,
                        device_ids=[self.config.local_rank],
                        output_device=self.config.local_rank,
                        find_unused_parameters=self.config.find_unused_parameters,
                        bucket_cap_mb=self.config.bucket_cap_mb
                    )
                
                return model
            
        except Exception as e:
            logger.error(f"Failed to wrap model for distributed training: {e}")
        
        return model
    
    def setup_data_loader(self, dataset, batch_size, shuffle=True):
        """Setup distributed data loader."""
        if not self._torch_available():
            logger.warning("PyTorch not available, returning None")
            return None
        
        try:
            import torch
            from torch.utils.data import DataLoader
            from torch.utils.data.distributed import DistributedSampler
            
            sampler = None
            if self.is_distributed:
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=self.config.world_size,
                    rank=self.config.rank,
                    shuffle=shuffle
                )
                shuffle = False  # DistributedSampler handles shuffling
            
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                sampler=sampler,
                num_workers=min(4, mp.cpu_count()),
                pin_memory=True if torch.cuda.is_available() else False
            )
            
        except Exception as e:
            logger.error(f"Failed to setup distributed data loader: {e}")
            return None


class ModelParallelism:
    """Implementation of model parallelism strategies."""
    
    def __init__(self):
        self.device_map = {}
        self.layer_devices = {}
    
    def create_device_map(self, model, num_gpus: int):
        """Create device mapping for model layers."""
        if not self._torch_available():
            return {}
        
        try:
            import torch
            
            if not torch.cuda.is_available() or num_gpus <= 1:
                return {}
            
            # Simple layer distribution
            layers = list(model.named_modules())
            layers_per_gpu = len(layers) // num_gpus
            
            device_map = {}
            for i, (name, _) in enumerate(layers):
                gpu_id = min(i // layers_per_gpu, num_gpus - 1)
                device_map[name] = f'cuda:{gpu_id}'
            
            return device_map
            
        except Exception as e:
            logger.error(f"Failed to create device map: {e}")
            return {}
    
    def apply_model_parallelism(self, model, device_map):
        """Apply model parallelism based on device map."""
        if not device_map:
            return model
        
        try:
            for name, module in model.named_modules():
                if name in device_map:
                    device = device_map[name]
                    module.to(device)
                    logger.debug(f"Moved {name} to {device}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to apply model parallelism: {e}")
            return model
    
    def _torch_available(self):
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            return False


class GradientAccumulation:
    """Gradient accumulation for large batch training."""
    
    def __init__(self, accumulation_steps: int = 4):
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        self.accumulated_gradients = {}
    
    def should_update(self):
        """Check if gradients should be updated."""
        return (self.current_step + 1) % self.accumulation_steps == 0
    
    def step(self):
        """Increment accumulation step."""
        self.current_step += 1
    
    def reset(self):
        """Reset accumulation counter."""
        self.current_step = 0


class MemoryOptimizer:
    """Memory optimization utilities."""
    
    @staticmethod
    def enable_gradient_checkpointing(model):
        """Enable gradient checkpointing to save memory."""
        try:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                logger.info("Enabled gradient checkpointing")
            elif hasattr(model, 'enable_gradient_checkpointing'):
                model.enable_gradient_checkpointing()
                logger.info("Enabled gradient checkpointing")
        except Exception as e:
            logger.warning(f"Could not enable gradient checkpointing: {e}")
    
    @staticmethod
    def optimize_memory_usage():
        """Apply general memory optimizations."""
        if MemoryOptimizer._torch_available():
            try:
                import torch
                
                # Set memory fraction if CUDA is available
                if torch.cuda.is_available():
                    # Don't use all GPU memory at once
                    torch.cuda.empty_cache()
                    
                    # Enable memory profiling
                    if hasattr(torch.cuda, 'memory_stats'):
                        stats = torch.cuda.memory_stats()
                        logger.info(f"CUDA memory stats: {stats}")
                
            except Exception as e:
                logger.warning(f"Could not optimize memory usage: {e}")
    
    @staticmethod
    def _torch_available():
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            return False


class PipelineParallelism:
    """Pipeline parallelism implementation."""
    
    def __init__(self, num_stages: int = 2):
        self.num_stages = num_stages
        self.pipeline_stages = []
        self.micro_batch_size = 1
    
    def create_pipeline(self, model, stage_configs: List[Dict]):
        """Create pipeline stages."""
        try:
            for i, config in enumerate(stage_configs):
                stage = PipelineStage(
                    stage_id=i,
                    layers=config.get('layers', []),
                    device=config.get('device', 'cpu')
                )
                self.pipeline_stages.append(stage)
            
            logger.info(f"Created pipeline with {len(self.pipeline_stages)} stages")
            
        except Exception as e:
            logger.error(f"Failed to create pipeline: {e}")
    
    def forward_pass(self, inputs):
        """Execute forward pass through pipeline."""
        current_input = inputs
        
        for stage in self.pipeline_stages:
            current_input = stage.forward(current_input)
        
        return current_input


class PipelineStage:
    """Individual stage in pipeline parallelism."""
    
    def __init__(self, stage_id: int, layers: List, device: str):
        self.stage_id = stage_id
        self.layers = layers
        self.device = device
        self.input_queue = []
        self.output_queue = []
    
    def forward(self, inputs):
        """Forward pass through this stage."""
        # Simplified forward pass
        outputs = inputs
        
        for layer in self.layers:
            if hasattr(layer, '__call__'):
                outputs = layer(outputs)
        
        return outputs


class AutoScaler:
    """Automatic scaling based on resource utilization."""
    
    def __init__(self, min_replicas: int = 1, max_replicas: int = 10):
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.current_replicas = min_replicas
        self.metrics_history = []
        
    def collect_metrics(self):
        """Collect system metrics for scaling decisions."""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            # GPU metrics if available
            gpu_util = 0
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_util = sum(gpu.load for gpu in gpus) / len(gpus) * 100
            except ImportError:
                pass
            
            metrics = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'gpu_util': gpu_util,
                'timestamp': time.time()
            }
            
            self.metrics_history.append(metrics)
            
            # Keep only recent metrics
            if len(self.metrics_history) > 100:
                self.metrics_history.pop(0)
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Could not collect metrics: {e}")
            return {}
    
    def should_scale_up(self):
        """Determine if should scale up."""
        if len(self.metrics_history) < 5:
            return False
        
        recent_metrics = self.metrics_history[-5:]
        avg_cpu = sum(m['cpu_percent'] for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m['memory_percent'] for m in recent_metrics) / len(recent_metrics)
        
        return (avg_cpu > 80 or avg_memory > 80) and self.current_replicas < self.max_replicas
    
    def should_scale_down(self):
        """Determine if should scale down."""
        if len(self.metrics_history) < 10:
            return False
        
        recent_metrics = self.metrics_history[-10:]
        avg_cpu = sum(m['cpu_percent'] for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m['memory_percent'] for m in recent_metrics) / len(recent_metrics)
        
        return avg_cpu < 30 and avg_memory < 50 and self.current_replicas > self.min_replicas
    
    def scale_up(self):
        """Scale up the number of replicas."""
        if self.current_replicas < self.max_replicas:
            self.current_replicas += 1
            logger.info(f"Scaled up to {self.current_replicas} replicas")
            return True
        return False
    
    def scale_down(self):
        """Scale down the number of replicas."""
        if self.current_replicas > self.min_replicas:
            self.current_replicas -= 1
            logger.info(f"Scaled down to {self.current_replicas} replicas")
            return True
        return False


class DistributedOptimizer:
    """Distributed optimization strategies."""
    
    def __init__(self, base_optimizer, model):
        self.base_optimizer = base_optimizer
        self.model = model
        self.gradient_compression = False
        self.allreduce_post_accumulation = True
    
    def enable_gradient_compression(self, compression_ratio: float = 0.01):
        """Enable gradient compression to reduce communication."""
        self.gradient_compression = True
        self.compression_ratio = compression_ratio
        logger.info(f"Enabled gradient compression with ratio {compression_ratio}")
    
    def step(self):
        """Optimized step with gradient synchronization."""
        if self.gradient_compression:
            self._compress_gradients()
        
        if hasattr(self.base_optimizer, 'step'):
            self.base_optimizer.step()
    
    def _compress_gradients(self):
        """Apply gradient compression."""
        try:
            for param in self.model.parameters():
                if param.grad is not None and self._torch_available():
                    import torch
                    
                    # Simple top-k compression
                    grad_flat = param.grad.view(-1)
                    k = int(grad_flat.numel() * self.compression_ratio)
                    
                    if k > 0:
                        _, indices = torch.topk(grad_flat.abs(), k)
                        compressed_grad = torch.zeros_like(grad_flat)
                        compressed_grad[indices] = grad_flat[indices]
                        param.grad = compressed_grad.view_as(param.grad)
                        
        except Exception as e:
            logger.warning(f"Failed to compress gradients: {e}")
    
    def _torch_available(self):
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            return False


# Export classes
__all__ = [
    "DistributedConfig",
    "DistributedTrainingManager", 
    "ModelParallelism",
    "GradientAccumulation",
    "MemoryOptimizer",
    "PipelineParallelism",
    "AutoScaler",
    "DistributedOptimizer"
]
