"""
Core ML Pipeline implementation for MDT Dashboard.
Provides a robust, modular pipeline framework for ML operations.
"""

import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineMetrics:
    """Pipeline execution metrics."""
    execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    rows_processed: int = 0
    features_created: int = 0
    model_accuracy: Optional[float] = None
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Pipeline execution result."""
    pipeline_id: str
    status: PipelineStatus
    result_data: Any = None
    error_message: Optional[str] = None
    metrics: PipelineMetrics = field(default_factory=PipelineMetrics)
    execution_log: List[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


class PipelineStep(ABC):
    """Abstract base class for pipeline steps."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.step_id = str(uuid.uuid4())
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def execute(self, input_data: Any, context: Dict[str, Any]) -> Any:
        """Execute the pipeline step."""
        pass
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data for the step."""
        return True
    
    def pre_execute(self, input_data: Any, context: Dict[str, Any]) -> None:
        """Pre-execution hook."""
        self.logger.info(f"Starting step: {self.name}")
    
    def post_execute(self, result: Any, context: Dict[str, Any]) -> None:
        """Post-execution hook."""
        self.logger.info(f"Completed step: {self.name}")


class MLPipeline:
    """
    Flexible ML Pipeline for building and executing ML workflows.
    Supports step composition, error handling, and comprehensive monitoring.
    """
    
    def __init__(
        self,
        name: str,
        steps: List[PipelineStep],
        config: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.steps = steps
        self.config = config or {}
        self.pipeline_id = str(uuid.uuid4())
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Pipeline state
        self.status = PipelineStatus.PENDING
        self.current_step_index = 0
        self.context = {}
        self.results = []
        self.execution_log = []
        
    def add_step(self, step: PipelineStep, position: Optional[int] = None) -> None:
        """Add a step to the pipeline."""
        if position is None:
            self.steps.append(step)
        else:
            self.steps.insert(position, step)
        
        self.logger.info(f"Added step '{step.name}' to pipeline")
    
    def remove_step(self, step_name: str) -> bool:
        """Remove a step from the pipeline."""
        for i, step in enumerate(self.steps):
            if step.name == step_name:
                removed_step = self.steps.pop(i)
                self.logger.info(f"Removed step '{removed_step.name}' from pipeline")
                return True
        return False
    
    def execute(
        self,
        input_data: Any,
        resume_from_step: Optional[int] = None,
        stop_on_error: bool = True
    ) -> PipelineResult:
        """
        Execute the pipeline with comprehensive error handling and monitoring.
        
        Args:
            input_data: Initial input data for the pipeline
            resume_from_step: Step index to resume from (for recovery)
            stop_on_error: Whether to stop execution on first error
            
        Returns:
            PipelineResult: Complete execution result with metrics
        """
        result = PipelineResult(
            pipeline_id=self.pipeline_id,
            status=PipelineStatus.RUNNING
        )
        
        start_time = time.time()
        current_data = input_data
        
        try:
            self.status = PipelineStatus.RUNNING
            self.context = {
                "pipeline_id": self.pipeline_id,
                "pipeline_name": self.name,
                "execution_start": result.started_at,
                "config": self.config
            }
            
            # Determine starting step
            start_step = resume_from_step if resume_from_step is not None else 0
            self.current_step_index = start_step
            
            self.logger.info(f"Starting pipeline '{self.name}' with {len(self.steps)} steps")
            result.execution_log.append(f"Pipeline started at {result.started_at}")
            
            # Execute steps
            for i in range(start_step, len(self.steps)):
                step = self.steps[i]
                self.current_step_index = i
                
                try:
                    # Pre-execution validation and hooks
                    if not step.validate_input(current_data):
                        error_msg = f"Input validation failed for step '{step.name}'"
                        self.logger.error(error_msg)
                        if stop_on_error:
                            raise ValueError(error_msg)
                        continue
                    
                    step.pre_execute(current_data, self.context)
                    
                    # Execute step
                    step_start = time.time()
                    step_result = step.execute(current_data, self.context)
                    step_duration = time.time() - step_start
                    
                    # Post-execution
                    step.post_execute(step_result, self.context)
                    
                    # Update pipeline state
                    current_data = step_result
                    self.results.append({
                        "step_name": step.name,
                        "step_id": step.step_id,
                        "duration": step_duration,
                        "status": "success"
                    })
                    
                    log_msg = f"Step '{step.name}' completed in {step_duration:.2f}s"
                    self.logger.info(log_msg)
                    result.execution_log.append(log_msg)
                    
                except Exception as e:
                    error_msg = f"Step '{step.name}' failed: {str(e)}"
                    self.logger.error(error_msg, exc_info=True)
                    
                    self.results.append({
                        "step_name": step.name,
                        "step_id": step.step_id,
                        "status": "failed",
                        "error": str(e)
                    })
                    
                    result.execution_log.append(error_msg)
                    
                    if stop_on_error:
                        raise
            
            # Pipeline completed successfully
            result.status = PipelineStatus.SUCCESS
            result.result_data = current_data
            self.status = PipelineStatus.SUCCESS
            
            success_msg = f"Pipeline '{self.name}' completed successfully"
            self.logger.info(success_msg)
            result.execution_log.append(success_msg)
            
        except Exception as e:
            # Pipeline failed
            result.status = PipelineStatus.FAILED
            result.error_message = str(e)
            self.status = PipelineStatus.FAILED
            
            error_msg = f"Pipeline '{self.name}' failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            result.execution_log.append(error_msg)
        
        finally:
            # Calculate metrics
            total_time = time.time() - start_time
            result.completed_at = datetime.utcnow()
            result.metrics.execution_time = total_time
            
            # Memory and performance metrics (simplified)
            try:
                import psutil
                process = psutil.Process()
                result.metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
                result.metrics.cpu_usage_percent = process.cpu_percent()
            except ImportError:
                self.logger.warning("psutil not available, skipping memory metrics")
            
            self.logger.info(f"Pipeline execution completed in {total_time:.2f}s")
        
        return result
    
    def get_step_by_name(self, step_name: str) -> Optional[PipelineStep]:
        """Get a step by name."""
        for step in self.steps:
            if step.name == step_name:
                return step
        return None
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get pipeline information."""
        return {
            "pipeline_id": self.pipeline_id,
            "name": self.name,
            "status": self.status.value,
            "steps": [
                {
                    "name": step.name,
                    "step_id": step.step_id,
                    "type": step.__class__.__name__
                }
                for step in self.steps
            ],
            "current_step": self.current_step_index,
            "total_steps": len(self.steps),
            "config": self.config
        }
    
    def save_checkpoint(self, checkpoint_path: Path) -> None:
        """Save pipeline checkpoint for recovery."""
        checkpoint_data = {
            "pipeline_id": self.pipeline_id,
            "name": self.name,
            "current_step_index": self.current_step_index,
            "status": self.status.value,
            "context": self.context,
            "results": self.results,
            "execution_log": self.execution_log,
            "config": self.config
        }
        
        checkpoint_path.write_text(json.dumps(checkpoint_data, indent=2, default=str))
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load pipeline checkpoint for recovery."""
        checkpoint_data = json.loads(checkpoint_path.read_text())
        
        self.pipeline_id = checkpoint_data["pipeline_id"]
        self.name = checkpoint_data["name"]
        self.current_step_index = checkpoint_data["current_step_index"]
        self.status = PipelineStatus(checkpoint_data["status"])
        self.context = checkpoint_data["context"]
        self.results = checkpoint_data["results"]
        self.execution_log = checkpoint_data["execution_log"]
        self.config = checkpoint_data["config"]
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")


class PipelineBuilder:
    """Builder pattern for creating ML pipelines."""
    
    def __init__(self, name: str):
        self.name = name
        self.steps = []
        self.config = {}
    
    def add_step(self, step: PipelineStep) -> 'PipelineBuilder':
        """Add a step to the pipeline."""
        self.steps.append(step)
        return self
    
    def with_config(self, config: Dict[str, Any]) -> 'PipelineBuilder':
        """Set pipeline configuration."""
        self.config.update(config)
        return self
    
    def build(self) -> MLPipeline:
        """Build the pipeline."""
        return MLPipeline(self.name, self.steps, self.config)


# Convenience function for creating pipelines
def create_pipeline(name: str, steps: List[PipelineStep], config: Optional[Dict[str, Any]] = None) -> MLPipeline:
    """Create a new ML pipeline."""
    return MLPipeline(name, steps, config)
