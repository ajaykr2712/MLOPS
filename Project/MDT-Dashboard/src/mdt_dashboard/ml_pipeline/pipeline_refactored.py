"""
Core ML Pipeline implementation for MDT Dashboard.
Provides a robust, modular pipeline framework for ML operations.

Refactored for improved code quality, type safety, and maintainability.
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable

try:
    from pydantic import BaseModel, ConfigDict, Field, field_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object
    
    def ConfigDict(**kwargs):
        return None
    
    def Field(default=None, **kwargs):
        return default
    
    def field_validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class PipelineStatus(str, Enum):
    """Pipeline execution status."""
    
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class StepStatus(str, Enum):
    """Pipeline step status."""
    
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class PipelineMetrics(BaseModel):
    """Pipeline execution metrics."""
    
    model_config = ConfigDict(extra="forbid") if PYDANTIC_AVAILABLE else None
    
    execution_time_seconds: float = Field(default=0.0, ge=0.0)
    memory_usage_mb: float = Field(default=0.0, ge=0.0)
    cpu_usage_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    rows_processed: int = Field(default=0, ge=0)
    features_created: int = Field(default=0, ge=0)
    model_accuracy: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    custom_metrics: Dict[str, Any] = Field(default_factory=dict)
    
    def add_custom_metric(self, key: str, value: Any) -> None:
        """Add a custom metric."""
        self.custom_metrics[key] = value
    
    def get_custom_metric(self, key: str, default: Any = None) -> Any:
        """Get a custom metric."""
        return self.custom_metrics.get(key, default)


class StepResult(BaseModel):
    """Result of a pipeline step execution."""
    
    model_config = ConfigDict(extra="forbid") if PYDANTIC_AVAILABLE else None
    
    step_name: str
    step_id: str
    status: StepStatus
    execution_time_seconds: float = Field(ge=0.0)
    error_message: Optional[str] = None
    output_size: Optional[int] = None
    memory_delta_mb: Optional[float] = None
    custom_data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class PipelineResult(BaseModel):
    """Complete pipeline execution result."""
    
    model_config = ConfigDict(extra="forbid") if PYDANTIC_AVAILABLE else None
    
    pipeline_id: str
    pipeline_name: str
    status: PipelineStatus
    result_data: Optional[Any] = None
    error_message: Optional[str] = None
    metrics: PipelineMetrics = Field(default_factory=PipelineMetrics)
    step_results: List[StepResult] = Field(default_factory=list)
    execution_log: List[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    @property
    def duration_seconds(self) -> float:
        """Get total execution duration."""
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return (datetime.now() - self.started_at).total_seconds()
    
    @property
    def success_rate(self) -> float:
        """Get success rate of steps."""
        if not self.step_results:
            return 0.0
        successful_steps = sum(1 for r in self.step_results if r.status == StepStatus.SUCCESS)
        return successful_steps / len(self.step_results)


class PipelineContext(BaseModel):
    """Context shared across pipeline steps."""
    
    model_config = ConfigDict(extra="allow") if PYDANTIC_AVAILABLE else None
    
    pipeline_id: str
    pipeline_name: str
    execution_start: datetime
    current_step_index: int = 0
    config: Dict[str, Any] = Field(default_factory=dict)
    shared_data: Dict[str, Any] = Field(default_factory=dict)
    
    def set_shared_data(self, key: str, value: Any) -> None:
        """Set shared data for use by other steps."""
        self.shared_data[key] = value
    
    def get_shared_data(self, key: str, default: Any = None) -> Any:
        """Get shared data from other steps."""
        return self.shared_data.get(key, default)


class PipelineStep(ABC):
    """Abstract base class for pipeline steps with enhanced functionality."""
    
    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        required_inputs: Optional[List[str]] = None,
        produces_outputs: Optional[List[str]] = None,
        timeout_seconds: Optional[int] = None
    ):
        self.name = name
        self.config = config or {}
        self.step_id = str(uuid.uuid4())
        self.required_inputs = required_inputs or []
        self.produces_outputs = produces_outputs or []
        self.timeout_seconds = timeout_seconds
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # State tracking
        self.status = StepStatus.PENDING
        self.error_message: Optional[str] = None
        self.execution_start: Optional[datetime] = None
        self.execution_end: Optional[datetime] = None
    
    @abstractmethod
    def execute(self, input_data: Any, context: PipelineContext) -> Any:
        """Execute the pipeline step."""
        pass
    
    def validate_input(self, input_data: Any, context: PipelineContext) -> bool:
        """Validate input data for the step."""
        return True
    
    def validate_config(self) -> bool:
        """Validate step configuration."""
        return True
    
    def pre_execute(self, input_data: Any, context: PipelineContext) -> None:
        """Pre-execution hook."""
        self.status = StepStatus.RUNNING
        self.execution_start = datetime.now()
        self.logger.info(f"Starting step: {self.name}")
    
    def post_execute(self, result: Any, context: PipelineContext) -> None:
        """Post-execution hook."""
        self.execution_end = datetime.now()
        self.status = StepStatus.SUCCESS
        self.logger.info(f"Completed step: {self.name}")
    
    def on_error(self, error: Exception, context: PipelineContext) -> None:
        """Error handling hook."""
        self.execution_end = datetime.now()
        self.status = StepStatus.FAILED
        self.error_message = str(error)
        self.logger.error(f"Step {self.name} failed: {error}")
    
    def can_skip(self, input_data: Any, context: PipelineContext) -> bool:
        """Determine if step can be skipped."""
        return False
    
    def get_execution_time(self) -> float:
        """Get step execution time in seconds."""
        if self.execution_start and self.execution_end:
            return (self.execution_end - self.execution_start).total_seconds()
        elif self.execution_start:
            return (datetime.now() - self.execution_start).total_seconds()
        return 0.0
    
    def get_step_info(self) -> Dict[str, Any]:
        """Get step information."""
        return {
            "name": self.name,
            "step_id": self.step_id,
            "type": self.__class__.__name__,
            "status": self.status.value,
            "config": self.config,
            "required_inputs": self.required_inputs,
            "produces_outputs": self.produces_outputs,
            "timeout_seconds": self.timeout_seconds,
            "execution_time": self.get_execution_time(),
            "error_message": self.error_message
        }


class ConditionalStep(PipelineStep):
    """Step that executes based on a condition."""
    
    def __init__(
        self,
        name: str,
        condition_func: Callable[[Any, PipelineContext], bool],
        true_step: PipelineStep,
        false_step: Optional[PipelineStep] = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.condition_func = condition_func
        self.true_step = true_step
        self.false_step = false_step
    
    def execute(self, input_data: Any, context: PipelineContext) -> Any:
        """Execute conditional logic."""
        try:
            condition_result = self.condition_func(input_data, context)
            
            if condition_result and self.true_step:
                self.logger.info(f"Condition true, executing: {self.true_step.name}")
                return self.true_step.execute(input_data, context)
            elif not condition_result and self.false_step:
                self.logger.info(f"Condition false, executing: {self.false_step.name}")
                return self.false_step.execute(input_data, context)
            else:
                self.logger.info("No step to execute based on condition, passing through")
                return input_data
                
        except Exception as e:
            self.logger.error(f"Conditional step failed: {e}")
            raise


class ParallelStep(PipelineStep):
    """Step that executes multiple steps in parallel."""
    
    def __init__(
        self,
        name: str,
        parallel_steps: List[PipelineStep],
        merge_results: bool = True,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.parallel_steps = parallel_steps
        self.merge_results = merge_results
    
    async def execute_async(self, input_data: Any, context: PipelineContext) -> Any:
        """Execute steps in parallel asynchronously."""
        
        async def run_step(step: PipelineStep) -> Any:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, step.execute, input_data, context)
        
        tasks = [run_step(step) for step in self.parallel_steps]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Parallel step {self.parallel_steps[i].name} failed: {result}")
                raise result
            else:
                successful_results.append(result)
        
        if self.merge_results:
            return successful_results
        else:
            return successful_results[-1]  # Return last result
    
    def execute(self, input_data: Any, context: PipelineContext) -> Any:
        """Execute steps in parallel."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.execute_async(input_data, context))
        finally:
            loop.close()


class ResourceMonitor:
    """Monitor system resources during pipeline execution."""
    
    def __init__(self):
        self.initial_memory: Optional[float] = None
        self.peak_memory: float = 0.0
        self.cpu_samples: List[float] = []
    
    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        if PSUTIL_AVAILABLE and psutil:
            process = psutil.Process()
            self.initial_memory = process.memory_info().rss / 1024 / 1024
            self.peak_memory = self.initial_memory
    
    def update_metrics(self) -> None:
        """Update resource metrics."""
        if PSUTIL_AVAILABLE and psutil:
            process = psutil.Process()
            current_memory = process.memory_info().rss / 1024 / 1024
            self.peak_memory = max(self.peak_memory, current_memory)
            
            cpu_percent = process.cpu_percent()
            self.cpu_samples.append(cpu_percent)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get resource metrics."""
        avg_cpu = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0.0
        memory_delta = self.peak_memory - (self.initial_memory or 0)
        
        return {
            "peak_memory_mb": self.peak_memory,
            "memory_delta_mb": memory_delta,
            "avg_cpu_percent": avg_cpu,
            "max_cpu_percent": max(self.cpu_samples) if self.cpu_samples else 0.0
        }


class MLPipeline:
    """
    Enhanced ML Pipeline with improved monitoring, error handling, and async support.
    """
    
    def __init__(
        self,
        name: str,
        steps: List[PipelineStep],
        config: Optional[Dict[str, Any]] = None,
        enable_checkpoints: bool = True,
        checkpoint_interval: int = 5  # Save checkpoint every N steps
    ):
        self.name = name
        self.steps = steps
        self.config = config or {}
        self.enable_checkpoints = enable_checkpoints
        self.checkpoint_interval = checkpoint_interval
        self.pipeline_id = str(uuid.uuid4())
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Pipeline state
        self.status = PipelineStatus.PENDING
        self.current_step_index = 0
        self.context = PipelineContext(
            pipeline_id=self.pipeline_id,
            pipeline_name=self.name,
            execution_start=datetime.now(),
            config=self.config
        )
        
        # Monitoring
        self.resource_monitor = ResourceMonitor()
        
        # Callbacks
        self.step_callbacks: List[Callable[[StepResult], None]] = []
        self.pipeline_callbacks: List[Callable[[PipelineResult], None]] = []
    
    def add_step_callback(self, callback: Callable[[StepResult], None]) -> None:
        """Add a callback for step completion."""
        self.step_callbacks.append(callback)
    
    def add_pipeline_callback(self, callback: Callable[[PipelineResult], None]) -> None:
        """Add a callback for pipeline completion."""
        self.pipeline_callbacks.append(callback)
    
    def validate_pipeline(self) -> List[str]:
        """Validate pipeline configuration and steps."""
        errors = []
        
        # Validate steps
        for i, step in enumerate(self.steps):
            if not step.validate_config():
                errors.append(f"Step {i} ({step.name}): Invalid configuration")
        
        # Check for name conflicts
        step_names = [step.name for step in self.steps]
        if len(step_names) != len(set(step_names)):
            errors.append("Duplicate step names found")
        
        return errors
    
    def _execute_step(
        self,
        step: PipelineStep,
        input_data: Any,
        step_index: int
    ) -> Tuple[Any, StepResult]:
        """Execute a single step with monitoring."""
        
        step_start_time = time.time()
        memory_before = 0.0
        
        if PSUTIL_AVAILABLE and psutil:
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            # Validate input
            if not step.validate_input(input_data, self.context):
                raise ValueError(f"Input validation failed for step '{step.name}'")
            
            # Check if step can be skipped
            if step.can_skip(input_data, self.context):
                self.logger.info(f"Skipping step: {step.name}")
                return input_data, StepResult(
                    step_name=step.name,
                    step_id=step.step_id,
                    status=StepStatus.SKIPPED,
                    execution_time_seconds=0.0
                )
            
            # Pre-execution
            step.pre_execute(input_data, self.context)
            
            # Execute with timeout
            if step.timeout_seconds:
                # This is a simplified timeout - in production you'd use more sophisticated timeout handling
                start_time = time.time()
                result = step.execute(input_data, self.context)
                if time.time() - start_time > step.timeout_seconds:
                    raise TimeoutError(f"Step {step.name} exceeded timeout of {step.timeout_seconds}s")
            else:
                result = step.execute(input_data, self.context)
            
            # Post-execution
            step.post_execute(result, self.context)
            
            # Calculate metrics
            execution_time = time.time() - step_start_time
            memory_after = 0.0
            if PSUTIL_AVAILABLE and psutil:
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024
            
            step_result = StepResult(
                step_name=step.name,
                step_id=step.step_id,
                status=StepStatus.SUCCESS,
                execution_time_seconds=execution_time,
                memory_delta_mb=memory_after - memory_before if PSUTIL_AVAILABLE else None,
                output_size=len(result) if hasattr(result, '__len__') else None
            )
            
            return result, step_result
            
        except Exception as e:
            step.on_error(e, self.context)
            execution_time = time.time() - step_start_time
            
            step_result = StepResult(
                step_name=step.name,
                step_id=step.step_id,
                status=StepStatus.FAILED,
                execution_time_seconds=execution_time,
                error_message=str(e)
            )
            
            return None, step_result
    
    def execute(
        self,
        input_data: Any,
        resume_from_step: Optional[int] = None,
        stop_on_error: bool = True,
        checkpoint_dir: Optional[Path] = None
    ) -> PipelineResult:
        """
        Execute the pipeline with comprehensive monitoring and error handling.
        
        Args:
            input_data: Initial input data for the pipeline
            resume_from_step: Step index to resume from (for recovery)
            stop_on_error: Whether to stop execution on first error
            checkpoint_dir: Directory to save checkpoints
            
        Returns:
            PipelineResult: Complete execution result with metrics
        """
        
        # Validate pipeline
        validation_errors = self.validate_pipeline()
        if validation_errors:
            return PipelineResult(
                pipeline_id=self.pipeline_id,
                pipeline_name=self.name,
                status=PipelineStatus.FAILED,
                error_message=f"Pipeline validation failed: {'; '.join(validation_errors)}"
            )
        
        # Initialize result
        result = PipelineResult(
            pipeline_id=self.pipeline_id,
            pipeline_name=self.name,
            status=PipelineStatus.RUNNING
        )
        
        # Start monitoring
        self.resource_monitor.start_monitoring()
        start_time = time.time()
        current_data = input_data
        
        try:
            self.status = PipelineStatus.RUNNING
            self.context.execution_start = result.started_at
            
            # Determine starting step
            start_step = resume_from_step if resume_from_step is not None else 0
            self.current_step_index = start_step
            
            self.logger.info(f"Starting pipeline '{self.name}' with {len(self.steps)} steps")
            result.execution_log.append(f"Pipeline started at {result.started_at}")
            
            # Execute steps
            for i in range(start_step, len(self.steps)):
                step = self.steps[i]
                self.current_step_index = i
                self.context.current_step_index = i
                
                # Update resource metrics
                self.resource_monitor.update_metrics()
                
                # Execute step
                step_output, step_result = self._execute_step(step, current_data, i)
                result.step_results.append(step_result)
                
                # Notify step callbacks
                for callback in self.step_callbacks:
                    try:
                        callback(step_result)
                    except Exception as e:
                        self.logger.warning(f"Step callback failed: {e}")
                
                # Handle step result
                if step_result.status == StepStatus.FAILED:
                    if stop_on_error:
                        raise RuntimeError(f"Step '{step.name}' failed: {step_result.error_message}")
                    else:
                        self.logger.warning(f"Step '{step.name}' failed but continuing due to stop_on_error=False")
                        continue
                elif step_result.status == StepStatus.SUCCESS:
                    current_data = step_output
                
                # Save checkpoint if enabled
                if (self.enable_checkpoints and 
                    checkpoint_dir and 
                    (i + 1) % self.checkpoint_interval == 0):
                    self.save_checkpoint(checkpoint_dir / f"checkpoint_{i+1}.json")
                
                # Log progress
                log_msg = f"Step '{step.name}' completed in {step_result.execution_time_seconds:.2f}s"
                self.logger.info(log_msg)
                result.execution_log.append(log_msg)
            
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
            # Calculate final metrics
            total_time = time.time() - start_time
            result.completed_at = datetime.now()
            result.metrics.execution_time_seconds = total_time
            
            # Update resource metrics
            resource_metrics = self.resource_monitor.get_metrics()
            result.metrics.memory_usage_mb = resource_metrics.get("peak_memory_mb", 0.0)
            result.metrics.cpu_usage_percent = resource_metrics.get("avg_cpu_percent", 0.0)
            
            # Calculate processed rows (if applicable)
            result.metrics.rows_processed = len(current_data) if hasattr(current_data, '__len__') else 0
            
            # Notify pipeline callbacks
            for callback in self.pipeline_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    self.logger.warning(f"Pipeline callback failed: {e}")
            
            self.logger.info(f"Pipeline execution completed in {total_time:.2f}s")
        
        return result
    
    async def execute_async(
        self,
        input_data: Any,
        **kwargs
    ) -> PipelineResult:
        """Execute pipeline asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, input_data, **kwargs)
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get comprehensive pipeline information."""
        return {
            "pipeline_id": self.pipeline_id,
            "name": self.name,
            "status": self.status.value,
            "current_step": self.current_step_index,
            "total_steps": len(self.steps),
            "steps": [step.get_step_info() for step in self.steps],
            "config": self.config,
            "enable_checkpoints": self.enable_checkpoints,
            "checkpoint_interval": self.checkpoint_interval
        }
    
    def save_checkpoint(self, checkpoint_path: Path) -> None:
        """Save pipeline checkpoint for recovery."""
        checkpoint_data = {
            "pipeline_id": self.pipeline_id,
            "name": self.name,
            "current_step_index": self.current_step_index,
            "status": self.status.value,
            "context": self.context.model_dump() if PYDANTIC_AVAILABLE else self.context.__dict__,
            "config": self.config,
            "timestamp": datetime.now().isoformat()
        }
        
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load pipeline checkpoint for recovery."""
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
        
        self.pipeline_id = checkpoint_data["pipeline_id"]
        self.name = checkpoint_data["name"]
        self.current_step_index = checkpoint_data["current_step_index"]
        self.status = PipelineStatus(checkpoint_data["status"])
        self.config = checkpoint_data["config"]
        
        # Reconstruct context
        context_data = checkpoint_data["context"]
        if PYDANTIC_AVAILABLE:
            self.context = PipelineContext(**context_data)
        else:
            # Fallback for when pydantic is not available
            self.context = PipelineContext(
                pipeline_id=context_data["pipeline_id"],
                pipeline_name=context_data["pipeline_name"],
                execution_start=datetime.fromisoformat(context_data["execution_start"]),
                current_step_index=context_data["current_step_index"],
                config=context_data.get("config", {}),
                shared_data=context_data.get("shared_data", {})
            )
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")


class PipelineBuilder:
    """Enhanced builder pattern for creating ML pipelines."""
    
    def __init__(self, name: str):
        self.name = name
        self.steps: List[PipelineStep] = []
        self.config: Dict[str, Any] = {}
        self.enable_checkpoints = True
        self.checkpoint_interval = 5
    
    def add_step(self, step: PipelineStep) -> 'PipelineBuilder':
        """Add a step to the pipeline."""
        self.steps.append(step)
        return self
    
    def add_conditional_step(
        self,
        name: str,
        condition_func: Callable[[Any, PipelineContext], bool],
        true_step: PipelineStep,
        false_step: Optional[PipelineStep] = None
    ) -> 'PipelineBuilder':
        """Add a conditional step."""
        conditional_step = ConditionalStep(name, condition_func, true_step, false_step)
        self.steps.append(conditional_step)
        return self
    
    def add_parallel_steps(
        self,
        name: str,
        parallel_steps: List[PipelineStep],
        merge_results: bool = True
    ) -> 'PipelineBuilder':
        """Add parallel execution step."""
        parallel_step = ParallelStep(name, parallel_steps, merge_results)
        self.steps.append(parallel_step)
        return self
    
    def with_config(self, config: Dict[str, Any]) -> 'PipelineBuilder':
        """Set pipeline configuration."""
        self.config.update(config)
        return self
    
    def with_checkpoints(self, enabled: bool = True, interval: int = 5) -> 'PipelineBuilder':
        """Configure checkpointing."""
        self.enable_checkpoints = enabled
        self.checkpoint_interval = interval
        return self
    
    def build(self) -> MLPipeline:
        """Build the pipeline."""
        return MLPipeline(
            self.name,
            self.steps,
            self.config,
            self.enable_checkpoints,
            self.checkpoint_interval
        )


# Convenience functions
def create_pipeline(
    name: str,
    steps: List[PipelineStep],
    config: Optional[Dict[str, Any]] = None
) -> MLPipeline:
    """Create a new ML pipeline."""
    return MLPipeline(name, steps, config)


def create_pipeline_builder(name: str) -> PipelineBuilder:
    """Create a pipeline builder."""
    return PipelineBuilder(name)
