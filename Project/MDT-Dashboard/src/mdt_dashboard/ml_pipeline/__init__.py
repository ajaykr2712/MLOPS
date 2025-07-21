"""
Advanced ML Pipeline module for MDT Dashboard.
Provides comprehensive ML pipeline components for enterprise-grade ML operations.
"""

from .pipeline import MLPipeline, PipelineStep
from .components import (
    DataLoader,
    Preprocessor,
    FeatureEngineering,
    ModelTrainer,
    ModelDeployer
)

# Advanced ML components
from .neural_networks import (
    ModelArchitectureConfig,
    TransformerModel,
    DiffusionModel,
    GraphNeuralNetwork,
    MultimodalModel,
    FoundationModelTrainer,
    ContrastiveLearning
)

from .distributed_training import (
    DistributedConfig,
    DistributedTrainingManager,
    ModelParallelism,
    GradientAccumulation,
    MemoryOptimizer,
    PipelineParallelism,
    AutoScaler,
    DistributedOptimizer
)

from .evaluation import (
    EvaluationConfig,
    ModelEvaluator
)

from .genetic_optimization import (
    GeneticConfig,
    Individual,
    HyperparameterSpace,
    GeneticAlgorithmOptimizer
)

from .vector_store import (
    VectorConfig,
    VectorDatabase,
    FAISSVectorDB,
    ChromaVectorDB,
    PineconeVectorDB,
    VectorFeatureStore,
    VectorSearchEngine
)

__all__ = [
    # Basic pipeline components
    "MLPipeline",
    "PipelineStep", 
    "DataLoader",
    "Preprocessor",
    "FeatureEngineering",
    "ModelTrainer",
    "ModelDeployer",
    
    # Neural networks and foundation models
    "ModelArchitectureConfig",
    "TransformerModel",
    "DiffusionModel",
    "GraphNeuralNetwork",
    "MultimodalModel",
    "FoundationModelTrainer",
    "ContrastiveLearning",
    
    # Distributed training
    "DistributedConfig",
    "DistributedTrainingManager",
    "ModelParallelism", 
    "GradientAccumulation",
    "MemoryOptimizer",
    "PipelineParallelism",
    "AutoScaler",
    "DistributedOptimizer",
    
    # Advanced evaluation
    "EvaluationConfig",
    "ModelEvaluator",
    
    # Genetic optimization
    "GeneticConfig",
    "Individual",
    "HyperparameterSpace",
    "GeneticAlgorithmOptimizer",
    
    # Vector databases
    "VectorConfig",
    "VectorDatabase",
    "FAISSVectorDB",
    "ChromaVectorDB",
    "PineconeVectorDB", 
    "VectorFeatureStore",
    "VectorSearchEngine"
]
