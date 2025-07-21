"""
Genetic Algorithm Implementation for Hyperparameter Optimization
Advanced evolutionary optimization for ML model hyperparameters
"""

import logging
import random
import time
from typing import List, Dict, Any, Callable, Tuple, Optional
from dataclasses import dataclass, field
import json
import math

logger = logging.getLogger(__name__)


@dataclass
class GeneticConfig:
    """Configuration for genetic algorithm optimization."""
    
    # Population parameters
    population_size: int = 50
    max_generations: int = 100
    elite_size: int = 10
    
    # Genetic operators
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    tournament_size: int = 3
    
    # Selection strategies
    selection_method: str = "tournament"  # tournament, roulette, rank
    crossover_method: str = "uniform"  # uniform, single_point, two_point
    mutation_method: str = "gaussian"  # gaussian, uniform, adaptive
    
    # Optimization parameters
    minimize: bool = True
    early_stopping_patience: int = 20
    convergence_threshold: float = 1e-6
    
    # Parallelization
    n_jobs: int = 1
    use_multiprocessing: bool = False
    
    # Logging
    verbose: bool = True
    save_history: bool = True


class Individual:
    """Represents an individual in the genetic algorithm population."""
    
    def __init__(self, genes: Dict[str, Any], fitness: float = float('inf')):
        self.genes = genes
        self.fitness = fitness
        self.age = 0
        self.generation = 0
    
    def __lt__(self, other):
        """Comparison for sorting (assumes minimization)."""
        return self.fitness < other.fitness
    
    def __repr__(self):
        return f"Individual(fitness={self.fitness:.4f}, genes={self.genes})"
    
    def copy(self):
        """Create a deep copy of the individual."""
        return Individual(self.genes.copy(), self.fitness)


class HyperparameterSpace:
    """Defines the search space for hyperparameters."""
    
    def __init__(self):
        self.parameters = {}
        self.constraints = []
    
    def add_continuous(self, name: str, low: float, high: float, log_scale: bool = False):
        """Add a continuous parameter."""
        self.parameters[name] = {
            'type': 'continuous',
            'low': low,
            'high': high,
            'log_scale': log_scale
        }
    
    def add_discrete(self, name: str, choices: List[Any]):
        """Add a discrete parameter."""
        self.parameters[name] = {
            'type': 'discrete',
            'choices': choices
        }
    
    def add_integer(self, name: str, low: int, high: int):
        """Add an integer parameter."""
        self.parameters[name] = {
            'type': 'integer',
            'low': low,
            'high': high
        }
    
    def add_categorical(self, name: str, categories: List[str]):
        """Add a categorical parameter."""
        self.parameters[name] = {
            'type': 'categorical',
            'categories': categories
        }
    
    def add_constraint(self, constraint_func: Callable[[Dict], bool]):
        """Add a constraint function."""
        self.constraints.append(constraint_func)
    
    def sample_random(self) -> Dict[str, Any]:
        """Sample a random point from the parameter space."""
        genes = {}
        
        for name, param in self.parameters.items():
            if param['type'] == 'continuous':
                if param.get('log_scale', False):
                    # Log-uniform sampling
                    log_low = math.log(param['low'])
                    log_high = math.log(param['high'])
                    genes[name] = math.exp(random.uniform(log_low, log_high))
                else:
                    genes[name] = random.uniform(param['low'], param['high'])
            
            elif param['type'] == 'discrete':
                genes[name] = random.choice(param['choices'])
            
            elif param['type'] == 'integer':
                genes[name] = random.randint(param['low'], param['high'])
            
            elif param['type'] == 'categorical':
                genes[name] = random.choice(param['categories'])
        
        # Check constraints
        if self.is_valid(genes):
            return genes
        else:
            # Retry sampling if constraints are violated
            return self.sample_random()
    
    def is_valid(self, genes: Dict[str, Any]) -> bool:
        """Check if genes satisfy all constraints."""
        for constraint in self.constraints:
            if not constraint(genes):
                return False
        return True
    
    def clip_to_bounds(self, genes: Dict[str, Any]) -> Dict[str, Any]:
        """Clip values to parameter bounds."""
        clipped = {}
        
        for name, value in genes.items():
            if name in self.parameters:
                param = self.parameters[name]
                
                if param['type'] == 'continuous':
                    clipped[name] = max(param['low'], min(param['high'], value))
                
                elif param['type'] == 'integer':
                    clipped[name] = max(param['low'], min(param['high'], int(round(value))))
                
                elif param['type'] in ['discrete', 'categorical']:
                    # Keep original value for discrete/categorical
                    clipped[name] = value
                
                else:
                    clipped[name] = value
            else:
                clipped[name] = value
        
        return clipped


class GeneticAlgorithmOptimizer:
    """Genetic Algorithm optimizer for hyperparameter tuning."""
    
    def __init__(self, objective_function: Callable, parameter_space: HyperparameterSpace, 
                 config: GeneticConfig):
        self.objective_function = objective_function
        self.parameter_space = parameter_space
        self.config = config
        
        self.population = []
        self.best_individual = None
        self.history = []
        self.generation = 0
        
        # Statistics
        self.fitness_history = []
        self.diversity_history = []
        self.convergence_history = []
    
    def optimize(self) -> Tuple[Dict[str, Any], float]:
        """Run the genetic algorithm optimization."""
        logger.info(f"Starting genetic algorithm optimization with population size {self.config.population_size}")
        
        start_time = time.time()
        
        # Initialize population
        self._initialize_population()
        
        # Evolution loop
        stagnation_counter = 0
        previous_best_fitness = float('inf')
        
        for generation in range(self.config.max_generations):
            self.generation = generation
            
            # Evaluate population
            self._evaluate_population()
            
            # Update best individual
            current_best = min(self.population)
            if self.best_individual is None or current_best.fitness < self.best_individual.fitness:
                self.best_individual = current_best.copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # Record statistics
            self._record_generation_stats()
            
            # Check convergence
            fitness_improvement = previous_best_fitness - current_best.fitness
            if fitness_improvement < self.config.convergence_threshold:
                stagnation_counter += 1
            
            previous_best_fitness = current_best.fitness
            
            # Early stopping
            if stagnation_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at generation {generation} due to stagnation")
                break
            
            # Log progress
            if self.config.verbose and generation % 10 == 0:
                logger.info(f"Generation {generation}: Best fitness = {current_best.fitness:.6f}")
            
            # Create next generation
            self._evolve_population()
        
        total_time = time.time() - start_time
        
        logger.info(f"Optimization completed in {total_time:.2f} seconds")
        logger.info(f"Best fitness: {self.best_individual.fitness:.6f}")
        logger.info(f"Best parameters: {self.best_individual.genes}")
        
        return self.best_individual.genes, self.best_individual.fitness
    
    def _initialize_population(self):
        """Initialize the population with random individuals."""
        self.population = []
        
        for _ in range(self.config.population_size):
            genes = self.parameter_space.sample_random()
            individual = Individual(genes)
            self.population.append(individual)
        
        logger.info(f"Initialized population with {len(self.population)} individuals")
    
    def _evaluate_population(self):
        """Evaluate fitness for all individuals in the population."""
        if self.config.use_multiprocessing and self.config.n_jobs > 1:
            self._evaluate_parallel()
        else:
            self._evaluate_sequential()
    
    def _evaluate_sequential(self):
        """Evaluate population sequentially."""
        for individual in self.population:
            if individual.fitness == float('inf'):  # Not evaluated yet
                try:
                    individual.fitness = self.objective_function(individual.genes)
                except Exception as e:
                    logger.warning(f"Evaluation failed for individual: {e}")
                    individual.fitness = float('inf') if self.config.minimize else float('-inf')
    
    def _evaluate_parallel(self):
        """Evaluate population in parallel."""
        try:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            
            unevaluated = [ind for ind in self.population if ind.fitness == float('inf')]
            
            if unevaluated:
                with ProcessPoolExecutor(max_workers=self.config.n_jobs) as executor:
                    genes_list = [ind.genes for ind in unevaluated]
                    fitness_values = list(executor.map(self.objective_function, genes_list))
                    
                    for individual, fitness in zip(unevaluated, fitness_values):
                        individual.fitness = fitness
                        
        except Exception as e:
            logger.warning(f"Parallel evaluation failed, falling back to sequential: {e}")
            self._evaluate_sequential()
    
    def _evolve_population(self):
        """Create the next generation."""
        new_population = []
        
        # Elitism: keep best individuals
        sorted_population = sorted(self.population)
        elite = sorted_population[:self.config.elite_size]
        new_population.extend([ind.copy() for ind in elite])
        
        # Generate offspring
        while len(new_population) < self.config.population_size:
            # Selection
            parent1 = self._select_parent()
            parent2 = self._select_parent()
            
            # Crossover
            if random.random() < self.config.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            if random.random() < self.config.mutation_rate:
                self._mutate(child1)
            if random.random() < self.config.mutation_rate:
                self._mutate(child2)
            
            # Add to new population
            child1.fitness = float('inf')  # Mark for re-evaluation
            child2.fitness = float('inf')
            child1.generation = self.generation + 1
            child2.generation = self.generation + 1
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        self.population = new_population[:self.config.population_size]
    
    def _select_parent(self) -> Individual:
        """Select a parent using the specified selection method."""
        if self.config.selection_method == "tournament":
            return self._tournament_selection()
        elif self.config.selection_method == "roulette":
            return self._roulette_selection()
        elif self.config.selection_method == "rank":
            return self._rank_selection()
        else:
            return random.choice(self.population)
    
    def _tournament_selection(self) -> Individual:
        """Tournament selection."""
        tournament = random.sample(self.population, self.config.tournament_size)
        return min(tournament) if self.config.minimize else max(tournament)
    
    def _roulette_selection(self) -> Individual:
        """Roulette wheel selection."""
        # Convert fitness to selection probabilities
        fitness_values = [ind.fitness for ind in self.population]
        
        if self.config.minimize:
            # For minimization, invert fitness
            max_fitness = max(fitness_values)
            weights = [max_fitness - f + 1e-8 for f in fitness_values]
        else:
            weights = [f + 1e-8 for f in fitness_values]
        
        total_weight = sum(weights)
        r = random.uniform(0, total_weight)
        
        cumulative_weight = 0
        for individual, weight in zip(self.population, weights):
            cumulative_weight += weight
            if cumulative_weight >= r:
                return individual
        
        return self.population[-1]  # Fallback
    
    def _rank_selection(self) -> Individual:
        """Rank-based selection."""
        sorted_population = sorted(self.population, reverse=not self.config.minimize)
        
        # Linear ranking
        n = len(sorted_population)
        weights = [i + 1 for i in range(n)]
        total_weight = sum(weights)
        
        r = random.uniform(0, total_weight)
        cumulative_weight = 0
        
        for individual, weight in zip(sorted_population, weights):
            cumulative_weight += weight
            if cumulative_weight >= r:
                return individual
        
        return sorted_population[-1]  # Fallback
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Perform crossover between two parents."""
        if self.config.crossover_method == "uniform":
            return self._uniform_crossover(parent1, parent2)
        elif self.config.crossover_method == "single_point":
            return self._single_point_crossover(parent1, parent2)
        elif self.config.crossover_method == "two_point":
            return self._two_point_crossover(parent1, parent2)
        else:
            return parent1.copy(), parent2.copy()
    
    def _uniform_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Uniform crossover."""
        child1_genes = {}
        child2_genes = {}
        
        for key in parent1.genes:
            if random.random() < 0.5:
                child1_genes[key] = parent1.genes[key]
                child2_genes[key] = parent2.genes[key]
            else:
                child1_genes[key] = parent2.genes[key]
                child2_genes[key] = parent1.genes[key]
        
        return Individual(child1_genes), Individual(child2_genes)
    
    def _single_point_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Single-point crossover."""
        keys = list(parent1.genes.keys())
        if len(keys) <= 1:
            return parent1.copy(), parent2.copy()
        
        crossover_point = random.randint(1, len(keys) - 1)
        
        child1_genes = {}
        child2_genes = {}
        
        for i, key in enumerate(keys):
            if i < crossover_point:
                child1_genes[key] = parent1.genes[key]
                child2_genes[key] = parent2.genes[key]
            else:
                child1_genes[key] = parent2.genes[key]
                child2_genes[key] = parent1.genes[key]
        
        return Individual(child1_genes), Individual(child2_genes)
    
    def _two_point_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Two-point crossover."""
        keys = list(parent1.genes.keys())
        if len(keys) <= 2:
            return self._single_point_crossover(parent1, parent2)
        
        point1, point2 = sorted(random.sample(range(1, len(keys)), 2))
        
        child1_genes = {}
        child2_genes = {}
        
        for i, key in enumerate(keys):
            if point1 <= i < point2:
                child1_genes[key] = parent2.genes[key]
                child2_genes[key] = parent1.genes[key]
            else:
                child1_genes[key] = parent1.genes[key]
                child2_genes[key] = parent2.genes[key]
        
        return Individual(child1_genes), Individual(child2_genes)
    
    def _mutate(self, individual: Individual):
        """Mutate an individual."""
        if self.config.mutation_method == "gaussian":
            self._gaussian_mutation(individual)
        elif self.config.mutation_method == "uniform":
            self._uniform_mutation(individual)
        elif self.config.mutation_method == "adaptive":
            self._adaptive_mutation(individual)
    
    def _gaussian_mutation(self, individual: Individual):
        """Gaussian mutation."""
        for param_name in individual.genes:
            if param_name in self.parameter_space.parameters:
                param = self.parameter_space.parameters[param_name]
                
                if param['type'] == 'continuous':
                    # Gaussian noise proportional to parameter range
                    param_range = param['high'] - param['low']
                    noise = random.gauss(0, param_range * 0.1)
                    individual.genes[param_name] += noise
                
                elif param['type'] == 'integer':
                    # Integer mutation
                    if random.random() < 0.1:  # 10% chance to mutate
                        param_range = param['high'] - param['low']
                        noise = random.gauss(0, max(1, param_range * 0.1))
                        individual.genes[param_name] += int(noise)
                
                elif param['type'] in ['discrete', 'categorical']:
                    # Random replacement
                    if random.random() < 0.1:
                        choices = param.get('choices', param.get('categories', []))
                        individual.genes[param_name] = random.choice(choices)
        
        # Clip to bounds
        individual.genes = self.parameter_space.clip_to_bounds(individual.genes)
    
    def _uniform_mutation(self, individual: Individual):
        """Uniform mutation."""
        for param_name in individual.genes:
            if random.random() < 0.1:  # 10% chance per parameter
                if param_name in self.parameter_space.parameters:
                    param = self.parameter_space.parameters[param_name]
                    
                    if param['type'] == 'continuous':
                        individual.genes[param_name] = random.uniform(param['low'], param['high'])
                    elif param['type'] == 'integer':
                        individual.genes[param_name] = random.randint(param['low'], param['high'])
                    elif param['type'] in ['discrete', 'categorical']:
                        choices = param.get('choices', param.get('categories', []))
                        individual.genes[param_name] = random.choice(choices)
    
    def _adaptive_mutation(self, individual: Individual):
        """Adaptive mutation based on population diversity."""
        # Calculate mutation strength based on population diversity
        diversity = self._calculate_population_diversity()
        
        # Higher diversity -> lower mutation rate, lower diversity -> higher mutation rate
        adaptive_rate = max(0.01, min(0.5, 1.0 / (1.0 + diversity)))
        
        for param_name in individual.genes:
            if random.random() < adaptive_rate:
                if param_name in self.parameter_space.parameters:
                    param = self.parameter_space.parameters[param_name]
                    
                    if param['type'] == 'continuous':
                        param_range = param['high'] - param['low']
                        noise = random.gauss(0, param_range * adaptive_rate)
                        individual.genes[param_name] += noise
                    
                    # Similar for other parameter types...
        
        individual.genes = self.parameter_space.clip_to_bounds(individual.genes)
    
    def _calculate_population_diversity(self) -> float:
        """Calculate population diversity."""
        if len(self.population) < 2:
            return 0.0
        
        total_distance = 0.0
        count = 0
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                distance = self._individual_distance(self.population[i], self.population[j])
                total_distance += distance
                count += 1
        
        return total_distance / count if count > 0 else 0.0
    
    def _individual_distance(self, ind1: Individual, ind2: Individual) -> float:
        """Calculate distance between two individuals."""
        distance = 0.0
        
        for param_name in ind1.genes:
            if param_name in ind2.genes and param_name in self.parameter_space.parameters:
                param = self.parameter_space.parameters[param_name]
                
                if param['type'] in ['continuous', 'integer']:
                    # Normalized distance
                    param_range = param['high'] - param['low']
                    if param_range > 0:
                        norm_dist = abs(ind1.genes[param_name] - ind2.genes[param_name]) / param_range
                        distance += norm_dist ** 2
                
                elif param['type'] in ['discrete', 'categorical']:
                    # Binary distance
                    distance += 0 if ind1.genes[param_name] == ind2.genes[param_name] else 1
        
        return math.sqrt(distance)
    
    def _record_generation_stats(self):
        """Record statistics for the current generation."""
        fitness_values = [ind.fitness for ind in self.population]
        
        stats = {
            'generation': self.generation,
            'best_fitness': min(fitness_values),
            'worst_fitness': max(fitness_values),
            'mean_fitness': sum(fitness_values) / len(fitness_values),
            'diversity': self._calculate_population_diversity()
        }
        
        self.history.append(stats)
        
        if self.config.save_history:
            self.fitness_history.append(stats['best_fitness'])
            self.diversity_history.append(stats['diversity'])
    
    def get_optimization_history(self) -> Dict[str, Any]:
        """Get complete optimization history."""
        return {
            'config': self.config.__dict__,
            'generation_stats': self.history,
            'best_individual': {
                'genes': self.best_individual.genes if self.best_individual else None,
                'fitness': self.best_individual.fitness if self.best_individual else None
            },
            'fitness_history': self.fitness_history,
            'diversity_history': self.diversity_history
        }


# Export main classes
__all__ = [
    "GeneticConfig",
    "Individual", 
    "HyperparameterSpace",
    "GeneticAlgorithmOptimizer"
]
