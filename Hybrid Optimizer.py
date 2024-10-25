# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:16:39 2024

@author: pahar
"""

class HybridOptimizer:
    def __init__(self, population_size, n_generations, bounds):
        self.population_size = population_size
        self.n_generations = n_generations
        self.bounds = bounds

        # PSO Parameters
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5

        # RDA Parameters
        self.roar_factor = 0.5

    def initialize_population(self):
        """Initialize population with random parameters."""
        population = []
        for _ in range(self.population_size):
            individual = {
                'lstm_units': np.random.randint(self.bounds['lstm_units'][0], self.bounds['lstm_units'][1]),
                'dense_units': np.random.randint(self.bounds['dense_units'][0], self.bounds['dense_units'][1]),
                'learning_rate': np.random.uniform(self.bounds['learning_rate'][0], self.bounds['learning_rate'][1]),
                'velocity': {
                    'lstm_units': 0,
                    'dense_units': 0,
                    'learning_rate': 0
                }
            }
            population.append(individual)
        return population

    def fitness_function(self, individual):
        """Fitness function to evaluate individuals (minimize error)."""

        return np.random.random()

    def apply_ga(self, parent1, parent2):
        """Apply Genetic Algorithm crossover and mutation."""
        child = {}
        for param in parent1:
            if np.random.random() < 0.5:
                child[param] = parent1[param]
            else:
                child[param] = parent2[param]

        # Mutation
        for param in child:
            if isinstance(child[param], (int, float)):
                if np.random.random() < 0.2:
                    if param == 'learning_rate':
                        child[param] *= np.random.uniform(0.8, 1.2)
                        child[param] = np.clip(child[param], self.bounds[param][0], self.bounds[param][1])
                    else:
                        child[param] = np.clip(child[param] + np.random.randint(-3, 4), self.bounds[param][0], self.bounds[param][1])

        return child

    def apply_pso(self, particle, personal_best, global_best):
        """Apply Particle Swarm Optimization (PSO) velocity and position update."""
        for param in particle:
            if param != 'velocity':  # Skip velocity param itself
                r1, r2 = np.random.random(), np.random.random()
                velocity_update = (
                    self.w * particle['velocity'][param]
                    + self.c1 * r1 * (personal_best[param] - particle[param])
                    + self.c2 * r2 * (global_best[param] - particle[param])
                )
                particle['velocity'][param] = velocity_update
                particle[param] += particle['velocity'][param]

                # Ensure bounds
                particle[param] = np.clip(particle[param], self.bounds[param][0], self.bounds[param][1])

    def apply_gsa(self, population, best_individual):
        """Apply Gravitational Search Algorithm (GSA) update."""
        G = 100  # Gravitational constant
        for individual in population:
            force = {}
            for param in individual:
                if param != 'velocity':
                    force[param] = np.random.uniform(0, 1) * (best_individual[param] - individual[param])
            for param in individual:
                if param != 'velocity':
                    acceleration = force[param] / np.random.uniform(1, 5)
                    individual[param] += G * acceleration  # Movement based on gravitational force

                    # Ensure bounds
                    individual[param] = np.clip(individual[param], self.bounds[param][0], self.bounds[param][1])

    def apply_rda(self, population, dominant_stags):
        """Apply Red Deer Algorithm (RDA) mating and roaring contest."""
        new_offspring = []

        stags = np.random.choice(dominant_stags, size=len(dominant_stags), replace=False)
        for i, stag in enumerate(stags):
            for hind in population:
                if np.random.random() < self.roar_factor:
                    offspring = self.apply_ga(stag, hind)
                    new_offspring.append(offspring)

        # Replace part of the population with new offspring
        population[:len(new_offspring)] = new_offspring[:len(population)]


        for individual in population:
            for param in individual:
                if param != 'velocity':
                    individual[param] = np.clip(individual[param], self.bounds[param][0], self.bounds[param][1])

    def apply_dbo(self, population, global_best):
        """Apply Dung Beetle Optimization (DBO) directional movement toward global best."""
        alpha, beta = 0.5, 0.3
        for individual in population:
            for param in individual:
                if param != 'velocity':
                    random_movement = np.random.uniform(-1, 1)
                    individual[param] += alpha * (global_best[param] - individual[param]) + beta * random_movement

                    # Ensure bounds
                    individual[param] = np.clip(individual[param], self.bounds[param][0], self.bounds[param][1])

    def optimize(self):
        population = self.initialize_population()
        global_best = None
        global_best_fitness = float('inf')
        personal_best = [deepcopy(individual) for individual in population]

        print(f"Initial Population Size: {len(population)}")  # Debug: Check initial population size

        for generation in range(self.n_generations):
            print(f"Generation {generation + 1}, Population Size: {len(population)}")

            for i, individual in enumerate(population):
                fitness = self.fitness_function(individual)
                if fitness < global_best_fitness:
                    global_best_fitness = fitness
                    global_best = deepcopy(individual)

                # Check personal best
                if fitness < self.fitness_function(personal_best[i]):
                    personal_best[i] = deepcopy(individual)

            # Apply the hybrid techniques
            for i in range(len(population)):
                self.apply_ga(population[i], personal_best[i])
                self.apply_pso(population[i], personal_best[i], global_best)

            self.apply_gsa(population, global_best)
            dominant_stags = population[:len(population) // 2]

            self.apply_rda(population, dominant_stags)
            self.apply_dbo(population, global_best)

            print(f"Generation {generation + 1}/{self.n_generations}, Global Best Fitness: {global_best_fitness}")

        return global_best, global_best_fitness
