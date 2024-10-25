# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:15:20 2024

@author: pahar
"""


class DBOLSTMOptimizer:
    def __init__(self, n_beetles, n_iterations, bounds):
        """
        bounds: dictionary with keys 'lstm_units', 'dense_units', 'learning_rate'
               and values as tuples of (min, max)
        """
        self.n_beetles = n_beetles
        self.n_iterations = n_iterations
        self.bounds = bounds

    def initialize_beetles(self):
        """Initialize beetle positions"""
        beetles = []
        for _ in range(self.n_beetles):
            beetle = {
                'lstm_units': np.random.randint(self.bounds['lstm_units'][0],
                                              self.bounds['lstm_units'][1]),
                'dense_units': np.random.randint(self.bounds['dense_units'][0],
                                               self.bounds['dense_units'][1]),
                'learning_rate': np.random.uniform(self.bounds['learning_rate'][0],
                                                 self.bounds['learning_rate'][1])
            }
            beetles.append(beetle)
        return beetles

    def optimize(self, X_train, y_train, X_val, y_val, timesteps, features):
        beetles = self.initialize_beetles()
        best_beetle = None
        best_fitness = float('inf')

        for iteration in range(self.n_iterations):
            for i, beetle in enumerate(beetles):
                # Create and evaluate model with current parameters
                model = create_base_lstm_model(
                    timesteps=timesteps,
                    features=features,
                    lstm_units=beetle['lstm_units'],
                    dense_units=beetle['dense_units'],
                    learning_rate=beetle['learning_rate']
                )

                _, fitness = train_evaluate_model(
                    model, X_train, y_train, X_val, y_val
                )

                # Update best solution if needed
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_beetle = deepcopy(beetle)

            # Update beetle positions
            for beetle in beetles:
                # Rolling behavior
                for param in beetle:
                    if param == 'learning_rate':
                        beetle[param] += np.random.uniform(-0.1, 0.1) * beetle[param]
                    else:
                        beetle[param] += np.random.randint(-5, 6)

                # Pushing behavior towards best beetle
                for param in beetle:
                    if param == 'learning_rate':
                        beetle[param] += 0.2 * (best_beetle[param] - beetle[param])
                    else:
                        beetle[param] += int(0.2 * (best_beetle[param] - beetle[param]))

                # Ensure bounds
                beetle['lstm_units'] = np.clip(beetle['lstm_units'],
                                             self.bounds['lstm_units'][0],
                                             self.bounds['lstm_units'][1])
                beetle['dense_units'] = np.clip(beetle['dense_units'],
                                              self.bounds['dense_units'][0],
                                              self.bounds['dense_units'][1])
                beetle['learning_rate'] = np.clip(beetle['learning_rate'],
                                                self.bounds['learning_rate'][0],
                                                self.bounds['learning_rate'][1])

            print(f"Iteration {iteration + 1}/{self.n_iterations}, Best fitness: {best_fitness}")

        return best_beetle, best_fitness

"""## GA Algo"""

class GALSTMOptimizer:
    def __init__(self, population_size, n_generations, bounds):
        self.population_size = population_size
        self.n_generations = n_generations
        self.bounds = bounds

    def initialize_population(self):
        """Initialize population with random parameters"""
        population = []
        for _ in range(self.population_size):
            individual = {
                'lstm_units': np.random.randint(self.bounds['lstm_units'][0],
                                              self.bounds['lstm_units'][1]),
                'dense_units': np.random.randint(self.bounds['dense_units'][0],
                                               self.bounds['dense_units'][1]),
                'learning_rate': np.random.uniform(self.bounds['learning_rate'][0],
                                                 self.bounds['learning_rate'][1])
            }
            population.append(individual)
        return population

    def crossover(self, parent1, parent2):
        """Perform crossover between two parents"""
        child = {}
        for param in parent1:
            if np.random.random() < 0.5:
                child[param] = parent1[param]
            else:
                child[param] = parent2[param]
        return child

    def mutate(self, individual):
        """Perform mutation on an individual"""
        mutated = deepcopy(individual)
        for param in mutated:
            if np.random.random() < 0.2:  # 20% mutation rate
                if param == 'learning_rate':
                    mutated[param] *= np.random.uniform(0.8, 1.2)
                else:
                    mutated[param] += np.random.randint(-3, 4)

                # Ensure bounds
                if param == 'learning_rate':
                    mutated[param] = np.clip(mutated[param],
                                           self.bounds[param][0],
                                           self.bounds[param][1])
                else:
                    mutated[param] = np.clip(mutated[param],
                                           self.bounds[param][0],
                                           self.bounds[param][1])
        return mutated

    def optimize(self, X_train, y_train, X_val, y_val, timesteps, features):
        population = self.initialize_population()
        best_individual = None
        best_fitness = float('inf')

        for generation in range(self.n_generations):
            # Evaluate fitness for each individual
            fitness_scores = []
            for individual in population:
                model = create_base_lstm_model(
                    timesteps=timesteps,
                    features=features,
                    lstm_units=individual['lstm_units'],
                    dense_units=individual['dense_units'],
                    learning_rate=individual['learning_rate']
                )

                _, fitness = train_evaluate_model(
                    model, X_train, y_train, X_val, y_val
                )
                fitness_scores.append(fitness)

                if fitness < best_fitness:
                    best_fitness = fitness
                    best_individual = deepcopy(individual)

            # Create new population
            new_population = [best_individual]  # Elitism

            while len(new_population) < self.population_size:
                # Tournament selection
                tournament_size = 3
                parent1_idx = np.argmin([fitness_scores[i] for i in
                                       np.random.choice(len(population), tournament_size)])
                parent2_idx = np.argmin([fitness_scores[i] for i in
                                       np.random.choice(len(population), tournament_size)])

                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]

                # Crossover
                child = self.crossover(parent1, parent2)

                # Mutation
                child = self.mutate(child)

                new_population.append(child)

            population = new_population
            print(f"Generation {generation + 1}/{self.n_generations}, Best fitness: {best_fitness}")

        return best_individual, best_fitness

"""## DBO-CNN"""


def create_cnn_lstm_model(timesteps, features, cnn_filters, cnn_kernel_size,
                         lstm_units, dense_units, learning_rate=0.001):
    """Create CNN-LSTM model with configurable parameters"""
    input_layer = Input(shape=(timesteps, features))

    # CNN layers
    conv = Conv1D(filters=cnn_filters, kernel_size=cnn_kernel_size,
                 activation='relu')(input_layer)
    pool = MaxPooling1D(pool_size=2)(conv)
    flatten = Flatten()(pool)

    # LSTM layer
    lstm_output = LSTM(int(lstm_units))(flatten)

    # Dense layers
    dense_1 = Dense(dense_units, activation='relu')(lstm_output)
    output = Dense(1, activation='linear')(dense_1)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    return model

class DBOCNNLSTMOptimizer:
    def __init__(self, n_beetles, n_iterations, bounds):
        """
        bounds: dictionary with keys 'cnn_filters', 'cnn_kernel_size', 'lstm_units',
               'dense_units', 'learning_rate' and values as tuples of (min, max)
        """
        self.n_beetles = n_beetles
        self.n_iterations = n_iterations
        self.bounds = bounds

    def initialize_beetles(self):
        """Initialize beetle positions"""
        beetles = []
        for _ in range(self.n_beetles):
            beetle = {
                'cnn_filters': np.random.randint(self.bounds['cnn_filters'][0],
                                               self.bounds['cnn_filters'][1]),
                'cnn_kernel_size': np.random.randint(self.bounds['cnn_kernel_size'][0],
                                                   self.bounds['cnn_kernel_size'][1]),
                'lstm_units': np.random.randint(self.bounds['lstm_units'][0],
                                              self.bounds['lstm_units'][1]),
                'dense_units': np.random.randint(self.bounds['dense_units'][0],
                                               self.bounds['dense_units'][1]),
                'learning_rate': np.random.uniform(self.bounds['learning_rate'][0],
                                                 self.bounds['learning_rate'][1])
            }
            beetles.append(beetle)
        return beetles

    def optimize(self, X_train, y_train, X_val, y_val, timesteps, features):
        beetles = self.initialize_beetles()
        best_beetle = None
        best_fitness = float('inf')

        for iteration in range(self.n_iterations):
            for i, beetle in enumerate(beetles):
                # Create and evaluate model with current parameters
                model = create_cnn_lstm_model(
                    timesteps=timesteps,
                    features=features,
                    cnn_filters=beetle['cnn_filters'],
                    cnn_kernel_size=beetle['cnn_kernel_size'],
                    lstm_units=beetle['lstm_units'],
                    dense_units=beetle['dense_units'],
                    learning_rate=beetle['learning_rate']
                )

                _, fitness = train_evaluate_model(
                    model, X_train, y_train, X_val, y_val
                )

                # Update best solution if needed
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_beetle = deepcopy(beetle)

            # Update beetle positions
            for beetle in beetles:
                # Rolling behavior
                for param in beetle:
                    if param == 'learning_rate':
                        beetle[param] += np.random.uniform(-0.1, 0.1) * beetle[param]
                    else:
                        beetle[param] += np.random.randint(-5, 6)

                # Pushing behavior towards best beetle
                for param in beetle:
                    if param == 'learning_rate':
                        beetle[param] += 0.2 * (best_beetle[param] - beetle[param])
                    else:
                        beetle[param] += int(0.2 * (best_beetle[param] - beetle[param]))

                # Ensure bounds
                for param in beetle:
                    if param == 'learning_rate':
                        beetle[param] = np.clip(beetle[param],
                                              self.bounds[param][0],
                                              self.bounds[param][1])
                    else:
                        beetle[param] = int(np.clip(beetle[param],
                                                  self.bounds[param][0],
                                                  self.bounds[param][1]))

            print(f"Iteration {iteration + 1}/{self.n_iterations}, Best fitness: {best_fitness}")

        return best_beetle, best_fitness


def create_cnn_lstm_model(timesteps, features, cnn_filters, cnn_kernel_size,
                         lstm_units, dense_units, learning_rate=0.001):
    """Create CNN-LSTM model with configurable parameters"""
    input_layer = Input(shape=(timesteps, features))

    # CNN layers
    conv = Conv1D(filters=cnn_filters, kernel_size=cnn_kernel_size,
                 activation='relu')(input_layer)
    pool = MaxPooling1D(pool_size=2)(conv)
    flatten = Flatten()(pool)

    # LSTM layer
    lstm_output = LSTM(int(lstm_units))(flatten)

    # Dense layers
    dense_1 = Dense(dense_units, activation='relu')(lstm_output)
    output = Dense(1, activation='linear')(dense_1)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    return model

class DBOCNNLSTMOptimizer:
    def __init__(self, n_beetles, n_iterations, bounds):
        """
        bounds: dictionary with keys 'cnn_filters', 'cnn_kernel_size', 'lstm_units',
               'dense_units', 'learning_rate' and values as tuples of (min, max)
        """
        self.n_beetles = n_beetles
        self.n_iterations = n_iterations
        self.bounds = bounds

    def initialize_beetles(self):
        """Initialize beetle positions"""
        beetles = []
        for _ in range(self.n_beetles):
            beetle = {
                'cnn_filters': np.random.randint(self.bounds['cnn_filters'][0],
                                               self.bounds['cnn_filters'][1]),
                'cnn_kernel_size': np.random.randint(self.bounds['cnn_kernel_size'][0],
                                                   self.bounds['cnn_kernel_size'][1]),
                'lstm_units': np.random.randint(self.bounds['lstm_units'][0],
                                              self.bounds['lstm_units'][1]),
                'dense_units': np.random.randint(self.bounds['dense_units'][0],
                                               self.bounds['dense_units'][1]),
                'learning_rate': np.random.uniform(self.bounds['learning_rate'][0],
                                                 self.bounds['learning_rate'][1])
            }
            beetles.append(beetle)
        return beetles

    def optimize(self, X_train, y_train, X_val, y_val, timesteps, features):
        beetles = self.initialize_beetles()
        best_beetle = None
        best_fitness = float('inf')

        for iteration in range(self.n_iterations):
            for i, beetle in enumerate(beetles):
                # Create and evaluate model with current parameters
                model = create_cnn_lstm_model(
                    timesteps=timesteps,
                    features=features,
                    cnn_filters=beetle['cnn_filters'],
                    cnn_kernel_size=beetle['cnn_kernel_size'],
                    lstm_units=beetle['lstm_units'],
                    dense_units=beetle['dense_units'],
                    learning_rate=beetle['learning_rate']
                )

                _, fitness = train_evaluate_model(
                    model, X_train, y_train, X_val, y_val
                )

                # Update best solution if needed
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_beetle = deepcopy(beetle)

            # Update beetle positions
            for beetle in beetles:
                # Rolling behavior
                for param in beetle:
                    if param == 'learning_rate':
                        beetle[param] += np.random.uniform(-0.1, 0.1) * beetle[param]
                    else:
                        beetle[param] += np.random.randint(-5, 6)

                # Pushing behavior towards best beetle
                for param in beetle:
                    if param == 'learning_rate':
                        beetle[param] += 0.2 * (best_beetle[param] - beetle[param])
                    else:
                        beetle[param] += int(0.2 * (best_beetle[param] - beetle[param]))

                # Ensure bounds
                for param in beetle:
                    if param == 'learning_rate':
                        beetle[param] = np.clip(beetle[param],
                                              self.bounds[param][0],
                                              self.bounds[param][1])
                    else:
                        beetle[param] = int(np.clip(beetle[param],
                                                  self.bounds[param][0],
                                                  self.bounds[param][1]))

            print(f"Iteration {iteration + 1}/{self.n_iterations}, Best fitness: {best_fitness}")

        return best_beetle, best_fitness

"""## PSO-GSA-RDA"""

class PSOLSTMOptimizer:
    def __init__(self, population_size, n_iterations, bounds, w=0.7, c1=1.5, c2=1.5):
        self.population_size = population_size
        self.n_iterations = n_iterations
        self.bounds = bounds
        self.w = w  # inertia weight
        self.c1 = c1  # cognitive weight
        self.c2 = c2  # social weight

    def initialize_population(self):
        """Initialize particles with random positions and velocities"""
        population = []
        for _ in range(self.population_size):
            particle = {
                'position': {
                    'lstm_units': np.random.randint(self.bounds['lstm_units'][0],
                                                  self.bounds['lstm_units'][1]),
                    'dense_units': np.random.randint(self.bounds['dense_units'][0],
                                                   self.bounds['dense_units'][1]),
                    'learning_rate': np.random.uniform(self.bounds['learning_rate'][0],
                                                     self.bounds['learning_rate'][1])
                },
                'velocity': {
                    'lstm_units': 0,
                    'dense_units': 0,
                    'learning_rate': 0
                },
                'best_position': None,
                'best_fitness': float('inf')
            }
            particle['best_position'] = deepcopy(particle['position'])
            population.append(particle)
        return population

    def update_particle(self, particle, global_best_position):
        """Update particle velocity and position"""
        for param in particle['position']:
            r1, r2 = np.random.random(2)

            # Update velocity
            particle['velocity'][param] = (
                self.w * particle['velocity'][param] +
                self.c1 * r1 * (particle['best_position'][param] - particle['position'][param]) +
                self.c2 * r2 * (global_best_position[param] - particle['position'][param])
            )

            # Update position
            if param == 'learning_rate':
                particle['position'][param] += particle['velocity'][param]
                particle['position'][param] = np.clip(particle['position'][param],
                                                    self.bounds[param][0],
                                                    self.bounds[param][1])
            else:
                particle['position'][param] = int(particle['position'][param] + particle['velocity'][param])
                particle['position'][param] = np.clip(particle['position'][param],
                                                    self.bounds[param][0],
                                                    self.bounds[param][1])
        return particle

    def optimize(self, X_train, y_train, X_val, y_val, timesteps, features):
        population = self.initialize_population()
        global_best_position = None
        global_best_fitness = float('inf')

        for iteration in range(self.n_iterations):
            for particle in population:
                model = create_base_lstm_model(
                    timesteps=timesteps,
                    features=features,
                    lstm_units=particle['position']['lstm_units'],
                    dense_units=particle['position']['dense_units'],
                    learning_rate=particle['position']['learning_rate']
                )

                _, fitness = train_evaluate_model(model, X_train, y_train, X_val, y_val)

                # Update particle's best position
                if fitness < particle['best_fitness']:
                    particle['best_fitness'] = fitness
                    particle['best_position'] = deepcopy(particle['position'])

                # Update global best position
                if fitness < global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = deepcopy(particle['position'])

            # Update all particles
            for particle in population:
                particle = self.update_particle(particle, global_best_position)

            print(f"Iteration {iteration + 1}/{self.n_iterations}, Best fitness: {global_best_fitness}")

        return global_best_position, global_best_fitness


class GSALSTMOptimizer:
    def __init__(self, population_size, n_iterations, bounds, G0=100, alpha=20):
        self.population_size = population_size
        self.n_iterations = n_iterations
        self.bounds = bounds
        self.G0 = G0  # initial gravitational constant
        self.alpha = alpha  # decay rate

    def initialize_population(self):
        """Initialize agents with random positions and masses"""
        population = []
        for _ in range(self.population_size):
            agent = {
                'position': {
                    'lstm_units': np.random.randint(self.bounds['lstm_units'][0],
                                                  self.bounds['lstm_units'][1]),
                    'dense_units': np.random.randint(self.bounds['dense_units'][0],
                                                   self.bounds['dense_units'][1]),
                    'learning_rate': np.random.uniform(self.bounds['learning_rate'][0],
                                                     self.bounds['learning_rate'][1])
                },
                'velocity': {
                    'lstm_units': 0,
                    'dense_units': 0,
                    'learning_rate': 0
                },
                'mass': 0,
                'fitness': float('inf')
            }
            population.append(agent)
        return population

    def calculate_masses(self, fitness_scores):
        """Calculate masses based on fitness scores"""
        worst = max(fitness_scores)
        best = min(fitness_scores)
        masses = (fitness_scores - worst) / (best - worst + 1e-10)
        masses = masses / (np.sum(masses) + 1e-10)
        return masses

    def update_agent(self, agent, population, masses, G, iteration):
        """Update agent velocity and position using gravitational forces"""
        for param in agent['position']:
            force = 0
            for other_agent, mass in zip(population, masses):
                if other_agent != agent:
                    distance = abs(other_agent['position'][param] - agent['position'][param]) + 1e-10
                    force += G * mass * (other_agent['position'][param] - agent['position'][param]) / distance

            # Update velocity and position
            agent['velocity'][param] = np.random.random() * agent['velocity'][param] + force

            if param == 'learning_rate':
                agent['position'][param] += agent['velocity'][param]
                agent['position'][param] = np.clip(agent['position'][param],
                                                 self.bounds[param][0],
                                                 self.bounds[param][1])
            else:
                agent['position'][param] = int(agent['position'][param] + agent['velocity'][param])
                agent['position'][param] = np.clip(agent['position'][param],
                                                 self.bounds[param][0],
                                                 self.bounds[param][1])
        return agent

    def optimize(self, X_train, y_train, X_val, y_val, timesteps, features):
        population = self.initialize_population()
        best_position = None
        best_fitness = float('inf')

        for iteration in range(self.n_iterations):
            # Calculate gravitational constant
            G = self.G0 * np.exp(-self.alpha * iteration / self.n_iterations)

            # Evaluate fitness for each agent
            fitness_scores = []
            for agent in population:
                model = create_base_lstm_model(
                    timesteps=timesteps,
                    features=features,
                    lstm_units=agent['position']['lstm_units'],
                    dense_units=agent['position']['dense_units'],
                    learning_rate=agent['position']['learning_rate']
                )

                _, fitness = train_evaluate_model(model, X_train, y_train, X_val, y_val)
                agent['fitness'] = fitness
                fitness_scores.append(fitness)

                if fitness < best_fitness:
                    best_fitness = fitness
                    best_position = deepcopy(agent['position'])

            # Calculate masses
            masses = self.calculate_masses(np.array(fitness_scores))

            # Update all agents
            for agent in population:
                agent = self.update_agent(agent, population, masses, G, iteration)

            print(f"Iteration {iteration + 1}/{self.n_iterations}, Best fitness: {best_fitness}")

        return best_position, best_fitness


class RDALSTMOptimizer:
    def __init__(self, population_size, n_iterations, bounds, radius_decay=0.95):
        self.population_size = population_size
        self.n_iterations = n_iterations
        self.bounds = bounds
        self.radius_decay = radius_decay

    def initialize_population(self):
        """Initialize random solutions"""
        population = []
        for _ in range(self.population_size):
            solution = {
                'lstm_units': np.random.randint(self.bounds['lstm_units'][0],
                                              self.bounds['lstm_units'][1]),
                'dense_units': np.random.randint(self.bounds['dense_units'][0],
                                               self.bounds['dense_units'][1]),
                'learning_rate': np.random.uniform(self.bounds['learning_rate'][0],
                                                 self.bounds['learning_rate'][1])
            }
            population.append(solution)
        return population

    def generate_neighbor(self, solution, radius):
        """Generate a neighbor solution within the given radius"""
        neighbor = {}
        for param in solution:
            if param == 'learning_rate':
                delta = np.random.uniform(-radius, radius) * (self.bounds[param][1] - self.bounds[param][0])
                neighbor[param] = np.clip(solution[param] + delta,
                                        self.bounds[param][0],
                                        self.bounds[param][1])
            else:
                delta = int(np.random.uniform(-radius, radius) *
                          (self.bounds[param][1] - self.bounds[param][0]))
                neighbor[param] = np.clip(solution[param] + delta,
                                        self.bounds[param][0],
                                        self.bounds[param][1])
        return neighbor

    def optimize(self, X_train, y_train, X_val, y_val, timesteps, features):
        population = self.initialize_population()
        best_solution = None
        best_fitness = float('inf')
        radius = 1.0

        for iteration in range(self.n_iterations):
            # Evaluate current population
            for solution in population:
                model = create_base_lstm_model(
                    timesteps=timesteps,
                    features=features,
                    lstm_units=solution['lstm_units'],
                    dense_units=solution['dense_units'],
                    learning_rate=solution['learning_rate']
                )

                _, fitness = train_evaluate_model(model, X_train, y_train, X_val, y_val)

                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = deepcopy(solution)

            # Generate new population
            new_population = [best_solution]  # Keep the best solution
            while len(new_population) < self.population_size:
                # Select random solution from current population
                base_solution = np.random.choice(population)
                # Generate neighbor
                neighbor = self.generate_neighbor(base_solution, radius)
                new_population.append(neighbor)

            population = new_population
            radius *= self.radius_decay  # Reduce search radius

            print(f"Iteration {iteration + 1}/{self.n_iterations}, Best fitness: {best_fitness}")

        return best_solution, best_fitness