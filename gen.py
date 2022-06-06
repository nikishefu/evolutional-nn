import copy

import numpy as np


class Organism:
    def __init__(self, dimensions, scoring_function, output='softmax'):
        self.layers = []
        self.biases = []
        self.output = self._activation(output)
        self.scoring_function = scoring_function
        for i in range(len(dimensions)-1):
            shape = (dimensions[i], dimensions[i+1])
            std = np.sqrt(2 / sum(shape)) * 2
            layer = np.random.normal(0, std, shape)
            bias = np.random.normal(0, std, (1,  dimensions[i+1]))
            self.layers.append(layer)
            self.biases.append(bias)

        self.fitness = scoring_function(self)

    @staticmethod
    def _activation(output):
        if output == 'softmax':
            return lambda x: np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)
        if output == 'sigmoid':
            return lambda x: (1 / (1 + np.exp(-x)))
        if output == 'linear':
            return lambda x: x

    def predict(self, x):
        if not x.ndim == 2:
            raise ValueError(f'Input has {x.ndim} dimensions, expected 2')
        if not x.shape[1] == self.layers[0].shape[0]:
            raise ValueError(f'Input has {x.shape[1]} features, expected {self.layers[0].shape[0]}')
        for index, (layer, bias) in enumerate(zip(self.layers, self.biases)):
            x = x @ layer + np.ones((x.shape[0], 1)) @ bias
            if index == len(self.layers) - 1:
                x = self.output(x)  # output activation
            else:
                # x = np.clip(x, 0, np.inf)  # ReLU
                x = self._activation('sigmoid')(x)
        
        return x

    def mutate(self, stdev=0.3):
        for i in range(len(self.layers)):
            self.layers[i] += np.random.normal(0, stdev, self.layers[i].shape)
            self.biases[i] += np.random.normal(0, stdev, self.biases[i].shape)

    def mate(self, other, mutate=True):
        if not len(self.layers) == len(other.layers):
            raise ValueError('Both parents must have same number of layers')
        if not all(self.layers[x].shape == other.layers[x].shape for x in range(len(self.layers))):
            raise ValueError('Both parents must have same shape')

        child = copy.deepcopy(self)
        for i in range(len(child.layers)):
            pass_on = np.random.rand(1, child.layers[i].shape[1]) < 0.5
            child.layers[i] = pass_on * self.layers[i] + ~pass_on * other.layers[i]
            child.biases[i] = pass_on * self.biases[i] + ~pass_on * other.biases[i]
        if mutate:
            child.mutate()
        child.fitness = self.scoring_function(child)
        return child


class Ecosystem:
    def __init__(self, dimensions, scoring_function, output, population_size=100, holdout='sqrt', mating=True):
        """
        original_f must be a function to produce Organisms, used for the original population
        scoring_function must be a function which accepts an Organism as input and returns a float
        """
        self.population_size = population_size
        self.scoring_function = scoring_function
        self.mating = mating
        self.population = [Organism(dimensions, scoring_function, output) for _ in range(population_size)]
        if holdout == 'sqrt':
            self.holdout = max(1, int(np.sqrt(population_size)))
        elif holdout == 'log':
            self.holdout = max(1, int(np.log(population_size)))
        elif 0 < holdout < 1:
            self.holdout = max(1, int(holdout * population_size))
        else:
            self.holdout = max(1, int(holdout))

    def get_fitness(self, repeats):
        # return [np.mean([self.scoring_function(x) for _ in range(repeats)]) for x in self.population]
        return [x.fitness for x in self.population]

    def generation(self, repeats=1, keep_best=True):
        rewards = self.get_fitness(repeats)
        self.population = [self.population[x] for x in np.argsort(rewards)[::-1]]
        new_population = []
        for i in range(self.population_size):
            parent_1_idx = i % self.holdout
            if self.mating:
                parent_2_idx = min(self.population_size - 1, int(np.random.exponential(self.holdout)))
            else:
                parent_2_idx = parent_1_idx
            offspring = self.population[parent_1_idx].mate(self.population[parent_2_idx])
            new_population.append(offspring)
        if keep_best:
            new_population[-1] = self.population[0]  # Ensure best organism survives
        self.population = new_population

    def get_best_organism(self, repeats=1):
        return self.population[np.argsort(self.get_fitness(repeats))[-1]]

    def get_best_fitness(self, repeats=1):
        return max(self.get_fitness(repeats))

    def evolve(self, generations=1):
        """
        Makes specified number of generations.
        Returns list of fitness scores of the best organism on each generation
        """
        fitness_history = [self.get_best_fitness()]

        print(f"Evolving for {generations} generations")
        for i in range(generations):
            self.generation()
            fitness_history.append(self.get_best_fitness())
            print(f'\r{round((i + 1) / generations * 100, 2)}% Current best fitness: {fitness_history[-1]}', end="")
        print()

        return fitness_history

    def best_prediction(self, data):
        return self.get_best_organism().predict(data)
