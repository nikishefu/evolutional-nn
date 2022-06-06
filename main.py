import matplotlib.pyplot as plt
import numpy as np

from time import time
from gen import Ecosystem
from mnist import MNIST


def load_mnist(ds_path):
    """Loads MNIST dataset of training and testing data"""

    print("Loading MNIST dataset ... ", end='')
    data, labels = MNIST(ds_path).load_training()
    data = np.matrix([[i / 255 for i in image]
                      for image in data])

    targets = np.zeros((data.shape[0], 10))
    for i in range(targets.shape[0]):
        targets[i, np.matrix(labels)[0, i]] = 1
    print("Done")

    return data, targets


def sine_problem(organism):
    """
    Randomly generate `replicates` samples in [0,1],
    use the organism to predict their corresponding value,
    and return the fitness score of the organism
    """
    in_data = np.random.random((100, 1))
    return np.mean(np.square(np.sin(2 * np.pi * in_data) - organism.predict(in_data)))


def xor_problem(organism):
    predictions = organism.predict(np.matrix([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]))
    return -np.mean(np.square(np.matrix([0, 1, 1, 0]).T - predictions))


input, output = load_mnist('mnist')


def digit_classification(organism):
    random_indices = np.random.choice(input.shape[0], size=100, replace=False)
    predictions = organism.predict(input[random_indices, :])
    return -np.mean(np.square(predictions - output[random_indices, :]))


def digit_classification2(organism):
    random_indices = np.random.choice(input.shape[0], size=200, replace=False)
    correct = 0
    for prediction, target in zip(organism.predict(input[random_indices, :]), output[random_indices, :]):
        if np.argmax(prediction) == np.argmax(target):
            correct += 1

    return correct


# Plot setup
plt.style.use('_mpl-gallery')
fig, ax = plt.subplots()


# ecosystem = Ecosystem([1, 16, 16, 16, 1], sine_problem, output='linear',
#                       population_size=100, holdout=0.1, mating=True)
# generations = 100

# ecosystem = Ecosystem([2, 4, 1], xor_problem, output='linear',
#                       population_size=20, holdout=0.1, mating=True)
# generations = 40

ecosystem = Ecosystem([input.shape[1], 64, 10], digit_classification2, output='sigmoid',
                      population_size=200, holdout=0.1, mating=True)
generations = 1000

start_time = time()
history = ecosystem.evolve(generations)
print(f'Evolution took {time() - start_time}s')

ax.plot(list(range(generations + 1)), history, linewidth=2.0)

plt.show()

# # Check for xor problem
# print(ecosystem.best_prediction(np.matrix([
#     [0, 0],
#     [0, 1],
#     [1, 0],
#     [1, 1]
# ])))

# # Check for sin problem
# print(ecosystem.best_prediction(np.matrix([
#     [0],
#     [0.5],
#     [0.75],
#     [0.25]
# ])))

# Check for digit classification
correct = 0
for prediction, target in zip(ecosystem.best_prediction(input), output):
    if np.argmax(prediction) == np.argmax(target):
        correct += 1
print('Accuracy: ', end='')
print(round(correct / input.shape[0] * 100, 2), end='%\n')
