import chromo as cr
import DNN_Entanglement as dent
from collections.abc import Callable
import numpy as np
import numpy.typing as npt
import random as rd
import matplotlib.pyplot as plt
import pennylane as qml
import os


# Test matrix used to ensure the GA worked before progressing to model evaluations
# target = np.array(
#     [[0., -300, -0.78401152, 500, -0.2533131],
#      [-0.05186097, 0., -0.44479764, -0.27236475, -0.0633621],
#      [1000, 1000, 0., 1000, -0.92172176],
#      [-0.39639624, -0.0873548, -0.12333476, 0., -0.98715809],
#      [-0.0808, -0.78747657, -0.20279267, -0.96499505, 0.]])


# Utility function that creates the pennylane CNOT gates as well as the model's standardised save name
def create_gates(entanglement_matrix: np.ndarray) -> (list[qml.CNOT], str):
    gates = []
    save_pairs = []
    for i in range(entanglement_matrix.shape[0]):
        for j in range(entanglement_matrix.shape[1]):
            if i == j:
                continue
            elif entanglement_matrix[i][j] == 1:
                gate = qml.CNOT(wires=[i, j])
                gates.append(gate)
                save_pairs.append(f"{i}{j}")
    save_name = "-".join(save_pairs)
    return gates, save_name


# Trains and evaluates a model with the given adjacency matrix, using the F1 score as the fitness score
def fit_func(chromo: cr.Chromo) -> float:
    # score = np.sum(np.multiply(target, chromo.matrix))
    # return float(score)
    dent.reset_keras()  # Fix memory leak issues with repeated keras model fitting and evaluating
    gates, save_name = create_gates(chromo.matrix)

    save_path = f"./Code/models_binary/DQiNN_{save_name}_{dent.N_QUBITS}-qubit_{dent.N_LAYERS}-layer_{dent.DATASET}.keras"

    if not os.path.isfile(save_path):
        my_q_model = dent.DQiNN(dent.TRAINING_PATH, dent.VALIDATION_PATH, dent.TESTING_PATH, save_path, layers=dent.N_LAYERS, dropout=dent.DROPOUT,
                                learning_rate=dent.LEARNING_RATE, batch_size=dent.BATCH_SIZE, epochs=dent.N_EPOCHS, threshold=dent.THRESHOLD,
                                n_qubits=dent.N_QUBITS, entanglements=gates)
        my_q_model.train_model()
        my_q_model.save()
        # my_q_model.evaluate_model(validation=True)
        my_q_model.evaluate_model(validation=False)
    else:
        my_q_model = dent.DQiNN(dent.TRAINING_PATH, dent.VALIDATION_PATH, dent.TESTING_PATH, save_path, loadpath=save_path,
                                layers=dent.N_LAYERS, dropout=dent.DROPOUT,
                                learning_rate=dent.LEARNING_RATE, batch_size=dent.BATCH_SIZE, epochs=dent.N_EPOCHS,
                                threshold=dent.THRESHOLD,
                                n_qubits=dent.N_QUBITS, entanglements=gates)
        my_q_model.evaluate_model(validation=False)

    score = my_q_model.f1_score
    del my_q_model

    return score


class GA(object):
    def __init__(self, dims: int, dom: cr.DatType, low_b: float, upp_b: float, pop_size_pairs: int, prob_xover: float, prob_mut: float, fitness: Callable[[cr.Chromo], float]) -> None:
        self.dims = dims  # In this case, the number of quantum wires in the circuits being tested
        self.dom = dom   # The numerical domain of the matrix entries (should always be cr.DatType.INTEGER)
        self.low_b = low_b  # The lower bound of acceptable matrix values (should always be 0)
        self.upp_b = upp_b  # The upper bound of acceptable matrix values (should always be 1)
        self.pop_size = pop_size_pairs * 2
        self.prob_xover = prob_xover   # Probability of performing crossover of two parents to create a child during iterations
        self.prob_mut = prob_mut  # Probability of mutating a child during iterations
        self.fitness = fitness
        self.population = []
        self.children = []
        self.history = []
        self.curr_scores = []
        self.children_scores = []
        self.cumul_scores = []
        self.minimum = []
        self.maximum = []
        self.average = []
        for i in range(self.pop_size):
            self.population.append(cr.Chromo(dims=self.dims, domain=self.dom, low_b=self.low_b, upp_b=self.upp_b))
            self.population[-1].fitness = self.fitness(self.population[-1])
            self.curr_scores.append(self.population[-1].fitness)
            curr_sum = sum(self.curr_scores)
            self.cumul_scores.append(curr_sum)
        self.history.append(self.curr_scores)
        self.minimum.append(min(self.curr_scores))
        self.maximum.append(max(self.curr_scores))
        self.average.append(sum(self.curr_scores)/self.pop_size)
        self.iter_index = 0

    # Roulette selector for use during iterations
    def roulette(self, unif: float) -> int:
        counter = 0
        total = sum(self.curr_scores)
        print(total)
        for i in range(self.pop_size - 1):
            if self.curr_scores[i] / total <= unif:
                break
            counter += 1

        return counter

    def gen_selector(self) -> dict[tuple[int, int], float]:
        tmp_dicto = {}
        for i in range(self.pop_size):
            tmp_dicto[(0, i)] = (self.curr_scores[i], self.population[i])
            tmp_dicto[(1, i)] = (self.children_scores[i], self.children[i])
        sorted_dicto = dict(sorted(tmp_dicto.items(), key=lambda x: x[1][0], reverse=True))
        return sorted_dicto

    def _iterate(self):
        self.children_scores = []
        # Create random numbers 0.0-1.0 used to determine if crossover will occur and if each individual child will be mutated
        tmp_prob_xover = rd.random()
        tmp_prob_mut1 = rd.random()
        tmp_prob_mut2 = rd.random()

        for k in range(int(self.pop_size / 2)):
            individual1, individual2 = None, None
            selector1, selector2 = self.roulette(rd.random()), self.roulette(rd.random())

            if tmp_prob_xover < self.prob_xover:
                rnd = rd.random()
                if rnd >= 0.5:
                    individual1, individual2 = self.population[selector1].x_over_cols(self.population[selector2], col=rd.randint(0, self.dims))
                else:
                    individual1, individual2 = self.population[selector1].x_over_rows(self.population[selector2], row=rd.randint(0, self.dims))
            else:
                individual1 = self.population[selector1].matrix
                individual2 = self.population[selector2].matrix
            self.children.append(cr.Chromo(dims=self.dims, domain=self.dom, low_b=self.low_b, upp_b=self.upp_b, init=False))
            self.children.append(cr.Chromo(dims=self.dims, domain=self.dom, low_b=self.low_b, upp_b=self.upp_b, init=False))
            self.children[-1].fitness = self.fitness(self.children[-1])
            self.children[-1].matrix = individual2
            self.children[-2].matrix = individual1
            self.children[-2].fitness = self.fitness(self.children[-2])
            self.children_scores.append(self.children[-2].fitness)
            self.children_scores.append(self.children[-1].fitness)
            if tmp_prob_mut1 < self.prob_mut:
                rnd = rd.choice([1, 2, 3])
                match rnd:
                    case 1:  # Mutation 1: Swap two full columns in a child
                        sample = rd.sample(list(range(0, self.dims)), 2)
                        self.children[-2]._swap_cols(sample[0], sample[1])
                        self.children[-2].fitness = self.fitness(self.children[-2])
                        self.children_scores[-2] = self.children[-2].fitness
                    case 2:  # Mutation 2: Swap two full rows in a child
                        sample = rd.sample(list(range(0, self.dims)), 2)
                        self.children[-2]._swap_rows(sample[0], sample[1])
                        self.children[-2].fitness = self.fitness(self.children[-2])
                        self.children_scores[-2] = self.children[-2].fitness
                    case 3:  # Mutation 3: Swap any two individual cells in a child
                        sample1 = rd.sample(list(range(0, self.dims)), 2)
                        sample2 = rd.sample(list(range(0, self.dims)), 2)
                        self.children[-2]._swap_elems(sample1[0], sample1[1], sample2[0], sample2[1])
                        self.children[-2].fitness = self.fitness(self.children[-2])
                        self.children_scores[-2] = self.children[-2].fitness
            if tmp_prob_mut2 < self.prob_mut:
                rnd = rd.choice([1, 2, 3])
                match rnd:
                    case 1:
                        sample = rd.sample(list(range(0, self.dims)), 2)
                        self.children[-1]._swap_cols(sample[0], sample[1])
                        self.children[-1].fitness = self.fitness(self.children[-1])
                        self.children_scores[-1] = self.children[-1].fitness
                    case 2:
                        sample = rd.sample(list(range(0, self.dims)), 2)
                        self.children[-1]._swap_rows(sample[0], sample[1])
                        self.children[-1].fitness = self.fitness(self.children[-1])
                        self.children_scores[-1] = self.children[-1].fitness
                    case 3:
                        sample1 = rd.sample(list(range(0, self.dims)), 2)
                        sample2 = rd.sample(list(range(0, self.dims)), 2)
                        self.children[-1]._swap_elems(sample1[0], sample1[1], sample2[0], sample2[1])
                        self.children[-1].fitness = self.fitness(self.children[-1])
                        self.children_scores[-1] = self.children[-1].fitness
        next_generation = list(self.gen_selector().values())
        self.population = []
        self.curr_scores = []
        self.cumul_scores = []
        curr_sum = 0
        counter = 0
        while counter < self.pop_size:  # Process the full population at each iteration
            self.population.append(next_generation[counter][1])
            self.curr_scores.append(self.population[-1].fitness)
            curr_sum = sum(self.curr_scores)
            self.cumul_scores.append(curr_sum)
            counter += 1
        self.history.append(self.curr_scores)
        self.minimum.append(min(self.curr_scores))
        self.maximum.append(max(self.curr_scores))
        self.average.append(sum(self.curr_scores)/self.pop_size)
        self.iter_index += 1
        self.cumul_scores = []
        self.children = []

    def run(self, max_iter: int, stagnancy: int) -> None:
        for i in range(max_iter):
            self._iterate()


if __name__ == '__main__':
    a = GA(10, cr.DatType.INTEGER, 0.0, 1.0, 5, 0.9, 0.1, fit_func)
    print(a.history, a.minimum, a.maximum, a.average)
    a.run(10, stagnancy=100)

    print(a.history)
    print(a.minimum)
    print(a.maximum)
    print(a.average)

    for chromo in a.population:
        chromo_string = chromo.get_string()
        print(f"({chromo_string})\n{chromo}\nFitness Score (F1): {chromo.fitness}\n\n")

    # Data for plotting
    t = np.arange(len(a.average))
    s = np.array(a.average)

    fig, ax = plt.subplots()
    ax.plot(t, s)

    ax.set(xlabel='Iterations', ylabel='Fitness Function',
           title='Average F1 Score Over Iterations (10-Qubit IoT23)')
    ax.grid()
    # plt.show()
    plt.savefig("Genalg.png")
