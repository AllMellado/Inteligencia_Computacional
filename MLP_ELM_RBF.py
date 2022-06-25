import matplotlib.pyplot as plt
from numpy import zeros, exp, array, random, dot, append, round, insert, linalg, transpose, ones, column_stack, tanh, \
    argmax


class MLP():
    def __init__(self, number_of_neurons, number_of_inputs):
        self.weights_layer1 = 2 * random.random_sample((number_of_inputs, number_of_neurons)) - 1
        self.weights_layer2 = 2 * random.random_sample((number_of_neurons, 1)) - 1

    def train(self, training_inputs, training_outputs, validation_inputs, validation_outputs, number_of_iterations):
        self.mse_array_training = [-1 for i in range(number_of_iterations)]
        self.mse_array_validation = [-1 for i in range(number_of_iterations)]
        self.best_mse = 500

        for iteration in range(number_of_iterations):

            layer1_output = 1 / (1 + exp(-1 * dot(training_inputs, self.weights_layer1)))
            layer2_output = 1 / (1 + exp(-1 * dot(layer1_output, self.weights_layer2)))

            layer2_error = training_outputs - layer2_output
            layer2_delta = layer2_error * (layer2_output * (1.0 - layer2_output))

            layer1_error = layer2_delta.dot(self.weights_layer2.T)
            layer1_delta = layer1_error * (layer1_output * (1.0 - layer1_output))

            #       VALIDATION STUFF    #
            layer1_output_validation = 1 / (1 + exp(-1 * dot(validation_inputs, self.weights_layer1)))
            layer2_output_validation = 1 / (1 + exp(-1 * dot(layer1_output_validation, self.weights_layer2)))

            layer2_error_validation = validation_outputs - layer2_output_validation

            self.mse_array_training[iteration] = (layer2_error ** 2).sum() / len(training_inputs)
            self.mse_array_validation[iteration] = (layer2_error_validation ** 2).sum() / len(validation_inputs)

            if self.mse_array_validation[iteration] < self.best_mse:
                self.best_mse = self.mse_array_validation[iteration]
                self.best_iteration = iteration
                self.layer1_best_weights = self.weights_layer1.copy()
                self.layer2_best_weights = self.weights_layer2.copy()
            #                           #

            self.weights_layer1 += 0.01 * training_inputs.T.dot(layer1_delta)
            self.weights_layer2 += 0.01 * layer1_output.T.dot(layer2_delta)

    def predict(self, inputs):
        layer1_output = 1 / (1 + exp(-1 * dot(inputs, self.weights_layer1)))
        return 1 / (1 + exp(-1 * dot(layer1_output, self.weights_layer2)))


class ELM():
    def __init__(self, n_hidden_units):
        self.n_hidden_units = n_hidden_units

    def train(self, inputs, outputs):
        self.W_in = random.randn(inputs.shape[1], self.n_hidden_units)
        G = tanh(inputs.dot(self.W_in))
        self.W_out = linalg.pinv(G).dot(outputs)

    def predict(self, inputs):
        G = tanh(inputs.dot(self.W_in))
        return G.dot(self.W_out)


class RBF():
    def __init__(self, hidden_shape, sigma=0.0):
        self.hidden_shape = hidden_shape
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def kernel_function(self, center, data_point):
        return exp(-self.sigma * linalg.norm(center - data_point) ** 2)

    def calculate_interpolation_matrix(self, X):
        G = zeros((len(X), self.hidden_shape))
        for data_point_arg, data_point in enumerate(X):
            for center_arg, center in enumerate(self.centers):
                G[data_point_arg, center_arg] = self.kernel_function(center, data_point)
        return G

    def select_centers(self, X):
        random_args = random.choice(len(X), self.hidden_shape)
        centers = X[random_args]
        return centers

    def train(self, inputs, outputs):
        self.centers = self.select_centers(inputs)
        G = self.calculate_interpolation_matrix(inputs)
        self.weights = dot(linalg.pinv(G), outputs)

    def predict(self, inputs):
        G = self.calculate_interpolation_matrix(inputs)
        return dot(G, self.weights)


class dataOrganizer():
    def __init__(self):
        self.HVPV = []
        self.LVPV = []
        self.matrixData = []

    # Parsing data file 
    def getMatrixData(self, filePath):
        with open(filePath, "r") as data:
            self.matrixData = []
            for nr, line in enumerate(data):
                # print(nr)
                parse = line.split(",")
                parse = [float(i) for i in parse]
                parse.insert(1, 1)
                self.matrixData.append(parse)

        # Saving highest and lowest value 
        self.HVPV = [-1 for i in self.matrixData[0]]
        self.LVPV = [-1 for i in self.matrixData[0]]
        for data in self.matrixData:
            for i in range(len(data)):
                if (data[i] > self.HVPV[i]):
                    self.HVPV[i] = data[i]
                if (data[i] < self.LVPV[i]):
                    self.LVPV[i] = data[i]

        self.originalMatrixData = self.matrixData.copy()

    def generateSets(self):
        random.shuffle(self.matrixData)
        size = len(self.matrixData)
        self.training_set_inputs = array([x[1:size] for x in self.matrixData[0:int((size * 60) / 100)]])
        self.training_set_outputs = array([[x[0] for x in self.matrixData[0:int((size * 60) / 100)]]]).T

        self.validation_set_inputs = array([x[1:size] for x in self.matrixData[int((size * 60) / 100):int((size * 80) / 100)]])
        self.validation_set_outputs = array([[x[0] for x in self.matrixData[int((size * 60) / 100):int((size * 80) / 100)]]]).T

        self.testing_set_inputs = array([x[1:size] for x in self.matrixData[int((size * 80) / 100):size]])
        self.testing_set_outputs = array([[x[0] for x in self.matrixData[int((size * 80) / 100):size]]]).T

    def normalizeData(self):
        for i in range(len(self.matrixData)):
            for j in range(len(self.matrixData[0])):
                self.matrixData[i][j] = -1 + 2 * (
                            (self.matrixData[i][j] - self.LVPV[j]) / (self.HVPV[j] - self.LVPV[j]))


if __name__ == "__main__":

    filenames = ["iris", "wine", "letter", "poker"]
    for name in filenames:
        fileName = 'M:\Downloads\\' + name + '.data'
        data = dataOrganizer()
        data.getMatrixData(fileName)
        data.normalizeData()
        random.seed()

        options = [1]
        for op in options:
            if (op == 1):
                NN = "MLP"
            if (op == 2):
                NN = "ELM"
            if (op == 3):
                NN = "RBF"

            numNeurons = [15, 20, 35, 50]
            for nrNeurons in numNeurons:
                it = 1000

                # Initialization
                bestMSE = 500
                best_validation = []
                best_training = []
                box_data = []

                mse = [0 for i in range(10)]
                best_acc = 0
                filename = "Accuracy_" + NN + "_" + name + '_' + str(nrNeurons) + '.txt'
                file = open(filename, 'w')
                print("Executing: " + name + " - " + NN + " - " + str(nrNeurons))
                for i in range(10):
                    data.generateSets()
                    predicted_outputs = []

                    if (op == 1):
                        mlp = MLP(nrNeurons, data.training_set_inputs.shape[1])

                        mlp.train(data.training_set_inputs, data.training_set_outputs,
                                  data.validation_set_inputs, data.validation_set_outputs, it)

                        predicted_outputs = mlp.predict(data.testing_set_inputs)

                    if (op == 2):
                        elm = ELM(nrNeurons)
                        elm.train(data.training_set_inputs, data.training_set_outputs)

                        predicted_outputs = elm.predict(data.testing_set_inputs)

                    if (op == 3):
                        rbf = RBF(nrNeurons, 1)
                        rbf.train(data.training_set_inputs, data.training_set_outputs)

                        predicted_outputs = rbf.predict(data.testing_set_inputs)

                    error = data.testing_set_outputs - predicted_outputs

                    mse[i] = (error ** 2).sum() / len(data.testing_set_inputs)

                    correct = 0
                    total = predicted_outputs.shape[0]
                    for i in range(total):
                        x = abs(data.testing_set_outputs[i] - predicted_outputs[i])
                        correct += (1 if x <= 0.2 else 0)

                    string = 'Accuracy: ' + str(correct / total)
                    print(string)
                    file.write("%s\n" % string)

                    if (op == 1 and correct / total > best_acc):
                        best_acc = correct / total
                        best_training_mse = mlp.mse_array_training
                        best_validation_mse = mlp.mse_array_validation
                        best_iteration = mlp.best_iteration
                file.close()
                filename = "Boxplot_" + NN + "_" + name + '_' + str(nrNeurons) + '.png'
                fig = plt.figure(nrNeurons)
                plt.title("Boxplot " + NN + ": " + name + ' - ' + str(nrNeurons) + ' neurônios')
                plt.ylabel('MSE')
                plt.boxplot(mse)
                fig.savefig(filename)
                plt.close(fig)

                if (op == 1):
                    iterations_array = list(range(it))
                    filename = "Gráfico_" + NN + "_" + name + '_' + str(nrNeurons) + '.png'
                    fig = plt.figure(nrNeurons + 1)
                    plt.plot(iterations_array, best_training_mse, label='Training')
                    plt.plot(iterations_array, best_validation_mse, 'r--', label='Validation')
                    plt.plot(best_iteration, best_validation_mse[best_iteration], 'go', label='Best weights')
                    plt.title("Gráfico " + NN + ": " + name + ' - ' + str(nrNeurons) + ' neurônios')
                    plt.ylabel('MSE')
                    plt.xlabel('Iteration')
                    plt.legend(loc='best')
                    # fig.savefig(filename)
                    plt.close(fig)
