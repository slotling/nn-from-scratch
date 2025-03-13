from alive_progress import alive_bar
import numpy as np
import emath as em
from datetime import datetime
import os

class LayerData:
    def __init__(self, neurons: int, prev_neurons: int, is_input_layer=False, activations=None, weights=None, biases=None, weighted_sum=None, random=False):
        rng = np.random.RandomState(1)

        self.activations: np.ndarray = np.full((neurons, ), 0.) if activations==None else activations
        
        if not is_input_layer:
            if random: 
                self.weights: np.ndarray = rng.randn(neurons, prev_neurons) if weights==None else weights
                self.biases: np.ndarray = rng.randn(neurons) if biases==None else biases
            else:
                self.weights: np.ndarray = np.full((neurons, prev_neurons), 0.) if weights==None else weights
                self.biases: np.ndarray = np.full((neurons, ), 0.) if biases==None else biases

            self.weighted_sum: np.ndarray = np.full((neurons, ), 0.) if weighted_sum==None else weighted_sum

            if self.weights.shape != (neurons, prev_neurons):
                raise Exception("Incorrect weights shape")
    
            if self.biases.shape != (neurons, ):
                raise Exception("Incorrect biases shape")
            
            if self.weighted_sum.shape != (neurons, ):
                raise Exception("Incorrect weighted sum shape")
        
        if self.activations.shape != (neurons, ):
            raise Exception("Incorrect activations shape")

class Layer:
    def __init__(self, neurons: int, prev_neurons: int | None, activation_function="sigmoid", values: LayerData | None = None):
        self.is_input_layer:bool = prev_neurons == None
        self.values: LayerData = LayerData(neurons, prev_neurons, self.is_input_layer, random=True) if values == None else values
        
        if not self.is_input_layer:
            self.changes: list[LayerData] = []
            self.debug_changes: list[LayerData] = []

        self.neurons = neurons
        self.prev_neurons = prev_neurons
        self.activation_function = activation_function

    def input_data(self, data: np.ndarray):
        if not self.is_input_layer:
            raise Exception("Can't manually input activations in hidden/output layer")

        self.values.activations = data

    def calculate(self, prev_activations: np.ndarray) -> None:
        if self.is_input_layer:
            raise Exception("Can't calculate activations in input layer")
        
        if prev_activations.shape == (1,):
            self.values.weighted_sum = (self.values.weights * prev_activations[0]).reshape(-1) + self.values.biases
        else:
            self.values.weighted_sum = np.dot(self.values.weights, prev_activations) + self.values.biases
        
        if self.activation_function == "sigmoid":
            self.values.activations = em.sigmoid(self.values.weighted_sum)
        elif self.activation_function == "softmax":
            self.values.activations = em.softmax(self.values.weighted_sum)

class LayerDeclaration():
    def __init__(self, neurons:int, activation_function:str="sigmoid", values:LayerData=None):
        self.neurons = neurons
        self.activation_function = activation_function
        self.values = values

class TrainDeclaration():
    def __init__(self, alpha, epoches):
        self.alpha = alpha
        self.epoches = epoches

class Model:
    def __init__(self, layers_info: list[LayerDeclaration], train_info: TrainDeclaration, cost_function="cross_entropy", premade_layers: list[Layer]=None, preload_folder: str = None):
        self.layers_info = layers_info
        self.train_info = train_info
        
        self.cost_function = cost_function

        if layers_info == None:
            self.layers = premade_layers
        else:
            self.layers: list[Layer] = []
            for i, declaration in enumerate(layers_info):
                if i == 0:
                    self.layers.append(Layer(declaration.neurons, None, None, declaration.values))
                    continue

                self.layers.append(Layer(declaration.neurons, layers_info[i-1].neurons, declaration.activation_function, declaration.values))

        if preload_folder != None:
            for i, LAYER in enumerate(self.layers):
                if i==0:
                    continue

                LAYER.values.weights = np.loadtxt(f"{preload_folder}/{i}_w.txt")
                LAYER.values.biases = np.loadtxt(f"{preload_folder}/{i}_b.txt")

        self.debug_cost_list = []
        self.debug_acc_list = []
    
    def calculate_and_predict(self, input: np.ndarray) -> np.ndarray:
        # FEEDFORWARD
        for i, LAYER in enumerate(self.layers):
            if i == 0:
                LAYER.input_data(input)
                continue

            prev_layer = self.layers[i-1]
            prev_activations = prev_layer.values.activations
            
            LAYER.calculate(prev_activations)

        return self.layers[-1].values.activations
    
    def train(self, inputs: np.ndarray, outputs: np.ndarray):
        with alive_bar(self.train_info.epoches * (inputs.shape[0] + (len(self.layers)-1))) as bar:
            for i in range(self.train_info.epoches):
                self.train_epoch(inputs, outputs, bar=bar, epoch=i)

                acc, cost = self.evaluate(inputs, outputs)
                self.debug_cost_list.append(cost)
                self.debug_acc_list.append(acc)

                bar.text(f"acc: {acc}, cost: {cost}, iter: {i}")
    
    def return_cost(self, prediction, output):
        if self.cost_function == "mse":
            return np.sum(np.square(prediction - output))
        elif self.cost_function == "cross_entropy":
            return -np.sum(output * np.log(prediction))

    def evaluate(self, inputs: np.ndarray, outputs: np.ndarray):
        avg_acc = 0
        avg_cost = 0

        for i, input in enumerate(inputs):
            output = outputs[i]
            prediction = self.calculate_and_predict(input)

            choice_index = prediction.argmax(axis=0)
            if output[choice_index] == 1:
                avg_acc += 1

            # cost calculation
            avg_cost += self.return_cost(prediction, output)

        avg_acc /= len(inputs)
        avg_cost /= len(inputs)

        return avg_acc, avg_cost
    
    def train_epoch(self, inputs: np.ndarray, outputs: np.ndarray, bar, epoch):
        for i, LAYER in enumerate(self.layers):
            if i == 0:
                continue
            LAYER.changes = []

        pass

        for i, input in enumerate(inputs):
            output = outputs[i]
            self.train_iter(input, output)

            bar()

        pass

        for i, LAYER in enumerate(self.layers):
            if i == 0:
                continue

            weights_changes = np.array([CHANGE.weights for CHANGE in LAYER.changes])
            biases_changes = np.array([CHANGE.biases for CHANGE in LAYER.changes])
            pass

            LAYER.values.weights -= np.average(weights_changes, axis=0) * self.train_info.alpha
            LAYER.values.biases -= np.average(biases_changes, axis=0) * self.train_info.alpha

            bar()
        
    
    def train_iter(self, input: np.ndarray, output: np.ndarray):
        prediction = self.calculate_and_predict(input)
        raw_costs = prediction - output

        start_layer_index = len(self.layers)-1

        # BACKPROPAGATION
        for layer_index in range(start_layer_index, -1, -1): # layer index counting backwards
            LAYER = self.layers[layer_index]
            
            if layer_index == 0:
                continue
            
            CHANGE = LayerData(LAYER.neurons, LAYER.prev_neurons, is_input_layer=False, random=False)
            
            # 1. activations change
            if layer_index == start_layer_index:
                if self.cost_function == "mse":
                    CHANGE.activations = 2*raw_costs
                else:
                    pass # ignore
            else:
                SUCCEEDING_LAYER = self.layers[layer_index+1]
                SUCCEEDING_CHANGE = SUCCEEDING_LAYER.changes[-1]
                CHANGE.activations = np.dot(SUCCEEDING_LAYER.values.weights.T, SUCCEEDING_CHANGE.weighted_sum)

            # 2. weighted sum changes
            if LAYER.activation_function == "softmax" and self.cost_function == "cross_entropy" and layer_index == start_layer_index:
                CHANGE.weighted_sum = prediction - output
            elif LAYER.activation_function == "sigmoid":
                CHANGE.weighted_sum = CHANGE.activations * em.sigmoid_derivative(LAYER.values.weighted_sum)

            # 3. weight changes
            PRECEDING_LAYER = self.layers[layer_index-1]
            CHANGE.weights = np.dot(np.array([CHANGE.weighted_sum]).T, np.array([PRECEDING_LAYER.values.activations]))

            # 4. bias changes
            CHANGE.biases = CHANGE.weighted_sum

            LAYER.changes.append(CHANGE)
            
            # # gradient checking
            # DEBUG_CHANGE = LayerData(LAYER.neurons, LAYER.prev_neurons, is_input_layer=False, random=False) 
            # for i, _ in enumerate(DEBUG_CHANGE.weights):
            #     for j, _ in enumerate(DEBUG_CHANGE.weights[i]):
            #         CLONE_MODEL = self.clone_model()
            #         cost1 = CLONE_MODEL.return_cost(CLONE_MODEL.calculate_and_predict(input), output)
            #         CLONE_MODEL.layers[layer_index].values.weights[i][j] += 0.000001 # small change
            #         cost2 = CLONE_MODEL.return_cost(CLONE_MODEL.calculate_and_predict(input), output)
            #         DEBUG_CHANGE.weights[i][j] = cost2 - cost1

            # for i, _ in enumerate(DEBUG_CHANGE.biases):
            #     CLONE_MODEL = self.clone_model()
            #     cost1 = CLONE_MODEL.return_cost(CLONE_MODEL.calculate_and_predict(input), output)
            #     CLONE_MODEL.layers[layer_index].values.biases[i] += 0.000001 # small change
            #     cost2 = CLONE_MODEL.return_cost(CLONE_MODEL.calculate_and_predict(input), output)
            #     DEBUG_CHANGE.biases[i] = cost2 - cost1
            
            # LAYER.debug_changes.append(DEBUG_CHANGE)

    def util_clone_model(self):
        return Model(premade_layers=self.layers)
    
    def util_write_params(self):
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d %H.%M.%S")

        if not os.path.exists(f"parameters/{dt_string}/"):
            os.makedirs(f"parameters/{dt_string}/")
        
        for i, LAYER in enumerate(self.layers):
            if i==0:
                continue
            
            pathw = f"parameters/{dt_string}/{i}_w.txt"
            pathb = f"parameters/{dt_string}/{i}_b.txt"
            open(pathw, 'a').close()
            open(pathb, 'a').close()

            np.savetxt(pathw, LAYER.values.weights)
            np.savetxt(pathb, LAYER.values.biases)

