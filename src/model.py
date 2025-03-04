from alive_progress import alive_bar
import numpy as np
import emath as em

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
    def __init__(self, neurons: int, prev_neurons: int | None, activation="sigmoid", values: LayerData | None = None):
        self.is_input_layer:bool = prev_neurons == None
        self.values: LayerData = LayerData(neurons, prev_neurons, self.is_input_layer, random=True) if values == None else values
        
        if not self.is_input_layer:
            self.changes: list[LayerData] = []

        self.neurons = neurons
        self.prev_neurons = prev_neurons
        self.activation_function = activation

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

class Model:
    def __init__(self, neurons_list: list[int], values_list: None | list[LayerData] = None):
        self.layers: list[Layer] = []
        for i, count in enumerate(neurons_list):
            layer: Layer = None
            if i == 0:
                layer = Layer(count, None, values=None if values_list == None else values_list[i])
            else:
                layer = Layer(count, neurons_list[i-1], values=None if values_list == None else values_list[i])
            
            self.layers.append(layer)

        self.debug_cost_list = []
    
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
    
    def train(self, inputs: np.ndarray, outputs: np.ndarray, alpha=0.01, epoches=200):
        with alive_bar(epoches * (inputs.shape[0] + len(self.layers)-1)) as bar:
            for i in range(epoches):
                self.train_epoch(inputs, outputs, alpha=alpha, bar=bar, epoch=i)
        
    def evaluate(self, inputs: np.ndarray, outputs: np.ndarray):
        acc = 0
        cost = 0

        for i, input in enumerate(inputs):
            output = outputs[i]
            prediction = self.calculate_and_predict(input)

            choice_index = prediction.argmax(axis=0)
            if output[choice_index] == 1:
                acc += 1
            
            cost += np.sum(np.square(prediction - output))
        
        acc /= len(inputs)
        cost /= len(inputs)

        return acc, cost
    
    def train_epoch(self, inputs: np.ndarray, outputs: np.ndarray, alpha:np.float64, bar, epoch):
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

            LAYER.values.weights -= np.average(weights_changes, axis=0) * alpha
            LAYER.values.biases -= np.average(biases_changes, axis=0) * alpha

            bar()
        
        acc, cost = self.evaluate(inputs, outputs)
        self.debug_cost_list.append(cost)

        bar.text(f"acc: {acc}, cost: {cost}, iter: {epoch}")
        
    
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
                CHANGE.activations = 2*raw_costs
            else:
                SUCCEEDING_LAYER = self.layers[layer_index+1]
                SUCCEEDING_CHANGE = SUCCEEDING_LAYER.changes[-1]
                for current_index in range(LAYER.neurons):
                    for succeeding_index in range(SUCCEEDING_LAYER.neurons):
                        CHANGE.activations[current_index] += SUCCEEDING_CHANGE.weighted_sum[succeeding_index] * SUCCEEDING_LAYER.values.weights[succeeding_index][current_index]

            # 2. weighted sum changes
            if LAYER.activation_function == "sigmoid":
                CHANGE.weighted_sum = np.multiply(CHANGE.activations, em.sigmoid_derivative(LAYER.values.weighted_sum))

            # 3. weight changes
            PRECEDING_LAYER = self.layers[layer_index-1]
            for current_index in range(LAYER.neurons):
                for preceding_index in range(PRECEDING_LAYER.neurons):
                    CHANGE.weights[current_index][preceding_index] = CHANGE.weighted_sum[current_index] * PRECEDING_LAYER.values.activations[preceding_index]

            # 4. bias changes
            CHANGE.biases = CHANGE.weighted_sum

            LAYER.changes.append(CHANGE)
            pass