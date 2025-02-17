from alive_progress import alive_bar
import numpy as np
import emath as em

class LayerData:
    def __init__(self, neurons: int, prev_neurons: int, is_input_layer=False, random=False):
        self.activations: np.ndarray = np.full((neurons, ), 0.)
        if not is_input_layer:
            if random: 
                self.weights: np.ndarray = np.random.randn(neurons, prev_neurons)
                self.biases: np.ndarray = np.random.randn(neurons)
            else:
                self.weights: np.ndarray = np.full((neurons, prev_neurons), 0.)
                self.biases: np.ndarray = np.full((neurons, ), 0.)

            self.weighted_sum: np.ndarray = np.full((neurons, ), 0.)

class Layer:
    def __init__(self, neurons: int, prev_neurons: int | None):
        self.is_input_layer:bool = prev_neurons == None
        self.values: LayerData = LayerData(neurons, prev_neurons, self.is_input_layer, random=True)
        
        if not self.is_input_layer:
            self.changes: list[LayerData] = []

        self.neurons = neurons
        self.prev_neurons = prev_neurons

    def input_data(self, data: np.ndarray):
        if not self.is_input_layer:
            raise Exception("Can't manually input activations in hidden/output layer")

        self.values.activations = data

    def calculate(self, prev_activations: np.ndarray) -> None:
        if self.is_input_layer:
            raise Exception("Can't calculate activations in input layer")
        
        if prev_activations.shape == (1,):
            self.values.weighted_sum = (self.values.weights * prev_activations[0]).reshape(-1)
        else:
            self.values.weighted_sum = np.dot(self.values.weights, prev_activations)
        self.values.activations = em.sigmoid(self.values.weighted_sum)

class Model:
    def __init__(self, neurons_list: list[int]):
        self.layers: list[Layer] = []
        for i, count in enumerate(neurons_list):
            layer: Layer = None
            if i == 0:
                layer = Layer(count, None)
            else:
                layer = Layer(count, neurons_list[i-1])
            
            self.layers.append(layer)

        self.debug_cost_list = []
    
    def predict(self, input: np.ndarray) -> np.ndarray:
        self.layers[0].input_data(input)
        for i, LAYER in enumerate(self.layers):
            if i == 0:
                continue

            prev_layer = self.layers[i-1]
            prev_activations = prev_layer.values.activations

            if i==1:
                pass
            
            LAYER.calculate(prev_activations)

        return self.layers[-1].values.activations
    
    def train(self, inputs: np.ndarray, outputs: np.ndarray, epoches=2000):
        with alive_bar(epoches * (inputs.shape[0] + len(self.layers)-1)) as bar:
            for i in range(epoches):
                self.train_epoch(inputs, outputs, bar, i)
        
    def evaluate(self, inputs: np.ndarray, outputs: np.ndarray):
        acc = 0
        cost = 0

        for i, input in enumerate(inputs):
            output = outputs[i]
            prediction = self.predict(input)

            choice_index = prediction.argmax(axis=0)
            if output[choice_index] == 1:
                acc += 1
            
            cost += np.sum(np.square(prediction - output))
        
        acc /= len(inputs)
        cost /= len(inputs)

        return acc, cost
    
    def train_epoch(self, inputs: np.ndarray, outputs: np.ndarray, bar, epoch):
        for i, LAYER in enumerate(self.layers):
            if i == 0:
                continue
            LAYER.changes = []

        for i, input in enumerate(inputs):
            output = outputs[i]
            self.train_iter(input, output)

            bar()
        
        alpha = 0.01
        for i, LAYER in enumerate(self.layers):
            if i == 0:
                continue
            
            weights_changes = np.array([])
            biases_changes = np.array([])
            for CHANGE in LAYER.changes:
                weights_changes = np.append(weights_changes, CHANGE.weights)
                biases_changes = np.append(biases_changes, CHANGE.biases)

            LAYER.values.weights -= np.average(weights_changes, axis=0) * alpha
            LAYER.values.biases -= np.average(weights_changes, axis=0) * alpha

            bar()
        
        acc, cost = self.evaluate(inputs, outputs)
        self.debug_cost_list.append(cost)

        bar.text(f"acc: {acc}, cost: {cost}, iter: {epoch}")
        
    
    def train_iter(self, input: np.ndarray, output: np.ndarray):
        prediction = self.predict(input)
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

            # 2. weighted sum changes (LOOK AGAIN)
            CHANGE.weighted_sum = np.multiply(CHANGE.activations, em.sigmoid_derivative(LAYER.values.weighted_sum))
            
            # 3. weight changes
            PRECEDING_LAYER = self.layers[layer_index-1]
            for current_index in range(LAYER.neurons):
                for preceding_index in range(PRECEDING_LAYER.neurons):
                    CHANGE.weights[current_index][preceding_index] = CHANGE.weighted_sum[current_index] * PRECEDING_LAYER.values.activations[preceding_index]

            # 4. bias changes
            CHANGE.biases = CHANGE.weighted_sum * 1

            LAYER.changes.append(CHANGE)
