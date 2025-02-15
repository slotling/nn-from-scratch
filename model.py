import numpy as np
import emath as em

class LayerData:
    def __init__(self, neurons: int, prev_neurons: int, is_input_layer: bool, random=False):
        self.activations: np.ndarray = np.full((neurons, ), 0)
        if not self.is_input_layer:
            if random: 
                self.weights: np.ndarray = np.random.randn(neurons, prev_neurons)
                self.biases: np.ndarray = np.random.randn(neurons)
            else:
                self.weights: np.ndarray = np.full((neurons, prev_neurons), 0)
                self.biases: np.ndarray = np.full((neurons, ), 0)

            self.weighted_sum: np.ndarray = np.full((neurons, ), 0)

class Layer:
    def __init__(self, neurons: int, prev_neurons: int):
        self.is_input_layer:bool = prev_neurons == None
        self.values: LayerData = LayerData(neurons, prev_neurons, self.is_input_layer, random=True)
        self.changes: list[LayerData] = []

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
    
    def predict(self, input: np.ndarray) -> np.ndarray:
        self.layers[0].input_data(input)
        for i, layer in enumerate(self.layers):
            if i == 0:
                continue

            prev_layer = self.layers[i-1]
            prev_activations = prev_layer.activations

            layer.calculate(prev_activations)
        
        return self.layers[-1].activations
    
    def train_epoch(self, inputs: np.ndarray, outputs: np.ndarray):
        for i, input in enumerate(inputs):
            output = outputs[i]

            prediction = self.predict(input)
            raw_costs = prediction - output

            
