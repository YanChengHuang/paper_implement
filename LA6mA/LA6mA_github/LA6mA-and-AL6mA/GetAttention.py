import numpy as np
import matplotlib.pyplot as plt
from DataProcessing import load_data
from keras.layers.core import K
from keras.models import load_model
from MyModel import AL6mA, LA6mA

def get_attentions(model, inputs, layer_name):

    attention = []
    inp = model.input
    outputs = [layer.output for layer in model.layers if layer.name == layer_name]
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        attention.append(layer_activations)
    return attention


def _main():
    filename = 'data/toydata.xlsx'
    model_path = "model/example_model.h5"
    #  load the random initialized LA6mA, or load a well_trained model for analysis
    # m = LA6mA()
    m = load_model(model_path)
    [x_test, _] = load_data(filename)
    attention_vectors = []
    for i in range(20):
        single_inputs = np.array([x_test[i, :, :]])
        attention_vector = np.mean(get_attentions(m, single_inputs, 'attention_vec')[0], axis=2).squeeze()
        assert (np.sum(attention_vector) - 1.0) < 1e-5
        attention_vectors.append(attention_vector)

    attention_vectors_mean = np.mean(np.array(attention_vectors), axis=0)
    X = np.arange(41) - 20
    plt.bar(X, attention_vectors_mean, alpha=0.9, width=0.9, facecolor='#5091E1', edgecolor='white', label='one', lw=1)
    plt.show()

if __name__ == '__main__':
    _main()


