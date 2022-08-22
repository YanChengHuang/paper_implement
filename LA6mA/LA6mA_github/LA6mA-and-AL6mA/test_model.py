from keras.models import load_model
from MyModel import AL6mA, LA6mA
model_path = "model/example_model.h5"
m = load_model(model_path)
AL = AL6mA()
LA= LA6mA()
print(m.summary())
print("-----")
print(AL.summary())
print("-----")
print(LA.summary())