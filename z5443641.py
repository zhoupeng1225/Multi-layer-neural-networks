#Done By
#Zhou Peng z5443641

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras as keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


ca = np.loadtxt("compoundA.txt")
s = np.loadtxt("substrate.txt")
b = np.loadtxt("biomass.txt")

cap = np.loadtxt("gen_compoundA_pre.txt")
sp = np.loadtxt("gen_substrate_pre.txt")
bp = np.loadtxt("gen_biomass_pre.txt")

# Identify variation range for input and output variables.
ca_min, ca_max = np.min(ca), np.max(ca)
s_min, s_max = np.min(s), np.max(s)
b_min, b_max = np.min(b), np.max(b)
#Plot each variable to observe the overall behaviour of the bioprocess
plt.figure(figsize=(12, 6))
plt.subplot(311)
plt.plot(ca, label="Compound A")
plt.ylabel("Compound A")
plt.legend()


plt.subplot(312)
plt.plot(s, label="Substrate")
plt.ylabel("Substrate")
plt.legend()

plt.subplot(313)
plt.plot(b, label="Biomass")
plt.ylabel("Biomass")
plt.legend()

plt.tight_layout()
plt.show()

#In case outliers are detected correct the data accordingly. For instance,
#as we are dealing with variables measured in grams, no value should
#be less than zero. A simple correction is to replace such values with a
#zero value

ca = np.where(ca < 0, 0, ca)
s = np.where(s < 0, 0, s)
b = np.where(b < 0, 0, b)



inputs = np.column_stack((ca, s)).reshape(-1,2)
outputs = b
idx = np.array(range(len(inputs)))
train_idx = np.random.choice(idx, len(inputs) - 400, replace = False)
#test_idx = np.array([i for i in idx if i not in train_idx])


# print(inputs.shape)


train_inputs = inputs[train_idx]
train_outputs = outputs[train_idx]
test_inputs = np.column_stack((cap, sp)).reshape(-1,2)
test_outputs = bp

#print(train_inputs.shape)
#print(test_inputs.shape)

#Design the multi-layer neural network. 
#Given the number of training samples, 
#propose a neural architecture taking into account 
#Ni – number of inputs, 
#Nh – number of units in the hidden layer, 
#and No – number of outputs.



Ni = 2
Nh = 8
No = 1

Nw = (Ni * Nh) + Nh + (Nh * No) + No
#print(Nw, len(b))
assert Nw < len(b) / 10, "The number of parameters is too high."




#Decide the number of layers and their respective activation functions.
#input layer -- 2  Hidden layer -- 8 output layer -- 1
model = keras.Sequential([
    layers.Dense(Nh, activation='relu', input_shape=(Ni,)),
    layers.Dense(Nh, activation='relu'),
    layers.Dense(No, activation='linear')
])
op = Adam(learning_rate=0.005)
#optimizer =SGD(learning_rate=0.01)
model.compile(loss='mean_squared_error', optimizer=op)
model.summary()

#Remember it’s recommended your network accomplish the maximal
#number of parameters Nw < (number of samples)/10.

# Nw < (len(train_ca) + len(train_s)) / 10
#the formula for calculating the number of parameters in a multi-layer neural network
# nW = (Ni * Nh) + Nh + (Nh * No) + No



# np / tensorflow seed

# Create early stopping (once our model stops improving, stop training)
ears = EarlyStopping(monitor='val_loss', patience=20)

# We are going to keep the weights of our best model during training
bestMW = ModelCheckpoint('best_weights.h5', save_best_only=True, monitor='val_loss', mode='auto')

# We include these callbacks into a 'callbacks' list
CB = [ears, bestMW]

#model.save()
#model.load()
# Then you can include this list in the 'fit' method

tr=model.fit(train_inputs, train_outputs, validation_split=0.25, epochs=500, batch_size=64, callbacks=CB)
predictions = model.predict(test_inputs)
plt.figure(figsize=(10, 6))
plt.plot(tr.history['loss'], label='Training Loss')
plt.plot(tr.history['val_loss'], label='Validation Loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(test_outputs, 'r*-', label='Real Biomass')
plt.plot(predictions, 'k+-', label='Estimated Biomass')
plt.title('Network output with training')
plt.ylabel('Biomass')
plt.xlabel('Index')
plt.legend(loc='upper left')
plt.show()


#Compute error indexes to complement the visual analysis. Use IA, RMS, and RSD.
'''
oi --observed values.
pi --predicted values.
N --number of data samples.
om --observations’ mean value.
oii = oi − om
pii = pi − om

'''
# observed values (b) and predicted values (predictions)

# adjusted observed and predicted values
# adjusted_observed_values = observed_values - observed_mean
# adjusted_predicted_values = predicted_values - observed_mean


oi = test_outputs
pi = predictions.reshape(-1)
N = len(oi)
om = np.mean(oi)

adjusted_observed_values = oi - om
adjusted_predicted_values = pi - om



# Index of Agreement (IA)
IA = 1 - (np.sum(np.square(oi - pi)) / np.sum(np.square(np.abs(adjusted_predicted_values) + np.abs(adjusted_observed_values))))
print(f"Index of Agreement (IA): {IA}")

# Root Mean Square Error (RMSE)
RMSE = np.sqrt(np.sum(np.square(oi - pi)) / np.sum(np.square(oi)))
print(f"Root Mean Square Error (RMSE): {RMSE}")

# Relative Standard Deviation (RSD)
RSD = np.sqrt(np.sum(np.square(oi - pi)) / N)
print(f"Relative Standard Deviation (RSD): {RSD}")
    


