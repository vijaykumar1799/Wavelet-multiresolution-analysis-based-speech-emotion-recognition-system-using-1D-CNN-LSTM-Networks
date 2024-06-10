#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns

from random import seed, random, randint, sample

from keras.models import Model, save_model, load_model
from keras.layers import Input, Dense, BatchNormalization, TimeDistributed, ELU
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras.utils import to_categorical

import pickle as pkl

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def r_squared(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res/(SS_tot + K.epsilon())


# In[3]:


# Input layer
input_layer = Input(shape=(4000, 125))

# Encoder
encoder_layer_1 = TimeDistributed(Dense(64))(input_layer)
encoder_layer_1 = TimeDistributed(BatchNormalization())(encoder_layer_1)
encoder_layer_1 = TimeDistributed(ELU())(encoder_layer_1)

encoder_layer_2 = TimeDistributed(Dense(16))(encoder_layer_1)
encoder_layer_2 = TimeDistributed(BatchNormalization())(encoder_layer_2)
encoder_layer_2 = TimeDistributed(ELU())(encoder_layer_2)

encoder_layer_3 = TimeDistributed(Dense(16))(encoder_layer_2)
encoder_layer_3 = TimeDistributed(BatchNormalization())(encoder_layer_3)
encoder_layer_3 = TimeDistributed(ELU())(encoder_layer_3)

# Latent Space
latent_space = TimeDistributed(Dense(8))(encoder_layer_3)
latent_space = TimeDistributed(BatchNormalization())(latent_space)
latent_space = TimeDistributed(ELU(), name="latent_space_out")(latent_space)

# Decoder
decoder_layer_1 = TimeDistributed(Dense(16))(latent_space)
decoder_layer_1 = TimeDistributed(BatchNormalization())(decoder_layer_1)
decoder_layer_1 = TimeDistributed(ELU())(decoder_layer_1)

decoder_layer_2 = TimeDistributed(Dense(32))(decoder_layer_1)
decoder_layer_2 = TimeDistributed(BatchNormalization())(decoder_layer_2)
decoder_layer_2 = TimeDistributed(ELU())(decoder_layer_2)

decoder_layer_3 = TimeDistributed(Dense(64))(decoder_layer_2)
decoder_layer_3 = TimeDistributed(BatchNormalization())(decoder_layer_3)
decoder_layer_3 = TimeDistributed(ELU())(decoder_layer_3)

# Output layer
output_layer = TimeDistributed(Dense(125))(decoder_layer_3)

# Autoencoder Model
autoencoder = Model(inputs=input_layer, outputs=output_layer)

# Display the model summary
autoencoder.summary()


# In[4]:


autoencoder.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error', metrics=[r_squared])


# In[5]:


training_data = np.load('training_features.npz')
validation_data = np.load('validation_features.npz')
testing_data = np.load('testing_features.npz')


# In[6]:


x_train, y_train = training_data['a'], training_data['b']
x_val, y_val = validation_data['a'], validation_data['b']
x_test, y_test = testing_data['a'], testing_data['b']


# In[7]:


x_test = np.concatenate((x_val, x_test), axis=0)
y_test = np.concatenate((y_val, y_test), axis=0)


# In[8]:


print(f"Training-> {x_train.shape},  {y_train.shape}")
#print(f"Validation-> {x_val.shape},  {y_val.shape}")
print(f"Testing-> {x_test.shape},  {y_test.shape}")


# In[9]:


# label shape formatting 
labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
num_classes = len(labels)
y_train = to_categorical(y_train, num_classes)
#y_val = to_categorical(y_val, num_classes)
y_test = to_categorical(y_test, num_classes)


# In[10]:


print(f"Training hot label-> {y_train.shape}")
#print(f"Validation hot label-> {y_val.shape}")
print(f"Testing hot label-> {y_test.shape}")


# In[11]:


history1 = autoencoder.fit(x_train, x_train, epochs=50, batch_size=128, shuffle=True, validation_data=(x_test, x_test))


# In[12]:


with open('./autoencoder_history.pkl', 'wb') as file:
    pkl.dump(history1.history, file)


# In[45]:


with open('./autoencoder_history.pkl', 'rb') as f:
    data = pkl.load(f)


# In[55]:


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
min_loss = min(data['loss'])
min_val_loss = min(data['val_loss'])
plt.plot(data['loss'], label='Training Error')
plt.plot(data['val_loss'], label='Testing Error')
plt.title(f'Training: {min_loss:.2f} & Testing Error:{min_val_loss:.2f}')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()

max_r = max(data['r_squared'])
max_val_r = max(data['val_r_squared'])
plt.subplot(1, 2, 2)
plt.plot(data['r_squared'], label='Training R^2 Score')
plt.plot(data['val_r_squared'], label='Testing R^2 Score')
plt.title(f'Training:{max_r:.2f} & Testing R^2 Score:{max_val_r:.2f}')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()

plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[13]:


save_model(autoencoder, './autoencoder_model.h5')
print("Autoencoder Model Saved...")


# In[14]:


#autoencoder = load_model('./autoencoder_model.h5')


# In[15]:


from keras.models import Model
from keras.layers import Input, Conv1D, ELU, BatchNormalization, MaxPooling1D, LSTM, Dense, Softmax
from keras.metrics import categorical_accuracy, Precision, Recall


# In[16]:


def unweighted_accuracy(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=0)
    class_samples = K.sum(y_true, axis=0)
    
    unweighted_accuracy = K.sum(true_positives / K.maximum(class_samples, 1)) / K.sum(K.cast(class_samples > 0, 'float32'))
    return unweighted_accuracy



def weighted_accuracy(y_true, y_pred):
    class_weights = K.sum(y_true, axis=0)  # Assuming one-hot encoded labels
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=0)
    total_samples = K.sum(class_weights)
    
    weighted_accuracy = K.sum((true_positives / K.maximum(class_weights, 1)) * (class_weights / total_samples))
    return weighted_accuracy


# In[17]:


# Assuming 'autoencoder' is your trained autoencoder model
# Assuming 'latent_space_layer_name' is the name of the latent space layer in your autoencoder

# Extracting the encoder part from the autoencoder model
encoder_model = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer("latent_space_out").output, name='encoder_model')

# Freeze the layers of the encoder during classification
for layer in encoder_model.layers:
    layer.trainable = False

# Classifier architecture on top of the encoder
classifier_input = encoder_model.output

# Convolutional layers
classifier_output = Conv1D(128, kernel_size=3, padding='same', name='conv1')(classifier_input)
classifier_output = ELU(name='elu1')(classifier_output)
classifier_output = TimeDistributed(BatchNormalization(name='batch_norm1'))(classifier_output)
classifier_output = MaxPooling1D(pool_size=2, name='maxpool1')(classifier_output)

classifier_output = Conv1D(256, kernel_size=3, padding='same', name='conv2')(classifier_output)
classifier_output = ELU(name='elu2')(classifier_output)
classifier_output = TimeDistributed(BatchNormalization(name='batch_norm2'))(classifier_output)
classifier_output = MaxPooling1D(pool_size=2, name='maxpool2')(classifier_output)

classifier_output = Conv1D(512, kernel_size=3, padding='same', name='conv3')(classifier_output)
classifier_output = ELU(name='elu3')(classifier_output)
classifier_output = TimeDistributed(BatchNormalization(name='batch_norm3'))(classifier_output)
classifier_output = MaxPooling1D(pool_size=2, name='maxpool3')(classifier_output)

# LSTM layer
classifier_output = LSTM(128, name='lstm')(classifier_output)

# Dense layers
classifier_output = Dense(128, name='dense1')(classifier_output)
classifier_output = ELU(name='elu_dense1')(classifier_output)
classifier_output = BatchNormalization(name='batch_norm_dense1')(classifier_output)

classifier_output = Dense(8, activation='softmax', name='output')(classifier_output)

# Create the classifier model
classifier_model = Model(inputs=encoder_model.input, outputs=classifier_output, name='classifier_model')

# Compile the classifier model
classifier_model.compile(optimizer=SGD(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy', weighted_accuracy, unweighted_accuracy])

# Display the classifier model summary
classifier_model.summary()


# In[34]:


# Train the final model
history2 = classifier_model.fit(
    x_train, y_train,
    epochs=50,             
    batch_size=64,        
    shuffle=True,
    validation_data=(x_test, y_test))


# In[35]:


with open('./classifier_history.pkl', 'wb') as file:
    pkl.dump(history2.history, file)


# In[36]:


save_model(classifier_model, './classifier_model.h5')
print("classifier Model Saved...")


# In[37]:


y_pred_train = classifier_model.predict(x_train)
#y_pred_val = classifier_model.predict(x_val)
y_pred_test = classifier_model.predict(x_test)


# In[38]:


y_pred_train_labels = np.argmax(y_pred_train, axis=1)
#y_pred_val_labels = np.argmax(y_pred_val, axis=1)
y_pred_test_labels = np.argmax(y_pred_test, axis=1)


# In[39]:


cm1 = confusion_matrix(np.argmax(y_train, axis=1), y_pred_train_labels)
#cm2 = confusion_matrix(np.argmax(y_val, axis=1), y_pred_val_labels)
cm3= confusion_matrix(np.argmax(y_test, axis=1), y_pred_test_labels)


# In[40]:


sns.heatmap(cm1, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[41]:


sns.heatmap(cm3, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[ ]:




