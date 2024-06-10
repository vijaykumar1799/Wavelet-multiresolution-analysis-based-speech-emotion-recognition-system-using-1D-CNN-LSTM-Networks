#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import matplotlib.pyplot as plt

import pywt
import librosa
import librosa.display

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

import random
from random import seed, random, randint, sample
from scipy.signal import hilbert, chirp
from scipy.io import wavfile
from tqdm import tqdm
from scipy.interpolate import interp2d

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# using speech data
speech_folder_name = './Audio_Speech_Actors_01-24/'
actors_folder_name = [os.path.join(speech_folder_name, actor) for actor in os.listdir(speech_folder_name)]
audio_files_path = [os.path.join(actor_num, file) for actor_num in actors_folder_name for file in os.listdir(actor_num)]
data = np.array([[file_path, int(file_path.split('\\')[-1].split('-')[2])-1] for file_path in audio_files_path])

labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
print(f"{len(audio_files_path)} Audio files fetched...\n")
labels_idx, count = np.unique(data[:, -1], return_counts=True)
for i in range(len(count)):
    print(f"{labels[int(labels_idx[i])]} -> {count[i]} samples.")


# In[3]:


def add_awgn(audio):
    snr_db = np.random.uniform(15, 30)
    noise_std = np.sqrt(np.var(audio) / (10 ** (snr_db / 10)))
    gaussian_noise = np.random.normal(0, noise_std, len(audio))
    return audio + gaussian_noise


# In[4]:


def preprocess_audio(audio):
    trimmed, idx = librosa.effects.trim(audio)
    norm_seq = (trimmed - np.mean(trimmed)) / np.std(trimmed)
    noisy = add_awgn(norm_seq)

    return norm_seq, noisy


# In[5]:


def compute_wavelet_features(audio, label):
    wavelet = 'morl'
    sr = 16000
    widths = np.arange(1, 256)
    #print(f"Scales using: {widths}")

    dt = 1/sr
    frequencies = pywt.scale2frequency(wavelet=wavelet, scale=widths) / dt
    #print(f"Frequencies associated with the scales: {frequencies}")

    #creating filter to select frequencies between 20Hz and 5Khz - this is where most speech lies
    upper = [x for x in range(len(widths)) if frequencies[x] > 2000][-1]
    lower = [x for x in range(len(widths)) if frequencies[x] < 100][0]

    widths = widths[upper:lower]

    #computing wavelet transform 
    wavelet_coefs, freqs = pywt.cwt(audio, widths, wavelet=wavelet, sampling_period=dt)
    #print(f"shape of wavelet transform: {wavelet_coefs.shape}")

    # Fixed Segment Generation
    start = 0
    end = wavelet_coefs.shape[1]
    frames = []
    frame_size = 4000
    count = 0

    while start + frame_size <= end -1:
        f = (wavelet_coefs)[:, start:start+frame_size]
        assert f.shape[1] == frame_size
        frames.append(f)
        start += frame_size

    frames = np.array(frames)
    frames = frames.reshape((len(frames), frame_size, wavelet_coefs.shape[0]))
    labels = np.ones(shape=(len(frames), 1))* int(label)

    return frames, labels


# In[6]:


#data = np.array([[file, int(file.split('\\')[-1].split('-')[2])-1] for file in audio_files_path])


# In[7]:


x_train, x_, y_train, y_ = train_test_split(data[:60, 0], data[:60, -1], test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_, y_, test_size=0.25, random_state=42)
labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

print(f"Training: {x_train.shape}, labels: {y_train.shape}")
print(f"Validation: {x_val.shape}, labels: {y_val.shape}")
print(f"Testing: {x_test.shape}, labels: {y_test.shape}")


# In[8]:


print(np.unique(y_train, return_counts=True))
print(np.unique(y_val, return_counts=True))
print(np.unique(y_test, return_counts=True))


# In[9]:


# Training data saving
# Set a seed for reproducibility
seed(42)

# Initialize lists to store data
x_train_wavelet = []
y_train_wavelet = []
uniq_id = []

# Iterate over individual labels
count = 0
num_rand_samp = 100

for label_index in range(len(labels)):
    label_indices = np.where(y_train == str(label_index))[0]
    selected_indices = sample(label_indices.tolist(), min(num_rand_samp, len(label_indices)))

    for audio_index in tqdm(selected_indices, desc=f"Label {label_index}"):
        current_sample = x_train[audio_index]
        seq, _ = librosa.load(current_sample, sr=16000)
        normalised_audio, noisy_audio = preprocess_audio(audio=seq)

        for audio_type, audio_data in enumerate([normalised_audio, noisy_audio]):
            features, labelss = compute_wavelet_features(audio=audio_data, label=label_index)

            # Randomly sample from features
            indices = np.arange(len(features))
            selected_indices = sample(indices.tolist(), min(num_rand_samp, len(indices)))
            selected_features = features[selected_indices]

            # Update lists
            uniq_id += [count] * len(selected_features)
            y_train_wavelet.extend(labelss)

            if count == 0:
                x_train_wavelet = selected_features
            else:
                x_train_wavelet = np.concatenate((x_train_wavelet, selected_features), axis=0)

            count += 1

print(f"X: {x_train_wavelet.shape}")


# In[10]:


y_train_wavelet = np.array(y_train_wavelet)
print("Y: ", y_train_wavelet.shape, " unique: ", np.unique(y_train_wavelet, return_counts=True))


# In[11]:


# Write all features to a .npz file
np.savez_compressed(os.getcwd()+"/training_features", a=x_train_wavelet, b=y_train_wavelet)


# ### Validation Data saving...

# In[12]:


# validation data saving
# Set a seed for reproducibility
seed(42)

# Initialize lists to store data
x_val_wavelet = []
y_val_wavelet = []
uniq_id = []

# Iterate over individual labels
count = 0
num_rand_samp = 100

for label_index in range(len(labels)):
    label_indices = np.where(y_val == str(label_index))[0]
    selected_indices = sample(label_indices.tolist(), min(num_rand_samp, len(label_indices)))

    for audio_index in tqdm(selected_indices, desc=f"Label {label_index}"):
        current_sample = x_val[audio_index]
        seq, _ = librosa.load(current_sample, sr=16000)
        normalised_audio, noisy_audio = preprocess_audio(audio=seq)

        for audio_type, audio_data in enumerate([normalised_audio, noisy_audio]):
            features, labelss = compute_wavelet_features(audio=audio_data, label=label_index)

            # Randomly sample from features
            indices = np.arange(len(features))
            selected_indices = sample(indices.tolist(), min(num_rand_samp, len(indices)))
            selected_features = features[selected_indices]

            # Update lists
            uniq_id += [count] * len(selected_features)
            y_val_wavelet.extend(labelss)

            if count == 0:
                x_val_wavelet = selected_features
            else:
                x_val_wavelet = np.concatenate((x_val_wavelet, selected_features), axis=0)

            count += 1

print(f"X: {x_val_wavelet.shape}")


# In[13]:


y_val_wavelet = np.array(y_val_wavelet)
print("Y: ", y_val_wavelet.shape, " unique: ", np.unique(y_val_wavelet, return_counts=True))
# Write all features to a .npz file
np.savez_compressed(os.getcwd()+"/validation_features", a=x_val_wavelet, b=y_val_wavelet)


# ### Testing data saving...

# In[14]:


# validation data saving
# Set a seed for reproducibility
seed(42)

# Initialize lists to store data
x_test_wavelet = []
y_test_wavelet = []
uniq_id = []

# Iterate over individual labels
count = 0
num_rand_samp = 100

for label_index in range(len(labels)):
    label_indices = np.where(y_test == str(label_index))[0]
    selected_indices = sample(label_indices.tolist(), min(num_rand_samp, len(label_indices)))

    for audio_index in tqdm(selected_indices, desc=f"Label {label_index}"):
        current_sample = x_test[audio_index]
        seq, _ = librosa.load(current_sample, sr=16000)
        normalised_audio, noisy_audio = preprocess_audio(audio=seq)

        for audio_type, audio_data in enumerate([normalised_audio, noisy_audio]):
            features, labelss = compute_wavelet_features(audio=audio_data, label=label_index)

            # Randomly sample from features
            indices = np.arange(len(features))
            selected_indices = sample(indices.tolist(), min(num_rand_samp, len(indices)))
            selected_features = features[selected_indices]

            # Update lists
            uniq_id += [count] * len(selected_features)
            y_test_wavelet.extend(labelss)

            if count == 0:
                x_test_wavelet = selected_features
            else:
                x_test_wavelet = np.concatenate((x_test_wavelet, selected_features), axis=0)

            count += 1

print(f"X: {x_test_wavelet.shape}")


# In[15]:


y_test_wavelet = np.array(y_test_wavelet)
print("Y: ", y_test_wavelet.shape, " unique: ", np.unique(y_test_wavelet, return_counts=True))
# Write all features to a .npz file
np.savez_compressed(os.getcwd()+"/testing_features", a=x_test_wavelet, b=y_test_wavelet)


# In[ ]:




