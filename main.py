#!/usr/bin/env python3

import os
from time import time

import numpy as np
import music21 as m21
import tensorflow as tf

# Parse MIDI file and convert notes to chords
# score = m21.converter.parse('./biga.mid').chordify()
# key = score.analyze('key')
# print(key)

# Define data directory
midi_dir = './dataset/130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]/Jazz_www.thejazzpage.de_MIDIRip/'
# Identify list of MIDI files
songList = os.listdir(midi_dir)[:10]
# Create empty list for scores
originalScores = [m21.converter.parse(midi_dir + song).chordify() for song in songList]

# Define empty lists of lists
originalChords = [[] for _ in originalScores]
originalDurations = [[] for _ in originalScores]
originalKeys = []

# Extract notes, chords, durations, and keys
for i, song in enumerate(originalScores):
    originalKeys.append(str(song.analyze('key')))
    for element in song:
        if isinstance(element, m21.note.Note):
            originalChords[i].append(element.pitch)
            originalDurations[i].append(element.duration.quarterLength)
        elif isinstance(element, m21.chord.Chord):
            originalChords[i].append('.'.join(str(n) for n in element.pitches))
            originalDurations[i].append(element.duration.quarterLength)

# Map unique chords to integers
uniqueChords = np.unique([i for s in originalChords for i in s])
chordToInt = dict(zip(uniqueChords, list(range(0, len(uniqueChords)))))
# Map unique durations to integers
uniqueDurations = np.unique([i for s in originalDurations for i in s])
durationToInt = dict(zip(uniqueDurations, list(range(0, len(uniqueDurations)))))
# Define sequence lenght
sequenceLength = 32
# Define empty array for train data
trainChords = []
trainDurations = []
targetChords = []
targetDurations = []
# Construct train and target sequences for chords and durations
for s in range(len(originalChords)):
    chordList = [chordToInt[c] for c in originalChords[s]]
    durationList = [durationToInt[d] for d in originalDurations[s]]
    for i in range(len(chordList) - sequenceLength):
        trainChords.append(chordList[i:i+sequenceLength])
        trainDurations.append(durationList[i:i+sequenceLength])
        targetChords.append(chordList[i+1])
        targetDurations.append(durationList[i+1])

# Convert to one-hot encoding and swap chord and sequence dimensions
trainChords = tf.keras.utils.to_categorical(trainChords).transpose(0,2,1)
trainDurations = tf.keras.utils.to_categorical(trainDurations).transpose(0,2,1)
targetChords = tf.keras.utils.to_categorical(targetChords).transpose(0,2,1)
targetDurations = tf.keras.utils.to_categorical(targetDurations).transpose(0,2,1)
# Convert data to numpy array of type float
trainChords = np.array(trainChords, np.float)
trainDurations = np.array(trainDurations, np.float)
targetChords = np.array(targetChords, np.float)
targetDurations = np.array(targetDurations, np.float)
# Define number of samples, notes and chords, and durations
nSamples = trainChords.shape[0]
nChords = trainChords.shape[1]
# TODO check what I've made wrong, it should be shape[1]
#nDurations = trainDurations.shape[1]
nDurations = trainDurations.shape[0]
# Set the input dimension
inputDim = nChords * sequenceLength
# Set the embedding layer dimension
embedDim = 64
# Define input layers
chordInput = tf.keras.layers.Input(shape = (None,))
durationInput = tf.keras.layers.Input(shape = (None,))
# Define embedding layers
chordEmbedding = tf.keras.layers.Embedding(nChords, embedDim, input_length = sequenceLength)(chordInput)
durationEmbedding = tf.keras.layers.Embedding(nDurations, embedDim, input_length = sequenceLength)(durationInput)
# Merge embedding layers using a concatenation layer
mergeLayer = tf.keras.layers.Concatenate(axis=1)([chordEmbedding, durationEmbedding])
# Define LSTM layer
lstmLayer = tf.keras.layers.LSTM(512, return_sequences=True)(mergeLayer)
# Define dense layer
denseLayer = tf.keras.layers.Dense(256)(lstmLayer)
# Define output layers
chordOutput = tf.keras.layers.Dense(nChords, activation = 'softmax')(denseLayer)
durationOutput = tf.keras.layers.Dense(nDurations, activation = 'softmax')(denseLayer)
# Define model
model = tf.keras.Model(inputs = [chordInput, durationInput], outputs = [chordOutput, durationOutput])
# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
# Train the model
model.fit(
    [trainChords, trainDurations], [targetChords, targetDurations],
    epochs=500, batch_size=64)
# Save the model
model.save(f'models/{time()}.h5')
