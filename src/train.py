import numpy as np
from preprocessing import load_data
from model import build_model
import tensorflow as tf

X1, X2, Y = load_data('data/raw/A', 'data/raw/B', 'data/raw/label')
model = build_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit([X1, X2], Y, batch_size=8, epochs=20, validation_split=0.1)
model.save('model_levir_cd.h5')
