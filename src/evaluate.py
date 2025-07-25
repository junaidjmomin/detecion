import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model
from preprocessing import load_data

X1, X2, Y = load_data('data/raw/A', 'data/raw/B', 'data/raw/label')
model = load_model('model_levir_cd.h5')
preds = model.predict([X1, X2]) > 0.5

Y_flat = Y.flatten()
preds_flat = preds.flatten()

precision = precision_score(Y_flat, preds_flat)
recall = recall_score(Y_flat, preds_flat)
f1 = f1_score(Y_flat, preds_flat)

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
