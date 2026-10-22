import pickle

with open("preprocess/preprocessed/splits.pkl", "rb") as f:
    data = pickle.load(f)

print(data)