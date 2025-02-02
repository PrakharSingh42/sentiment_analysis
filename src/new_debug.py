import pickle

with open("logistic_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

print(f"Loaded model type: {type(model)}")  # Debugging
