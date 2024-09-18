import numpy as np
import pyLOM


DATASET_PATH = "./mean_yaw_dataset.h5"
original_dataset = pyLOM.Dataset.load(DATASET_PATH)
print(original_dataset["Cp"].min(), original_dataset["Cp"].max(), original_dataset.mesh.xyz.min(), original_dataset.mesh.xyz.max())

alphas = [2.5, 5, 7.5, 10]

input_scaler = pyLOM.NN.MinMaxScaler()
output_scaler = pyLOM.NN.MinMaxScaler(feature_range=(-1, 1))

dataset = pyLOM.NN.Dataset(
    variables_out=(original_dataset["Cp"],),
    variables_in=original_dataset.mesh.xyz,
    parameters=[alphas],
    inputs_scaler=input_scaler,
    outputs_scaler=output_scaler,
)
dataset_train, dataset_test = dataset.get_splits((0.7, 0.3), random=True)
print(len(dataset_train), len(dataset_test), len(dataset))

x, y = dataset[:]
print(x.min(), x.max(), y.min(), y.max())
x, y = dataset_train[:]
print(x.min(), x.max(), y.min(), y.max())
x, y = dataset_test[:]
print(x.min(), x.max(), y.min(), y.max())


optimizer = pyLOM.NN.OptunaOptimizer(
    optimization_params={
        "lr": 0.01,  # fixed parameter
        "n_layers": (1, 4),  # optimizable parameter,
        "batch_size": (128, 512),
        "hidden_size": 256,
        "epochs": 30,
    },
    n_trials=10,
    direction="minimize",
    pruner=None,
    save_dir=None,
)

pipeline = pyLOM.NN.Pipeline(
    train_dataset=dataset_train,
    test_dataset=dataset_test,
    optimizer=optimizer,
    model_class=pyLOM.NN.MLP,
)

pipeline.run()
# check saving and loading the model
pipeline.model.save("model.pth")
model = pyLOM.NN.MLP.load("model.pth")

preds = model.predict(dataset, batch_size=2048)
scaled_preds = output_scaler.inverse_transform([preds])[0]
scaled_y = output_scaler.inverse_transform([dataset[:][1]])[0]
# check that the scaling is correct
print(scaled_y.min(), scaled_y.max())

print(f"MAE: {np.abs(scaled_preds - np.array(scaled_y)).mean()}")