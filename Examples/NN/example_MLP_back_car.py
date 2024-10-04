import numpy as np
import pyLOM
from pathlib import Path

DATASET_DIR = Path("/home/david/Desktop/Datos_DLR_pylom")


input_scaler = pyLOM.NN.MinMaxScaler()
output_scaler = pyLOM.NN.MinMaxScaler()   

def load_dataset(path):
    original_dataset = pyLOM.Dataset.load(path)
    # print(original_dataset.fieldnames, original_dataset.varnames, len(original_dataset.get_variable('AoA')), len(original_dataset.get_variable('Mach')))
    print(original_dataset["CP"].shape, original_dataset["CP"].max(), original_dataset.xyz.shape, original_dataset.xyz.min(), original_dataset.xyz.max())
    
    # print(len([*zip(original_dataset.get_variable('AoA'), original_dataset.get_variable('Mach'))]), torch.tensor([*zip(original_dataset.get_variable('AoA'), original_dataset.get_variable('Mach'))]).shape)
    # import sys; sys.exit() 
    dataset = pyLOM.NN.Dataset(
        variables_out=(original_dataset["CP"],),
        variables_in=original_dataset.xyz,
        parameters=[[*zip(original_dataset.get_variable('AoA'), original_dataset.get_variable('Mach'))]], # , original_dataset.get_variable('Mach')
        # parameters=[original_dataset.get_variable('AoA'), original_dataset.get_variable("Mach")],
        inputs_scaler=input_scaler,
        outputs_scaler=output_scaler,
    )
    return dataset
# LOS DATASETS PUEDE QUE ESTEN MAL. HAY QUE COMPROBAR SI EL ORDEN DEL MACH Y AOA ES CORRECTO
dataset_train = load_dataset(DATASET_DIR / "TRAIN.h5")
dataset_test = load_dataset(DATASET_DIR / "TEST.h5")
val_dataset = load_dataset(DATASET_DIR / "VAL.h5")
print(len(dataset_train), len(dataset_test), len(val_dataset))
x, y = dataset_train[:]
print(x[:598])
print(x.min(dim=0), x.max(dim=0), y.min(dim=0), y.max(dim=0), x.shape, y.shape)
x, y = dataset_test[:]
print(x.min(), x.max(), y.min(), y.max(), x.shape, y.shape)
x, y = val_dataset[:]
print(x.min(), x.max(), y.min(), y.max(), x.shape, y.shape)


optimizer = pyLOM.NN.OptunaOptimizer(
    optimization_params={
        "lr": 0.01,  # fixed parameter
        "n_layers": (1, 4),  # optimizable parameter,
        "hidden_size": (128, 512),
        "batch_size": 256,
        "epochs": 50,
    },
    n_trials=1,
    direction="minimize",
    pruner=None,
    save_dir=None,
)

pipeline = pyLOM.NN.Pipeline(
    train_dataset=dataset_train,
    test_dataset=dataset_test,
    valid_dataset=val_dataset,
    optimizer=optimizer,
    model_class=pyLOM.NN.MLP,
)

pipeline.run()
# check saving and loading the model
pipeline.model.save("model.pth")
model = pyLOM.NN.MLP.load("model.pth")

preds = model.predict(dataset_test, batch_size=2048)
scaled_preds = output_scaler.inverse_transform([preds])[0]
scaled_y = output_scaler.inverse_transform([dataset_test[:][1]])[0]
# check that the scaling is correct
print(scaled_y.min(), scaled_y.max())

print(f"MAE: {np.abs(scaled_preds - np.array(scaled_y)).mean()}")