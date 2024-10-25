#!/usr/bin/env python
#
# Example of MLP.
#
# Last revision: 23/10/2024

import os, numpy as np, torch, matplotlib.pyplot as plt
import pyLOM


def generate_synthetic_data(n_samples=1000, noise_level=0.05):
    # Generate input features
    X = np.random.randn(n_samples, 2)

    # Generate target values using a nonlinear function
    y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + 0.5 * X[:, 0] ** 2

    # Add noise
    y += noise_level * np.random.randn(n_samples)

    return X, y


def true_vs_pred_plot(y_true, y_pred, path):
    """
    Auxiliary function to plot the true vs predicted values
    """
    num_plots = y_true.shape[1]
    plt.figure(figsize=(10, 5 * num_plots))
    for j in range(num_plots):
        plt.subplot(num_plots, 1, j + 1)
        plt.scatter(y_true[:, j], y_pred[:, j], s=1, c="b", alpha=0.5)
        plt.xlabel("True values")
        plt.ylabel("Predicted values")
        plt.title(f"Scatterplot for Component {j+1}")
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(path, dpi=300)

def plot_train_test_loss(train_loss, test_loss, path):
    """
    Auxiliary function to plot the training and test loss
    """
    plt.figure()
    plt.plot(range(1, len(train_loss) + 1), train_loss, label="Training Loss")
    total_epochs = len(test_loss) # test loss is calculated at the end of each epoch
    total_iters = len(train_loss) # train loss is calculated at the end of each iteration/batch
    iters_per_epoch = total_iters // total_epochs
    plt.plot(np.arange(iters_per_epoch, total_iters+1, step=iters_per_epoch), test_loss, label="Test Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Epoch")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.savefig(path, dpi=300)

device = pyLOM.NN.select_device("cpu") # Force CPU for this example, if left in blank it will automatically select the device
RESUDIR = 'MLP_DLR_airfoil'
pyLOM.NN.create_results_folder(RESUDIR)

input_scaler = pyLOM.NN.MinMaxScaler()
output_scaler = pyLOM.NN.MinMaxScaler()

X, y = generate_synthetic_data(n_samples=1250)

dataset = pyLOM.NN.Dataset(
    variables_out=(y,), 
    variables_in=X,
    inputs_scaler=input_scaler,
    outputs_scaler=output_scaler,
    snapshots_by_column=False
)

td_train, td_test = dataset.get_splits([0.8, 0.2])

training_params = {
    "epochs": 50,
    "lr": 0.00125,
    "lr_gamma": 0.96,
    "lr_scheduler_step": 1,
    "batch_size": 32,
    "loss_fn": torch.nn.MSELoss(),
    "optimizer_class": torch.optim.Adam,
    "print_rate_epoch": 10,
}

sample_input, sample_output = td_train[0]
model = pyLOM.NN.MLP(
    input_size=sample_input.shape[0],
    output_size=sample_output.shape[0],
    hidden_size=32,
    n_layers=2,
    p_dropouts=0.1,
)

pipeline = pyLOM.NN.Pipeline(
    train_dataset=td_train,
    test_dataset=td_test,
    model=model,
    training_params=training_params,
)

training_logs = pipeline.run()


## check saving and loading the model
pipeline.model.save(os.path.join(RESUDIR,"model.pth"))
model = pyLOM.NN.MLP.load(RESUDIR + "/model.pth")
preds = model.predict(td_test, batch_size=250)

scaled_preds = output_scaler.inverse_transform([preds])[0]
scaled_y     = output_scaler.inverse_transform([td_test[:][1]])[0]


evaluator = pyLOM.NN.RegressionEvaluator()
evaluator(scaled_y, scaled_preds)
evaluator.print_metrics()

true_vs_pred_plot(scaled_y, scaled_preds, RESUDIR + '/true_vs_pred.png')
plot_train_test_loss(training_logs['train_loss'], training_logs['test_loss'], RESUDIR + '/train_test_loss.png')

pyLOM.cr_info()
plt.show()