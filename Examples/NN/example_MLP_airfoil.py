import pyLOM

model = pyLOM.NN.MLP(
        input_size=5,#x.shape[1],
        output_size=1, #y.shape[1],
        hidden_size=512,
        n_layers=3,
        p_dropouts=[0.07158, 0.03035, 0.15853]
    )

optimizer = pyLOM.NN.OptunaOptimizer(
    optimization_params={
        "lr": 0.01,  # fixed parameter
        "n_layers": (1, 3),  # optimizable parameter
        'hidden_size': 512,
        'n_layers': 3,
        'p_dropouts': [0.07158, 0.03035, 0.15853]
    },
    n_trials=100,
    direction='minimize',
    pruner=None,
    save_dir=None
)

print(model)