import pyLOM

model = pyLOM.NN.MLP(
        input_size=5,#x.shape[1],
        output_size=1, #y.shape[1],
        hidden_size=512,
        n_layers=3,
        p_dropouts=[0.07158, 0.03035, 0.15853]
    )

print(model)