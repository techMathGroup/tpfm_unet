import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import plot_quiver, plot_pressure


def prepare_dataset(dataset_dir, out_file, dimensions=(64, 64)):
    height, width = dimensions

    files = os.listdir(dataset_dir)
    files = [f for f in files if f.endswith('.dat')]
    files.sort()

    coords_file = pd.read_csv(f'{dataset_dir}/coordinates.csv')
    coords_array = coords_file.drop(columns=['cellI']).to_numpy()

    inputs = []
    outputs = []
    for f in files:
        print(f)
        df = pd.read_csv(f'{dataset_dir}/{f}').drop(columns=['cellI'])
        df['x'] = coords_file['x']
        df['y'] = coords_file['y']
        df['z'] = coords_file['z']
        df = df.to_numpy().T.reshape(-1, height, width)
        inputs.append(df[4:])
        outputs.append(df[0:4])

    # Verify shapes
    print(f"Input shape: {np.array(inputs).shape}")
    print(f"Output shape: {np.array(outputs).shape}")

    # Visualize a sample
    sample_input = inputs[0]
    sample_output = outputs[0]

    plt.imshow(sample_input[0], cmap='gray', origin='lower')

    plot_quiver(sample_output, coords_array)
    plot_pressure(sample_output, coords_array)

    # Create a test split and put it aside
    test_size = int(0.1 * len(inputs))
    test_inputs = np.array(inputs[:-test_size])
    test_outputs = np.array(outputs[:-test_size])

    # Save
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    np.savez(out_file, inputs=inputs, outputs=outputs)
    np.savez(out_file.replace('.npz', '_test.npz'), inputs=test_inputs, outputs=test_outputs)


if __name__ == "__main__":
    dim = 128
    trait = "_upscaled_x2"
    dataset_dir = f"../working/mixer_{dim}{trait}"
    out_file = f"../data/mixer_{dim}{trait}.npz"
    prepare_dataset(dataset_dir, out_file, dimensions=(dim, dim))
    print("Dataset preparation complete.")