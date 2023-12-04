import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import h5py

from goodnet import GoodNet


def calculate_accuracy(predictions, labels):
    """
    Calculates the accuracy percentage between predicted labels and true labels.
    """
    return np.mean(np.equal(predictions, labels)) * 100


def prune_model(
    model_path, model, layer_index, threshold, validation_data, backdoored_data
):
    """
    Prunes a specified layer of a neural network model and
    evaluates its performance on validation and backdoored data.
    """

    original_accuracy = calculate_accuracy(
        np.argmax(model.predict(validation_data[0]), axis=1),
        validation_data[1],
    )
    pruned_model = keras.models.load_model(model_path)
    print(f"Original Validation Accuracy: {original_accuracy}")

    intermediate_model = keras.models.Model(
        inputs=model.input, outputs=model.layers[layer_index].output
    )
    activations = intermediate_model.predict(validation_data[0])
    avg_activations = activations.mean(axis=(0, 1, 2))

    total_channels = len(avg_activations)
    results = []

    # Channels are removed in non-decreasing order of average
    # activation values over the entire validation set.
    # See: Section 3.1 Pruning Defense (https://arxiv.org/pdf/1805.12185.pdf)
    for pruned_channels, channel in enumerate(np.argsort(avg_activations)):
        layer = pruned_model.layers[layer_index - 1]
        layer.kernel[:, :, :, channel].assign(
            tf.zeros_like(layer.kernel[:, :, :, channel])
        )
        layer.bias[channel].assign(tf.zeros_like(layer.bias[channel]))

        current_accuracy = calculate_accuracy(
            np.argmax(pruned_model.predict(validation_data[0]), axis=1),
            validation_data[1],
        )
        attack_success_rate = calculate_accuracy(
            np.argmax(pruned_model.predict(backdoored_data[0]), axis=1),
            backdoored_data[1],
        )
        fraction = pruned_channels / total_channels
        results.append((fraction, attack_success_rate, current_accuracy))
        print(
            f"Channels Pruned: {(fraction * 100):.2f}, "
            f"ASR: {attack_success_rate:.2f}, "
            f"Validation Accuracy: {current_accuracy:.2f}"
        )

        if threshold is not None:
            if abs(original_accuracy - current_accuracy) > threshold:
                break

    return pruned_model, results


def prune_defense(
    accuracy_threshold,
    model_path,
    cl_x_valid,
    cl_y_valid,
    bd_x_valid,
    bd_y_valid,
    cl_x_test,
    cl_y_test,
):
    """
    Performs pruning on a neural network model to defend against
    backdoor attacks and evaluates the effectiveness of the defense.
    """
    original_model = keras.models.load_model(model_path)

    # Prune the last pooling layer by removing one channel
    # at a time from that layer. Every time a channel is pruned,
    # we measure the new validation accuracy of the pruned badnet.
    # We stop pruning once the validation accuracy drops by
    # `accuracy_threshold` of the original accuracy.
    pruned_model, results = prune_model(
        model_path,
        model=original_model,
        layer_index=6,
        threshold=accuracy_threshold,
        validation_data=(cl_x_valid, cl_y_valid),
        backdoored_data=(bd_x_valid, bd_y_valid),
    )

    # For each test input, we run the `goodnet` through
    # both original badnet and pruned badnet. If the
    # classification outputs are the same, i.e., for class 'i',
    # it gives output class 'i'. If they differ, it will
    # output N+1
    goodnet = GoodNet(original_model, pruned_model)

    if accuracy_threshold is not None:
        print("======================================")
        print(f"Accuracy Drop Threshold: {accuracy_threshold:.2f}%")
        clean_accuracy = calculate_accuracy(
            np.argmax(goodnet(cl_x_test), axis=1), cl_y_test
        )
        print(f"Clean Test Accuracy of GoodNet: {clean_accuracy:.2f}")

        attack_success_rate = calculate_accuracy(
            np.argmax(goodnet(bd_x_valid), axis=1), bd_y_valid
        )
        print(
            f"Attack Success Rate of GoodNet: {attack_success_rate:.2f}",
        )
        print("======================================\n\n")

    return pruned_model, results


def load_h5_dataset(filepath):
    """
    Load dataset from an H5 file.

    Parameters:
    filepath (str): The file path to the H5 dataset file.

    Returns:
    tuple: A tuple containing two numpy arrays: features (x_data) and labels (y_data).
    """
    try:
        with h5py.File(filepath, "r") as data:
            x_data = np.array(data["data"])
            y_data = np.array(data["label"])
            x_data = x_data.transpose(
                (0, 2, 3, 1)
            )  # Transpose the data to the correct format if necessary
            return x_data, y_data
    except (IOError, KeyError) as e:
        print(f"An error occurred while loading the dataset from {filepath}: {e}")
        return None, None


def generate_plots(filename, results):
    """
    Generate and save a plot of Attack Success Rate (ASR) and
    Validation Accuracy against the fraction of channels pruned.
    """
    fractions_pruned, asr_values, accuracy_values = zip(*results)
    plt.plot(
        fractions_pruned,
        asr_values,
        color="r",
        label="Attack Success Rate (%)",
    )
    plt.plot(
        fractions_pruned,
        accuracy_values,
        color="b",
        label="Validation Accuracy (%)",
    )
    plt.title("ASR and Validation Accuracy vs Fraction of Channels Pruned")
    plt.xlabel("Fraction of Channels Pruned")
    plt.ylabel("Percentage (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plots/{filename}")
    plt.close()
    print("Plot has been saved successfully.")


def run():
    # Base directory for all file paths
    base_path = "."

    # Path for the bad network model
    bad_net_model_path = f"{base_path}/models/bd_net.h5"

    # Paths for clean data
    clean_data_paths = {
        "validation": f"{base_path}/data/cl/valid.h5",
        "test": f"{base_path}/data/cl/test.h5",
    }

    # Paths for poisoned data
    poisoned_data_paths = {
        "validation": f"{base_path}/data/bd/bd_valid.h5",
        "test": f"{base_path}/data/bd/bd_test.h5",
    }

    # Clean data
    cl_x_valid, cl_y_valid = load_h5_dataset(clean_data_paths["validation"])
    cl_x_test, cl_y_test = load_h5_dataset(clean_data_paths["test"])

    # Poisoned data
    bd_x_valid, bd_y_valid = load_h5_dataset(poisoned_data_paths["validation"])
    bd_x_test, bd_y_test = load_h5_dataset(poisoned_data_paths["test"])
    print("Datasets loaded ...")

    # Validation accuracy thresholds
    accuracy_thresholds = [2, 4, 10]
    for x in accuracy_thresholds:
        model, pruning_results = prune_defense(
            x,
            bad_net_model_path,
            cl_x_valid,
            cl_y_valid,
            bd_x_valid,
            bd_y_valid,
            cl_x_test,
            cl_y_test,
        )
        model.save(f"{base_path}/models/pruned_net_{x}.h5", save_format="h5")
        generate_plots(f"pruned_net_{x}", pruning_results)

    # Measure the attack success rate (on backdoored test data)
    # as a function of the fraction of channels pruned.
    model, pruning_results = prune_defense(
        None,
        bad_net_model_path,
        cl_x_valid,
        cl_y_valid,
        bd_x_valid,
        bd_y_valid,
        cl_x_test,
        cl_y_test,
    )
    model.save(f"{base_path}/models/pruned_net.h5", save_format="h5")
    generate_plots("pruned_net", pruning_results)


if __name__ == "__main__":
    run()
