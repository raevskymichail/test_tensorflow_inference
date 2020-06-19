import os
import time
import numpy as np
import tensorflow as tf
import pandas as pd
import h5py
import seaborn as sns
import matplotlib.pyplot as plt


def test_inference_time(model, input_data, device='Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz', n_samples_list=[1, 50, 100, 150, 200], n_rounds=10):
    results = pd.DataFrame(columns=['inference_time', 'n_samples'])
    for n_samples in n_samples_list:
        time_n_samples = np.array([])
        for f in range(n_rounds):
            start_time = time.time()
            y_pred = model.predict(input_data[0:n_samples, :, :])
            end_time = time.time()
            time_n_samples = np.append(time_n_samples, end_time - start_time)

        results_n_samples = pd.DataFrame({'inference_time': time_n_samples})
        results_n_samples['n_samples'] = n_samples
        results = results.append(results_n_samples)

    results['device'] = device
    return results


def test_inference_time_on_n_threads(model, input_data, n_treads_list=[1, 4, 8, 16, 32]):
    results = pd.DataFrame()
    for n_treads in n_treads_list:
        tf.config.threading.set_intra_op_parallelism_threads(n_treads)
        tf.config.threading.set_inter_op_parallelism_threads(n_treads)
        results_n_threads = test_inference_time(model, input_data)
        results_n_threads['n_treads'] = n_treads
        results = results.append(results_n_threads)
    return results


def plot_inference_results(results, title, save_path=None):
    results['n_treads'] = results['n_treads'].astype(str) + ' cores'
    plt.figure(figsize=(8, 5))
    sns.pointplot(x='n_samples', y='inference_time', hue='n_treads',
                  data=results, palette='Set2', capsize=.1)
    plt.grid(color='grey', linestyle='--')
    # plt.title("ResNet on Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz")
    # plt.title("ResNet on Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz 2.00GHz")
    plt.axhline(0.5, color='red', linestyle='--', label='0.5 sec')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# # Process the inference

# with np.load('input_data.npz') as f:
#     input_data = f['data']

# model_best = tf.keras.models.load_model('model_best.hdf5')
# model_last = tf.keras.models.load_model('model_last.hdf5')

# results_model_best = test_inference_time_on_n_threads(model_best)

# plot_inference_results(results_model_best,
#                        title='ResNet on Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz 2.00GHz',
#                        save_path='data/resnet_inference_laptop.svg')
