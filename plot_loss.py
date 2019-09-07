from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os

def plot_and_save_loss(df, dest_path):
    losses = df.loc[:, 1]
    fig = plt.figure()
    plt.plot(losses)
    plt.savefig(dest_path)

def plot_and_save_train_test_loss(df, dest_path):
    train_loss = df.loc[:, 2]
    test_loss = df.loc[:, 3]
    indices = df.loc[:, 1]
    fig = plt.figure()
    plt.plot(train_loss, label='train_loss')
    plt.plot(test_loss, label='test_loss')
    plt.legend()
    plt.savefig(dest_path)

loss_dir = 'losses/results_some_russians_band_sf4_n12'
losses = os.listdir(loss_dir)
losses.sort()
df = None
for loss_file in losses:
    if '.log' not in loss_file:
        continue
    temp_df = pd.read_csv(os.path.join(loss_dir, loss_file), delimiter='|', header=None, index_col=None)
    if df is None:
        df = temp_df
    else:
        df = df.append(temp_df, ignore_index=True)
print('Final loss value: {}'.format(df.values[-1, :]))
dest_path = os.path.join(loss_dir, 'loss.png')
if len(df.columns) == 2:
    plot_and_save_loss(df, dest_path)
else:
    plot_and_save_train_test_loss(df, dest_path)
