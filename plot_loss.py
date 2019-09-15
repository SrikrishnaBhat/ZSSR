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

scenes_dir = 'results_signal_fire_productions_sf2'
scenes_list = os.listdir(scenes_dir)
scenes_list.sort()
loss_src_dir = os.path.join('losses', scenes_dir)
if not os.path.exists(loss_src_dir ):
    os.makedirs(loss_src_dir )
for scene in scenes_list:
    df = None
    loss_dir = os.path.join(scenes_dir, scene, 'loss')
    losses = os.listdir(loss_dir)
    losses.sort()
    for loss_file in losses:
        if '.log' not in loss_file:
            continue
        temp_df = pd.read_csv(os.path.join(loss_dir, loss_file), delimiter='|', header=None, index_col=None)
        if df is None:
            df = temp_df
        else:
            df = df.append(temp_df, ignore_index=True)

    dest_path = os.path.join(loss_src_dir , 'loss_{}.png'.format(scene))
    print(dest_path)
    if len(df.columns) == 2:
        plot_and_save_loss(df, dest_path)
    else:
        plot_and_save_train_test_loss(df, dest_path)
