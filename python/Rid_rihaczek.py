#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('config', 'Completer.use_jedi = False')


# In[2]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import scipy.io as sio
import mne

from scipy import signal

import pac


def plot_pac(pac, high_freq, low_freq):
    fig = plt.figure(figsize=(7, 15))
    ax = fig.subplots()
    im = ax.imshow(pac, origin='lower', interpolation='nearest')

    xticks_num = (low_freq[1] - low_freq[0]) / 5
    yticks_num = (high_freq[1] - high_freq[0]) / 10

    ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
    ticks_loc = ax.get_xticks()
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    xticks = [''] + [int(n) for n in np.linspace(low_freq[0],
                                                 low_freq[1], ticks_loc.shape[0]-2).tolist()] + ['']
    ax.set_xticklabels(xticks)

    ax.yaxis.set_major_locator(mticker.MaxNLocator(10))
    ticks_loc = ax.get_yticks()
    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    yticks = [''] + [int(n) for n in np.linspace(high_freq[0],
                                                 high_freq[1], ticks_loc.shape[0]-2).tolist()] + ['']
    ax.set_yticklabels(yticks)

    plt.show()


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def bandpass_firwin(ntaps, lowcut, highcut, fs, window='hamming'):
    nyq = 0.5 * fs
    taps = signal.firwin(ntaps, [lowcut, highcut], nyq=nyq, pass_zero=False,
                         window=window, scale=False)
    return taps


def create_tasks_df(df):
    tasks = []

    for _, participant in df.iterrows():
        if participant.Group == 'PD':
            sessions = [(1, 1 * (participant.sess1_Med == 'ON')),
                        (2, 1 * (participant.sess2_Med == 'ON'))]
        else:
            sessions = [(1, 2)]

        for sess, pd_drug_type in sessions:
            participant_tasks = {}
            participant_tasks['participant_id'] = participant['participant_id']
            participant_tasks['pd_drug_type'] = pd_drug_type
            participant_tasks['isMale'] = participant['sex'] == 'Male'
            participant_tasks['age'] = participant['age']
            participant_tasks['dir'] = os.path.join(
                DS_PATH, participant['participant_id'], f'ses-{sess:02}', 'eeg',)
            participant_tasks['file'] = f'{participant["participant_id"]}_ses-{sess:02}_eeg_{participant["participant_id"]}_ses-{sess:02}_task-Rest_eeg.mat'
            participant_tasks['file_formatter'] = f'{participant["participant_id"]}_ses-{sess:02}_task-Rest_{{}}'
            participant_tasks['path'] = os.path.join(
                participant_tasks['dir'], participant_tasks['file'])

            tasks.append(participant_tasks)

    tasks_df = pd.DataFrame(tasks)

    return tasks_df


def _test_tasks_df(tasks_df, i=0):
    task = tasks_df.iloc[i]

    assert os.path.exists(task.path)

    ds = sio.loadmat(task.path)

    ds['data'] = ds['EEG']

    nbchan = ds['data'][0, 0]['nbchan'][0, 0]  # .dtype
    Fs = ds['data'][0, 0]['srate'][0, 0]
    times = ds['data'][0, 0]['times']
    data = ds['data'][0, 0]['data']

    dtypes = [k for k in ds['data'][0, 0]['event'].dtype.names]
    events = pd.DataFrame([{n: event[n].item() if event[n].size > 0 else None for n in dtypes}
                           for event in ds['data'][0, 0]['event'][0]])

    electrodes = pd.read_csv(os.path.join(
        task['dir'], task['file_formatter'].format('electrodes.tsv')), sep='\t')

    reject = ds['data'][0, 0]['reject']

    print(events)


def analyse_events(events, data, Fs, task=None):
    epochs = {'S200': [], 'S201': [], 'S202': []}
    mvls = {'S200': [], 'S201': [], 'S202': []}
    mvl_2ds = {'S200': np.zeros((data.shape[0] - 4, data.shape[0] - 4, 200-32+1, 40-4+1)),
               'S201': np.zeros((data.shape[0] - 4, data.shape[0] - 4, 200-32+1, 40-4+1)),
               'S202': np.zeros((data.shape[0] - 4, data.shape[0] - 4, 200-32+1, 40-4+1))}

    for event in events.iloc:
        if event.type not in epochs.keys():
            continue

        epoch = data[:, event.latency-100:event.latency+500]
        epochs[event.type].append(epoch)

        mvl_2d = np.zeros(
            (epoch.shape[0] - 4, epoch.shape[0] - 4, 200-32+1, 40-4+1))
        mvl = np.zeros((epoch.shape[0] - 4, epoch.shape[0] - 4, ))
        tfds = []
        for ch in range(epoch.shape[0] - 4):
            # print(ch)
            tfd = pac.rid_rihaczek(epoch[ch], Fs)
            tfds.append(tfd)

        if task is not None:
            print(task.participant_id, event.bvmknum,
                  'tfd completed for all channels')

        for chx in range(epoch.shape[0] - 4):
            chy = chx
            # for chy in range(chx, epoch.shape[0] - 4):
            # print(chx, chy)

            mvl_2d[chx, chy] = pac.tfMVL_tfd2_2d(
                tfds[chx], tfds[chy], [32, 200], [4, 40])
            #pac.tfMVL_tfd_2d(tfd, [32, 200], [4, 40])
            mvl[chx, chy] = pac.tfMVL_tfd2(
                tfds[chx], tfds[chy], [32, 200], [4, 40])
            #pac.tfMVL_tfd(tfd, [32, 200], [4, 40])

        mvls[event.type].append(mvl)
        mvl_2ds[event.type] += mvl_2d

        # break

    epochs = np.array(epochs)
    mvls = np.array(mvls)

    return epochs, mvls, mvl_2ds


def analyse_task(task):
    print(task.participant_id, 'started')

    if not os.path.exists(task.path):
        return

    ds = sio.loadmat(task.path)

    ds['data'] = ds['EEG']

    nbchan = ds['data'][0, 0]['nbchan'][0, 0]  # .dtype
    Fs = ds['data'][0, 0]['srate'][0, 0]
    times = ds['data'][0, 0]['times']
    data = ds['data'][0, 0]['data']

    dtypes = [k for k in ds['data'][0, 0]['event'].dtype.names]
    events = pd.DataFrame([{n: event[n].item() if event[n].size > 0 else None for n in dtypes}
                           for event in ds['data'][0, 0]['event'][0]])

    electrodes = pd.read_csv(os.path.join(
        task['dir'], task['file_formatter'].format('electrodes.tsv')), sep='\t')

    reject = ds['data'][0, 0]['reject']

    b, a = signal.iirnotch(60, 30, Fs)
    data_notch = signal.filtfilt(b, a, data, padlen=150, axis=1)
    # b, a = butter_bandpass(1, 120, Fs, order=9)
    taps_hamming = bandpass_firwin(128, 1, 150, fs=Fs)
    # data_filt = signal.lfilter(taps_hamming, 1, data_notch)
    # data_filt = signal.lfilter(taps_hamming, 1, data_filt[:, ::-1])
    # data_filt = data_filt[:, ::-1]
    data_filt = signal.filtfilt(
        taps_hamming, 1, data_notch, padlen=150, axis=1)

    print(task.participant_id, 'goes for analyse events')

    epochs, mvls, mvl_2ds = analyse_events(events, data_filt, Fs, task)
    with open(os.path.join(task['dir'], task['file_formatter'].format('mvls.npy')), 'wb') as f:
        np.save(f, epochs)
        np.save(f, mvls)
        np.save(f, mvl_2ds)


if __name__ == '__main__':
    BASE_PATH = os.path.dirname(os.path.dirname(  # os.path.abspath(__file__)))
        os.path.abspath('')))
    DS_PATH = os.path.join(BASE_PATH, 'ds003490-master')

    df = pd.read_csv(os.path.join(DS_PATH, 'participants.tsv'), sep="\t")

    tasks_df = create_tasks_df(df)

    # __test__ = 1
    if '__test__' in locals():
        _test_tasks_df(tasks_df, 0)

    from multiprocessing import Pool
    with Pool(4) as p:
        p.map(analyse_task, tasks_df.iloc[4:])

    for task in tasks_df.iloc:
        analyse_task(task)
