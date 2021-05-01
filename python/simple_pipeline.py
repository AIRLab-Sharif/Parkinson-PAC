#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('config', 'Completer.use_jedi = False')


# In[16]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mne
import json
import sys

# import scipy.io as sio
# from scipy import signal

import pac


# In[2]:


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


def create_elc_file(task):
    with open(os.path.join(task.dir, task.file_formatter.format('electrodes.elc')), 'w') as elc:
        locs = pd.read_csv(os.path.join(
            task.dir, task.file_formatter.format('electrodes.tsv')), sep="\t")
        locs = locs.iloc[:64]
        locs = locs.append(pd.DataFrame([{'name': 'RPA', 'x': 0, 'y': 1, 'z': 0},
                                  {'name': 'LPA', 'x': 0, 'y': -1, 'z': 0},
                                  {'name': 'Nz', 'x': 1, 'y': 0, 'z': 0}, ]))
        elc.write('\n'.join([
            '# ASA electrode file',
            'ReferenceLabel	avg',
            'UnitPosition	mm',
            f'NumberPositions=	{locs.shape[0]}',
            'Positions',
        ]))
        elc.write('\n')
        elc.write(
            '\n'.join(locs.agg(lambda loc: f'{loc.x}\t{loc.y}\t{loc.z}', axis=1)))
        elc.write('\n')
        elc.write('Labels\n')
        elc.write('\n'.join(locs.name))


# In[3]:



def create_tasks_df(ds_path=None):
    if ds_path is None:
        with open('config.json') as f:
            config = json.load(f)
        BASE_PATH = config['BASE_PATH']
        ds_path = os.path.join(BASE_PATH, 'ds003490-download')
    
    df = pd.read_csv(os.path.join(ds_path, 'participants.tsv'), sep="\t")
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
            participant_tasks['dir'] = os.path.join(ds_path, participant['participant_id'], f'ses-{sess:02}', 'eeg',)
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


# # Create Task list in `tasks_df`

# In[17]:


def check_completed(task, event=None) -> bool:
    json_path = os.path.join(task['dir'], task['file_formatter'].format('completed.json'))
    completed = {}
    if os.path.exists(json_path):
        with open(json_path) as f:
            completed = json.load(f)
    
    if event is None:
        return completed.get('total', False)
    else:
        return completed.get(event, False)


def update_completed(task, event=None) -> bool:
    json_path = os.path.join(task['dir'], task['file_formatter'].format('completed.json'))
    completed = {}
    if os.path.exists(json_path):
        with open(json_path) as f:
            completed = json.load(f)

    if event is None:
        completed['total'] = True
    else:
        completed[event] = True
        
    with open(json_path, 'w') as f:
        json.dump(completed, f)


# In[18]:


def analyse_erps(erps: dict, task=None):
    mvls = {}
    mvl_2ds = {}

    for event_type, erp in erps.items():
        mvl_2d = np.zeros(
            (erp.info['nchan'], erp.info['nchan'], 200-32+1, 40-4+1))
        mvl = np.zeros((erp.info['nchan'], erp.info['nchan'], ))
        tfds = {}

        erp_df = erp.to_data_frame()
        erp_df = erp_df.set_index('time')

        if task is not None:
            print(f'{task.participant_id} {event_type} tfd started')
        
        for ch in erp_df:
            tfd = pac.rid_rihaczek(erp_df[ch], int(erp.info['sfreq']))
            tfds[ch] = tfd

        for chx, chxname in enumerate(erp_df):
            chy = chx
            chyname = chxname
            for chy, chyname in enumerate(erp_df):
            # print(chxname, chyname)

            # todo:
            # if(check_completed(task, f'{event_type}_{chxname}_{chyname}')):
            #     continue

                mvl_2d[chx, chy] = pac.tfMVL_tfd2_2d(
                    tfds[chxname], tfds[chyname], [32, 200], [4, 40])
                mvl[chx, chy] = pac.tfMVL_tfd2(
                    tfds[chxname], tfds[chyname], [32, 200], [4, 40])
            
        mvls[event_type] = mvl
        mvl_2ds[event_type] = mvl_2d

    return mvls, mvl_2ds


# In[19]:


def analyse_sub(task):
    if(check_completed(task)):
        return

    raw = mne.io.read_raw_eeglab(os.path.join(task['dir'], task['file_formatter'].format('eeg.set')),
                                 preload=True, verbose=0)
    raw.set_eeg_reference()
    raw.drop_channels(['X', 'Y', 'Z'])

    create_elc_file(task)
    montage = mne.channels.read_custom_montage(os.path.join(
        task.dir, task.file_formatter.format('electrodes.elc')))
    raw.set_montage(montage)

    eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=True)
    freqs = (60, 120, 180, 240)
    raw_notch = raw.copy().notch_filter(freqs=freqs, picks=eeg_picks, verbose=0)
    raw_filtered = raw_notch.copy().filter(l_freq=1, h_freq=150, verbose=0)

    events, event_dict = mne.events_from_annotations(raw_filtered, verbose=0)
    epochs = mne.Epochs(raw_filtered, events, event_id=event_dict,
                        tmin=-0.2, tmax=1, preload=True, verbose=0)

    selected_events = ['S200', 'S201', 'S202']
    erps = {}
    for ev in selected_events:
        erps[ev] = epochs[ev].average()
        
    mvls, mvl_2ds = analyse_erps(erps, task)
    np.savez_compressed(os.path.join(task['dir'], task['file_formatter'].format('mvls')),
                        **mvls)
    np.savez_compressed(os.path.join(task['dir'], task['file_formatter'].format('mvl_2ds')),
                        **mvl_2ds)
    update_completed(task)
    print(f'{task.participant_id} completed')


# In[9]:


if __name__ == '__main__':
    tasks_df = create_tasks_df(DS_PATH)

    # __test__ = 1
    if '__test__' in locals():
        _test_tasks_df(tasks_df, 0)

    # analyse_sub(tasks_df.iloc[0])
    
    if len(sys.argv) <= 1:
        from multiprocessing import Pool
        with Pool(4) as p:
            p.map(analyse_sub, tasks_df.iloc)
    else:
        for task in tasks_df.iloc:
            analyse_sub(task)


