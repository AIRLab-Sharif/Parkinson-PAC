import json
import os
import sys

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import simple_pipeline as sp


tasks_df = sp.create_tasks_df('D:\Master sharif\MasterProject\data\parkinsons-oddball')

task = tasks_df.iloc[1]

raw = mne.io.read_raw_eeglab(os.path.join(task['dir'], 'pre_' + task['file_formatter'].format('eeg.set')),
                                 preload=True, verbose=0)
# raw.drop_channels(['X', 'Y', 'Z'])
# raw.drop_channels(['VEOG'])
# raw.set_eeg_reference()

for ch in raw._data:
    ch -= ch.mean()
    ch /= ch.std()
    ch *= 1e-6

# create_elc_file(task)
# montage = mne.channels.read_custom_montage(os.path.join(
#     task.dir, task.file_formatter.format('electrodes.elc')))
montage = mne.channels.read_custom_montage('Standard-10-20-Cap81.locs')
raw.set_montage(montage)

# mne.viz.plot_topomap(raw._data[:, 194000], raw.info, axes=ax,
#                      show=False)

# eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, eog=True)
# freqs = (60, 120, 180, 240)
# raw_notch = raw.copy().notch_filter(freqs=freqs, picks=eeg_picks, verbose=0)
# raw_filtered = raw_notch.copy().filter(l_freq=1, h_freq=150, verbose=0)

events, event_dict = mne.events_from_annotations(raw, verbose=0)
epochs = mne.Epochs(raw, events, event_id=event_dict,
                    tmin=-0.2, tmax=1, preload=True, verbose=0)

selected_events = ['S200', 'S201', 'S202']
erps = {}
for ev in selected_events:
    erps[ev] = epochs[ev].average()



# Plot each of the erp for a channel name
erp_df_s200 = erps['S200'].to_data_frame()
erp_df = erp_df_s200.set_index('time')

erp_df_s201 = erps['S201'].to_data_frame()
erp_df = erp_df_s201.set_index('time')

erp_df_s202 = erps['S202'].to_data_frame()
erp_df = erp_df_s202.set_index('time')

channel_name = 'Fz'
# Plots ERPS
fig, ax = plt.subplots(3,1)

ax[0].plot(erp_df_s200[channel_name])
ax[0].set_title('TargetTone')
ax[1].plot(erp_df_s201[channel_name])
ax[1].set_title('StandardTone')
ax[2].plot(erp_df_s202[channel_name])
ax[2].set_title('NoveltyTone')


## Plots MNNS for a channel
fig.tight_layout(pad=2.0)

fig, ax = plt.subplots(2,1)
Tr = erp_df_s200[channel_name]-erp_df_s201[channel_name]
ax[0].plot(Tr)
ax[0].set_title('MMN=TargetTone-StandardTone')
Nov = erp_df_s202[channel_name]-erp_df_s201[channel_name]
ax[1].plot(Nov)
ax[1].set_title('MMN=NoveltyTone-StandardTone')

fig.tight_layout(pad=2.0)

suffix = 'limited'

plt.show()
# Check calculation for all channel
erp_df_s200=erp_df_s200.drop('time',axis=1)
erp_df_s201=erp_df_s201.drop('time',axis=1)
erp_df_s202=erp_df_s202.drop('time',axis=1)

Tr = erp_df_s200-erp_df_s201
Nov = erp_df_s202-erp_df_s201

fig, ax = plt.subplots(2,1)
ax[0].plot(Tr[channel_name])
ax[0].set_title('MMN=TargetTone-StandardTone')
ax[1].plot(Nov[channel_name])
ax[1].set_title('MMN=NoveltyTone-StandardTone')

fig.tight_layout(pad=2.0)
plt.show()
# Check save and load using csv



# Check save and load using numpy


np.savez_compressed(os.path.join(task['dir'], task['file_formatter'].format(f'mnn_Tr_{suffix}')),
                        Tr)
np.savez_compressed(os.path.join(task['dir'], task['file_formatter'].format(f'mnn_Nov_{suffix}')),
                        Nov)

Tr_loaded = np.load(os.path.join(task['dir'], task['file_formatter'].format(f'mnn_Tr_{suffix}.npz')))
Nov_loaded = np.load(os.path.join(task['dir'], task['file_formatter'].format(f'mnn_Nov_{suffix}.npz')))

# plot load data
Tr_loaded = Tr_loaded[Tr_loaded.files[0]]
Nov_loaded = Nov_loaded[Nov_loaded.files[0]]

fig, ax = plt.subplots(2,1)
ax[0].plot(Tr_loaded[:,1])
ax[0].set_title('MMN=TargetTone-StandardTone')
ax[1].plot(Nov_loaded[:,1])
ax[1].set_title('MMN=NoveltyTone-StandardTone')

fig.tight_layout(pad=2.0)

plt.show()

