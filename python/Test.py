import json
import itertools
import os
import sys
import pac
import mne
import numpy as np
import pandas as pd
import simple_pipeline as sp
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12, 9) # (w, h)
gamma = [20, 80]
beta  = [ 4, 16]


def plot_pac(pac, high_freq=gamma, low_freq=beta, ax=None, **kwargs):
    if ax is None:
        fig = plt.figure(figsize=(7, 15))
        ax = fig.subplots()

    im = ax.imshow((pac), origin='lower', interpolation='spline36',  # 'nearest',
                   extent=low_freq + high_freq,
                   #                    aspect='auto', )
                   aspect=np.diff(low_freq) / np.diff(high_freq), **kwargs)

    if ax is None:
        plt.show()

    return im


def mmn_sub_calculation(task):
    raw = mne.io.read_raw_eeglab(os.path.join(task['dir'], 'pre_' + task['file_formatter'].format('eeg.set')),
                                 preload=True, verbose=0)
    for ch in raw._data:
        ch -= ch.mean()
        ch /= ch.std()
        ch *= 1e-6

    montage = mne.channels.read_custom_montage('Standard-10-20-Cap81.locs')
    raw.set_montage(montage)

    events, event_dict = mne.events_from_annotations(raw, verbose=0)
    epochs = mne.Epochs(raw, events, event_id=event_dict,
                        tmin=-0.2, tmax=1, preload=True, verbose=0)

    selected_events = ['S200', 'S201', 'S202']
    erps_1 = {}
    erps = {}
    for ev in selected_events:
        erps_1[ev] = epochs[ev].average()
        erps[ev] = np.mean(epochs[ev]._data,axis=0)

    erp1_df_s200 = erps_1['S200'].to_data_frame()
    erp_df = erp1_df_s200.set_index('time')

    erp1_df_s201 = erps_1['S201'].to_data_frame()
    erp1_df = erp1_df_s201.set_index('time')

    erp1_df_s202 = erps_1['S202'].to_data_frame()
    erp_df = erp1_df_s202.set_index('time')

    erp_df_s200 = pd.DataFrame(erps['S200'],index=epochs.ch_names)
    erp_df_s201 = pd.DataFrame(erps['S201'],index=epochs.ch_names)
    erp_df_s202 = pd.DataFrame(erps['S202'],index=epochs.ch_names)

    # Tr = erp_df_s200 - erp_df_s201
    # Nov = erp_df_s202 - erp_df_s201
    return erp_df_s200, erp_df_s201, erp_df_s202,erp1_df_s200, erp1_df_s201, erp1_df_s202

tasks_df = sp.create_tasks_df('D:\Mastersharif\MasterProject\data\parkinsons-oddball')


temp = tasks_df['pd_drug_type']
index = 3
t = np.linspace(-0.2,1,601)

off_medication = temp[temp==0].index[index]
on_medication = temp[temp==1].index[index]
ctl = temp[temp==2].index[index]

erp_df_s200, erp_df_s201, erp_df_s202, erp1_df_s200, erp1_df_s201, erp1_df_s202 = mmn_sub_calculation(tasks_df.iloc[off_medication])


channel_name_1 = 'C3'
channel_name_2 = 'C3'
range = 1.3
fig, ax = plt.subplots(3,1)
ax[0].plot(erp1_df_s200[channel_name_1]*1e-6)
ax[1].plot(erp_df_s200.loc[channel_name_2])
ax[2].plot(erp1_df_s200[channel_name_1]*1e-6-erp_df_s200.loc[channel_name_2])
# ax[0,0].plot(t,Tr_off[channel_name])
# ax[0,0].set_title('MMN=TargetTone-StandardTone_channel {} off {}'.format(channel_name,tasks_df['participant_id'][off_medication]))
# ax[0,0].set_ylim([-range,range])
# ax[0,1].plot(t,Nov_off[channel_name])
# ax[0,1].set_title('MMN=NoveltyTone-StandardTone_channel {} off {}'.format(channel_name,tasks_df['participant_id'][off_medication]))
# ax[0,1].set_ylim([-range,range])
# ax[1,0].plot(t,Tr_on[channel_name])
# ax[1,0].set_title('MMN=TargetTone-StandardTone_channel {} on {}'.format(channel_name,tasks_df['participant_id'][on_medication]))
# ax[1,0].set_ylim([-range,range])
# ax[1,1].plot(t,Nov_on[channel_name])
# ax[1,1].set_title('MMN=NoveltyTone-StandardTone_channel {} on {}'.format(channel_name,tasks_df['participant_id'][on_medication]))
# ax[1,1].set_ylim([-range,range])
# ax[2,0].plot(t,Tr_ctl[channel_name])
# ax[2,0].set_title('MMN=TargetTone-StandardTone_channel {} ctl {}'.format(channel_name,tasks_df['participant_id'][ctl]))
# ax[2,0].set_ylim([-range,range])
# ax[2,1].plot(t,Nov_ctl[channel_name])
# ax[2,1].set_title('MMN=NoveltyTone-StandardTone_channel {} ctl {}'.format(channel_name,tasks_df['participant_id'][ctl]))
# ax[2,1].set_ylim([-range,range])
fig.tight_layout(pad=2.0)
plt.show()