

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.io as scio
import os
import pac
import mne
import pickle
import itertools



start_time = 300
end_time = 600


start_sample = np.int32((start_time+200)/2)

end_sample = np.int32((end_time+200)/2)


event_types = ['Target','Standard','Novelty']
groups = ['PD_ON','PD_OFF','CTL']

# Pair of frequency
# maybe it needs to average on Pairs of frequency in the interested region
High_fre = 30
Low_fre = 6

# duration for PAC time series
# it might change to a greater value due to possibility of temporal loss information of low frequency activities
duration = 100

with open('..\data\channels_name.pkl', 'rb') as f:
    channels_name = pickle.load(f)

Path_to_load_ERP_collection = '..\data\PAC_electordes.npy'
PAC_on_electrodes = np.load(Path_to_load_ERP_collection)

vmax = PAC_on_electrodes.max()
vmin = PAC_on_electrodes.min()

fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(15, 15))

mne_info = mne.create_info(ch_names=channels_name, sfreq=500., ch_types='eeg')

montage = mne.channels.read_custom_montage('Standard-10-20-Cap81.locs')

for i, j in itertools.product(range(3), range(3)):
    #     data = mvls[i, :, j, :].mean(axis=0).reshape((-1, 1))
    temp = np.expand_dims(PAC_on_electrodes[i, j, :], axis=1)
    PAC_grand_mean = mne.EvokedArray(temp, mne_info)
    PAC_grand_mean.set_montage(montage)
    im, cm = mne.viz.plot_topomap(PAC_grand_mean.data[:, 0],
                                  PAC_grand_mean.info, axes=axs[i, j], show=False,
                                  names=channels_name, show_names=True,
                                  vmin=vmin, vmax=vmax, res=200, cmap='viridis',fontsize='small')

for i in range(3):
    axs[i, 0].set_ylabel(groups[i])
    axs[0, i].set_title(event_types[i])

    axs[-1, i].xaxis.set_visible(True)
    axs[i, 0].yaxis.set_visible(True)

cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])
clb = fig.colorbar(im, cax=cbar_ax)

plt.show()
# nobaseline = True
# if nobaseline == True:
#         fname = f'PAC_grand_mean_dur_{start_time}_{end_time}_pair_fre_{High_fre}_{Low_fre}_nobaseline.png'
# else:
#         fname = f'PAC_grand_mean_dur_{start_time}_{end_time}_pair_fre_{High_fre}_{Low_fre}.png'
# if start_time == -200 and end_time==1000:
#         Path_dir = f'D:\Mastersharif\MasterProject\data\parkinsons-oddball\\results\\topography_figures\PAC-total-grand-mean-high-low_fre-{High_fre}-{Low_fre}'
# else:
#         Path_dir = f'D:\Mastersharif\MasterProject\data\parkinsons-oddball\\results\\topography_figures\PAC-time_window-grand-mean-high-low_fre-{High_fre}-{Low_fre}'

# if not os.path.isdir(Path_dir):
#         os.mkdir(Path_dir)

# file_path = os.path.join(Path_dir,fname)


# plt.savefig(file_path)



# import json
# import itertools
# import os
# import sys
# import pac
# import mne
# import numpy as np
# import pandas as pd
# import simple_pipeline as sp
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# plt.rcParams["figure.figsize"] = (12, 9) # (w, h)
# gamma = [20, 80]
# beta  = [ 4, 16]
#
#
# def plot_pac(pac, high_freq=gamma, low_freq=beta, ax=None, **kwargs):
#     if ax is None:
#         fig = plt.figure(figsize=(7, 15))
#         ax = fig.subplots()
#
#     im = ax.imshow((pac), origin='lower', interpolation='spline36',  # 'nearest',
#                    extent=low_freq + high_freq,
#                    #                    aspect='auto', )
#                    aspect=np.diff(low_freq) / np.diff(high_freq), **kwargs)
#
#     if ax is None:
#         plt.show()
#
#     return im
#
#
# def covariance_sub_calculation(task):
#     raw = mne.io.read_raw_eeglab(os.path.join(task['dir'], 'pre_' + task['file_formatter'].format('eeg.set')),
#                                  preload=True, verbose=0)
#     for ch in raw._data:
#         ch -= ch.mean()
#         ch /= ch.std()
#         # ch *= 1e-6
#
#     montage = mne.channels.read_custom_montage('Standard-10-20-Cap81.locs')
#     raw.set_montage(montage)
#
#     events, event_dict = mne.events_from_annotations(raw, verbose=0)
#     epochs = mne.Epochs(raw, events, event_id=event_dict,
#                         tmin=-0.2, tmax=1, preload=True, verbose=0)
#
#     selected_events = ['S200', 'S201', 'S202']
#     erps_1 = {}
#     covariance = {}
#     pca_reduced = {}
#
#
#     for counter,ev in enumerate(selected_events):
#         trials = epochs[ev]._data[:,1,:]
#         trial_de_mean = trials-np.mean(trials)
#         covariance[ev] = np.matmul(trial_de_mean.T,trial_de_mean)
#
#
#
#
#
#     # Tr = erp_df_s200 - erp_df_s201
#     # Nov = erp_df_s202 - erp_df_s201
#     return covariance
#
# tasks_df = sp.create_tasks_df('D:\Mastersharif\MasterProject\data\parkinsons-oddball')
#
#
# temp = tasks_df['pd_drug_type']
# index = 3
# t = np.linspace(-0.2,1,601)
#
# off_medication = temp[temp==0].index[index]
# on_medication = temp[temp==1].index[index]
# ctl = temp[temp==2].index[index]
#
# covariance_off = covariance_sub_calculation(tasks_df.iloc[off_medication])
# covariance_on = covariance_sub_calculation(tasks_df.iloc[on_medication])
# covariance_ctl = covariance_sub_calculation(tasks_df.iloc[ctl])
#
#
#
# Marker = ['^','*','o']
# fig,ax = plt.subplots(3,3)
#
# fig, ax = plt.subplots(3,3)
# range = 120
# ax[0,0].plot(covariance_off['S200'][0,:])
# ax[0,0].set_title('autocov target off {} {}'.format(0,tasks_df['participant_id'][off_medication]))
# ax[0,0].set_ylim([-range,range])
#
# ax[0,1].plot(covariance_off['S201'][0,:])
# ax[0,1].set_title('StandardTone off {} {}'.format(0,tasks_df['participant_id'][off_medication]))
# ax[0,1].set_ylim([-range,range])
#
# ax[0,2].plot(covariance_off['S202'][0,:])
# ax[0,2].set_title('NoveltyTone off {} {}'.format(0,tasks_df['participant_id'][off_medication]))
# ax[0,2].set_ylim([-range,range])
#
# ax[1,0].plot(covariance_on['S200'][0,:])
# ax[1,0].set_title('TargetTone on {} {}'.format(0,tasks_df['participant_id'][on_medication]))
# ax[1,0].set_ylim([-range,range])
#
# ax[1,1].plot(covariance_on['S201'][0,:])
# ax[1,1].set_title('StandardTone on {} {}'.format(0,tasks_df['participant_id'][on_medication]))
# ax[1,1].set_ylim([-range,range])
#
# ax[1,2].plot(covariance_on['S202'][0,:])
# ax[1,2].set_title('NoveltyTone on {} {}'.format(0,tasks_df['participant_id'][on_medication]))
# ax[1,2].set_ylim([-range,range])
#
# ax[2,0].plot(covariance_ctl['S200'][0,:])
# ax[2,0].set_title('TargetTone ctl {} {}'.format(0,tasks_df['participant_id'][ctl]))
# ax[2,0].set_ylim([-range,range])
#
# ax[2,1].plot(covariance_ctl['S201'][0,:])
# ax[2,1].set_title('StandardTone ctl {} {}'.format(0,tasks_df['participant_id'][ctl]))
# ax[2,1].set_ylim([-range,range])
#
# ax[2,2].plot(covariance_ctl['S202'][0,:])
# ax[2,2].set_title('NoveltyTone ctl {} {}'.format(0,tasks_df['participant_id'][ctl]))
# ax[2,2].set_ylim([-range,range])
#
# # ax = fig.add_subplot(1, 3, 2, projection='3d')
# # ax.scatter(pca_off['S201'][:,0],pca_off['S201'][:,1],pca_off['S201'][:,2])
# # ax.scatter(pca_on['S201'][:,0],pca_on['S201'][:,1],pca_on['S201'][:,2])
# # ax.scatter(pca_ctl['S201'][:,0],pca_ctl['S201'][:,1],pca_ctl['S201'][:,2])
#
#
# fig.tight_layout(pad=2.0)
# plt.show()