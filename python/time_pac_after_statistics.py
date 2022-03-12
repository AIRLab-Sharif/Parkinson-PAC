import json
import os
import sys

import mne
import numpy as np
import pandas as pd

from multiprocessing import Pool

import pac
import warnings

warnings.simplefilter("ignore", RuntimeWarning)

BASE_DIR = os.path.abspath('')
N = 10
window = np.ones((N, )) / N * 100
suffix = '_33_36_double_stat'
selected_events = ['S200', 'S201', 'S202']
selected_channels = ['F3', 'F4', 'FC3', 'FC4', 'Fz', 'Pz']

selected_channels_pairs = [
    ('Fz', 'FC3'), ('Fz', 'FC4'),
    ('Pz', 'FC3'), ('Pz', 'FC4'),
    ('Fz', 'F3'), ('Fz', 'F4'),
    ('Pz', 'F3'), ('Pz', 'F4'),
]

gamma = [33, 36]
beta = [5, 8]


def create_tasks_df(ds_path=None):
    if ds_path is None:
        with open('config.json') as f:
            config = json.load(f)
        base_path = config['BASE_PATH']
        ds_path = os.path.join(base_path, 'ds003490-download')

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
            participant_tasks['dir'] = os.path.join(
                ds_path, participant['participant_id'], f'ses-{sess:02}', 'eeg', )
            participant_tasks[
                'file'] = f'{participant["participant_id"]}_ses-{sess:02}_eeg_{participant["participant_id"]}_ses-{sess:02}_task-Rest_eeg.mat'
            participant_tasks['file_formatter'] = f'{participant["participant_id"]}_ses-{sess:02}_task-Rest_{{}}'
            participant_tasks['path'] = os.path.join(
                participant_tasks['dir'], participant_tasks['file'])

            tasks.append(participant_tasks)

    tasks_df = pd.DataFrame(tasks)

    return tasks_df


def _get_details_task(task):
    raw = mne.io.read_raw_eeglab(os.path.join(task['dir'], 'pre_' + task['file_formatter'].format('eeg_double.set')),
                                 preload=True, verbose=0)
    events, event_dict = mne.events_from_annotations(raw, verbose=0)

    global selected_events

    rev_d = {}
    for k, v in event_dict.items():
        rev_d[v] = k

    stims = np.array(list(
        filter(lambda ev: rev_d[ev] in selected_events, events[:, 2])))
    stims -= event_dict[selected_events[0]]

    #
    return {
        'task_num': task.task_num,
        'num_events': stims.shape[0],
        'stim': stims,
    }


def add_details(tasks_df):
    tasks_df['task_num'] = range(tasks_df.shape[0])
    tasks_df['sub_num'] = range(tasks_df.shape[0])

    subs = [0, 0, 0]
    for i, task in enumerate(tasks_df.iloc):
        tasks_df.at[i, 'sub_num'] = subs[task.pd_drug_type]
        subs[task.pd_drug_type] += 1

    with Pool(4) as p:
        task_details = p.map(_get_details_task, tasks_df.iloc)

    df = tasks_df.set_index('task_num').join(pd.DataFrame(task_details))
    df.to_json(f'data/all_tasks.json', orient='records')
    return df


def load_df():
    return pd.read_json(f'data/all_tasks.json')


def check_completed(task, event=None) -> bool:
    json_path = os.path.join(
        task['dir'], task['file_formatter'].format(f'completed{suffix}.json'))
    completed = {}
    if os.path.exists(json_path):
        with open(json_path) as f:
            completed = json.load(f)

    if event is None:
        return completed.get('total', False)
    else:
        return completed.get(event, False)


def update_completed(task, event=None) -> bool:
    json_path = os.path.join(
        task['dir'], task['file_formatter'].format(f'completed{suffix}.json'))
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


def analyse_erps2(erps: dict, task=None):
    mvl_cross_times = {}

    steps = list(range(-200, 1000 + 1, 200))

    groups = ['PD Med Off', 'PD Med On', 'CTL']

    global selected_channels
    # selected_channels_index = [38, 57, 33, 61, 2, 29, 1, 12, 23, 39]

#     print(erps.items())

    for event_type, erp in erps.items():
        mvl_cross_time = np.zeros((10, 10, gamma[1] - gamma[0] + 1,
                                   beta[1] - beta[0] + 1, len(steps) - 1))

        tfds_time = {}

        erp_df = erp.to_data_frame()
        erp_df.time = list(range(-200, 1000 + 1, 2))
        erp_df = erp_df.set_index('time')

        if task is not None:
            print(
                f'{task.participant_id} {groups[task.pd_drug_type]:10} {event_type} tfd started')

        for ch in selected_channels:  # todo
            tfd_time = []

            for i, ts in enumerate(zip(steps[:-1], steps[1:])):
                tstart, tend = ts
#                 print(tstart, tend, min(erp_df.index), max(erp_df.index))
                ind_start = np.where(erp_df.index == tstart)[0][0]
                ind_end = np.where(erp_df.index == tend)[0][0]
                tfd_time.append(pac.rid_rihaczek(
                    erp_df[ch][ind_start:ind_end], int(erp.info['sfreq'])))
            tfds_time[ch] = tfd_time

        # for ch in erp_df: #todo
        #     tfd = pac.rid_rihaczek(erp_df[ch], int(erp.info['sfreq']))
        #     tfds[ch] = tfd

        for chx, chxname in enumerate(selected_channels):
            for chy, chyname in enumerate(selected_channels):
                for i, ts in enumerate(zip(steps[:-1], steps[1:])):
                    tstart, tend = ts
                    ind_start = np.where(erp_df.index == tstart)[0][0]
                    ind_end = np.where(erp_df.index == tend)[0][0]
                    mvl_cross_time[chx, chy, :, :, i] = pac.tfMVL_tfd2_2d(
                        tfds_time[chxname][i], tfds_time[chyname][i], gamma, beta)

        mvl_cross_times[event_type] = mvl_cross_time

    return mvl_cross_times


def analyse_sub2(task):
    raw = mne.io.read_raw_eeglab(os.path.join(task['dir'], 'pre_' + task['file_formatter'].format('eeg_double.set')),
                                 preload=True, verbose=0)
    # raw.drop_channels(['X', 'Y', 'Z'])
    # raw.drop_channels(['VEOG'])
    # raw.set_eeg_reference()

    for ch in raw._data:
        ch -= ch.mean()
        ch /= ch.std()

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

    selected_events = ['S200', 'S201', 'S202']
    event_types = {'S200': 'Target', 'S201': 'Standard', 'S202': 'Novelty'}
    kwargs = {'S200': {'baseline': (0.250, 0.450), 'tmin': 0.250, 'tmax': 1.450},
              'S201': {'baseline': (0.250, 0.450), 'tmin': 0.250, 'tmax': 1.450},
              'S202': {'baseline': (-.200,     0), 'tmin': -.200, 'tmax': 1.000}, }

    erps = {}
    epochs_data = {}
    for ev in selected_events:
        epochs = mne.Epochs(raw, events[events[:, 2] == event_dict[ev]],
                            event_id={ev: event_dict[ev]}, preload=True, verbose=0, **(kwargs[ev]))
        erps[ev] = epochs[ev].average()
        epochs_data[ev] = epochs[ev]._data

    mvl_cross_times = analyse_erps2(erps, task)
    np.savez_compressed(os.path.join(task['dir'], task['file_formatter'].format(f'mvl_cross_times{suffix}')),
                        **mvl_cross_times)
    update_completed(task)
    print(f'{task.participant_id} completed')


def analyse_erps3(erps: dict, task=None):
    global selected_channels
    mvl_cross_times = {}

    steps = list(range(-200, 1000 + 1, 200))

    groups = ['PD Med Off', 'PD Med On', 'CTL']

    # selected_channels_index = [38, 57, 33, 61, 2, 29, 1, 12, 23, 39]

#     print(erps.items())

    for event_type, erp in erps.items():
        mvl_cross_time = np.zeros((len(selected_channels), len(selected_channels),
                                   gamma[1] - gamma[0] + 1,
                                   beta[1] - beta[0] + 1, len(steps) - 1))

        tfds_time = {}

        erp_df = erp.to_data_frame()
        erp_df.time = list(range(-200, 1000 + 1, 2))
        erp_df = erp_df.set_index('time')
        erp_df *= 1e-6

        if task is not None:
            print(
                f'{task.participant_id} {groups[task.pd_drug_type]:10} {event_type} tfd started')

        for ch in selected_channels:  # todo
            tfd_time = []

            for i, ts in enumerate(zip(steps[:-1], steps[1:])):
                tstart, tend = ts
#                 print(tstart, tend, min(erp_df.index), max(erp_df.index))
                ind_start = np.where(erp_df.index == tstart)[0][0]
                ind_end = np.where(erp_df.index == tend)[0][0]
                tfd_time.append(pac.rid_rihaczek(
                    erp_df[ch][ind_start:ind_end], int(erp.info['sfreq'])))
            tfds_time[ch] = tfd_time

        # for ch in erp_df: #todo
        #     tfd = pac.rid_rihaczek(erp_df[ch], int(erp.info['sfreq']))
        #     tfds[ch] = tfd

        for chx, chxname in enumerate(selected_channels):
            for chy, chyname in enumerate(selected_channels):
                for i, ts in enumerate(zip(steps[:-1], steps[1:])):
                    tstart, tend = ts
                    ind_start = np.where(erp_df.index == tstart)[0][0]
                    ind_end = np.where(erp_df.index == tend)[0][0]
                    mvl_cross_time[chx, chy, :, :, i] = pac.tfMVL_tfd2_2d(
                        tfds_time[chxname][i], tfds_time[chyname][i], gamma, beta)

        mvl_cross_times[event_type] = mvl_cross_time

    return mvl_cross_times


def analyse_sub3(task):
    global window, N, selected_events, selected_channels
    raw = mne.io.read_raw_eeglab(os.path.join(task['dir'], 'pre_' + task['file_formatter'].format('eeg_double.set')),
                                 preload=True, verbose=0)
    for ch in raw._data:
        ch -= ch.mean()
        ch /= ch.std()

    montage = mne.channels.read_custom_montage('Standard-10-20-Cap81.locs')
    raw.set_montage(montage)

    #
    events, event_dict = mne.events_from_annotations(raw, verbose=0)

    rev_d = {}
    for k, v in event_dict.items():
        rev_d[v] = k

    in_selection = np.array(list(
        map(lambda ev: rev_d[ev] in selected_events, events[:, 2])))
    conv_sub_stims = np.zeros((3, sum(in_selection) - N + 1))
    selection = np.ones((conv_sub_stims.shape[1], ), dtype=np.bool)
    for k, ev in enumerate(selected_events):
        conv_sub_stims[k] = np.convolve(
            events[in_selection, 2] == event_dict[ev], window, 'valid')
        lt_mean_p_std = conv_sub_stims[k] < (
            conv_sub_stims[k].mean() + conv_sub_stims[k].std())
        gt_mean_m_std = conv_sub_stims[k] > (
            conv_sub_stims[k].mean() - conv_sub_stims[k].std())
        selection = selection & lt_mean_p_std & gt_mean_m_std

    in_std_ev = events[in_selection][N-1:][selection]
    out_std_ev = events[in_selection][N-1:][~selection]

    # selected_events = ['S200', 'S201', 'S202']
    # event_types = {'S200': 'Target', 'S201': 'Standard', 'S202': 'Novelty'}
    kwargs = {'S200': {'baseline': (0.250, 0.450), 'tmin': 0.250, 'tmax': 1.450},
              'S201': {'baseline': (0.250, 0.450), 'tmin': 0.250, 'tmax': 1.450},
              'S202': {'baseline': (-.200,     0), 'tmin': -.200, 'tmax': 1.000}, }

    erps = {}
    epochs_data = {}
    for ev in selected_events:
        in_epochs = mne.Epochs(raw, in_std_ev[in_std_ev[:, 2] == event_dict[ev]],
                               event_id={ev: event_dict[ev]}, preload=True, verbose=0, **(kwargs[ev]))
        erps[f'{ev}_i'] = in_epochs[ev].average()
        epochs_data[f'{ev}_i'] = in_epochs[ev]._data * 1e-6

        out_epochs = mne.Epochs(raw, out_std_ev[out_std_ev[:, 2] == event_dict[ev]],
                                event_id={ev: event_dict[ev]}, preload=True, verbose=0, **(kwargs[ev]))
        erps[f'{ev}_o'] = out_epochs[ev].average()
        epochs_data[f'{ev}_o'] = out_epochs[ev]._data

    mvl_cross_times = analyse_erps3(erps, task)
    np.savez_compressed(os.path.join(task['dir'], task['file_formatter'].format(f'mvl_cross_times{suffix}')),
                        **mvl_cross_times)
    update_completed(task)
    print(f'{task.participant_id} completed')

# # Main


if __name__ == '__main__':
    # tasks_df = create_tasks_df()
    # add_details(tasks_df)

    df = load_df()

    # analyse_sub(tasks_df.iloc[0])

    with Pool(4) as p:
        p.map(analyse_sub3, df.iloc)

    # for task in tasks_df.iloc:
    #     analyse_sub2(task)
    print('finished')
