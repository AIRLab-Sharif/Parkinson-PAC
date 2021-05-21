# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import os



# %%
file_name = os.path.join('/home/kiani/DS/ds003490-download','participants.tsv')
df_participants = pd.read_csv('participants.tsv',delimiter='\t')


path_datasets = ''
all_number_task = 75;
# tasks = {'local_file_path': np.array([]),
#         'raw_data_file_name':np.array([]),
#         'preprocessed_one':np.array([]),'flag1':np.zeros(all_number_task,),
#         'ERP_mat_file':np.array([]),'flag2':np.zeros(all_number_task,),
#         'PAC_dist':np.array([]),'flag3':np.zeros(all_number_task,),
#         'PAC_dist_mean_channel':np.array([]),'flag4':np.zeros(all_number_task,)}
tasks = {'local_file_path': np.array([]),
        'raw_data_file_name':np.array([]),
        'preprocessed_one':np.array([]),
        'ERP_mat_file':np.array([]),
        'PAC_dist':np.array([]),
        'PAC_dist_mean_channel':np.array([])}

for row in df_participants.iterrows():
    if row[1].Group == 'PD':
        file_path = os.path.join('',row[1].participant_id,'ses-{0:0>2}'.format(1),'eeg','')
        tasks['local_file_path'] = np.append( tasks['local_file_path'],file_path)
        file_path = os.path.join('',row[1].participant_id,'ses-{0:0>2}'.format(2),'eeg','')
        tasks['local_file_path'] = np.append( tasks['local_file_path'],file_path)

        tasks['raw_data_file_name'] = np.append(tasks['raw_data_file_name'],row[1].participant_id + '_ses-{0:0>2}_task-Rest_eeg.set'.format(1))
        tasks['raw_data_file_name'] = np.append(tasks['raw_data_file_name'],row[1].participant_id + '_ses-{0:0>2}_task-Rest_eeg.set'.format(2))

        tasks['preprocessed_one'] = np.append(tasks['preprocessed_one'],'pre_' + row[1].participant_id + '_ses-{0:0>2}_task-Rest_eeg.set'.format(1))
        tasks['preprocessed_one'] = np.append(tasks['preprocessed_one'],'pre_' + row[1].participant_id + '_ses-{0:0>2}_task-Rest_eeg.set'.format(2))

        tasks['ERP_mat_file'] = np.append(tasks['ERP_mat_file'],'ERP_' + row[1].participant_id + '_ses-{0:0>2}_task-Rest_eeg.mat'.format(1))
        tasks['ERP_mat_file'] = np.append(tasks['ERP_mat_file'],'ERP_' + row[1].participant_id + '_ses-{0:0>2}_task-Rest_eeg.mat'.format(2))

        tasks['PAC_dist'] = np.append(tasks['PAC_dist'],'PAC_dist_' + row[1].participant_id + '_ses-{0:0>2}_task-Rest_eeg.mat'.format(1))
        tasks['PAC_dist'] = np.append(tasks['PAC_dist'],'PAC_dist_' + row[1].participant_id + '_ses-{0:0>2}_task-Rest_eeg.mat'.format(2))

        tasks['PAC_dist_mean_channel'] = np.append(tasks['PAC_dist_mean_channel'],'PAC_mean_' + row[1].participant_id + '_ses-{0:0>2}_task-Rest_eeg.mat'.format(1))
        tasks['PAC_dist_mean_channel'] = np.append(tasks['PAC_dist_mean_channel'],'PAC_mean_' + row[1].participant_id + '_ses-{0:0>2}_task-Rest_eeg.mat'.format(2))

    else:
        file_path = os.path.join('',row[1].participant_id,'ses-{0:0>2}'.format(1),'eeg','')
        tasks['local_file_path'] = np.append( tasks['local_file_path'],file_path)
        
        tasks['raw_data_file_name'] = np.append(tasks['raw_data_file_name'],row[1].participant_id + '_ses-{0:0>2}_task-Rest_eeg.set'.format(1))

        tasks['preprocessed_one'] = np.append(tasks['preprocessed_one'],'pre_' + row[1].participant_id + '_ses-{0:0>2}_task-Rest_eeg.set'.format(1))
        
        tasks['ERP_mat_file'] = np.append(tasks['ERP_mat_file'],'ERP_' + row[1].participant_id + '_ses-{0:0>2}_task-Rest_eeg.mat'.format(1))
        
        tasks['PAC_dist'] = np.append(tasks['PAC_dist'],'PAC_dist_' + row[1].participant_id + '_ses-{0:0>2}_task-Rest_eeg.mat'.format(1))
        
        tasks['PAC_dist_mean_channel'] = np.append(tasks['PAC_dist_mean_channel'],'PAC_mean_' + row[1].participant_id + '_ses-{0:0>2}_task-Rest_eeg.mat'.format(1))
            

task_df = pd.DataFrame(tasks)
task_df.to_csv('task_track_file.csv')





    



# %%
