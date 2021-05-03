%% Pipline
% Preproces
% Make ERP
% PAC
% Plot comodulogram
% P value
%% Preprocess
clear all
Highpass_low = 1; % in Hz
Highpass_High = 150; % in Hz
Notch_low = 59.9; % in Hz
Notch_High = 60.1;% in Hz
Notch_order = 16500;

sub_num = [30,31,80];
temp = size(sub_num);
filename = cell(temp(1),temp(2));
filepath = cell(temp(1),temp(2));
channellocationfile = cell(temp(1),temp(2));

for i = 1:temp(2)
    filename{1,i} = sprintf('sub-%03d_ses-%02d_task-Rest_eeg.set',floor(sub_num(i)/2),mod(sub_num(i),2)+1);
    filepath{1,i} = sprintf('G:\\filmuniversity\\Master sharif\\Master Project\\data\\parkinsons-oddball\\sub-%03d\\ses-%02d\\eeg',floor(sub_num(i)/2),mod(sub_num(i),2)+1);
    channellocationfile{1,i} = sprintf('G:\\filmuniversity\\Master sharif\\Master Project\\data\\parkinsons-oddball\\sub-%03d\\ses-%02d\\eeg\\sub-%03d_ses-%02d_task-Rest_electrodes.tsv',floor(sub_num(i)/2),mod(sub_num(i),2)+1,floor(sub_num(i)/2),mod(sub_num(i),2)+1);
end

% filname1 = 'sub-001_ses-01_task-Rest_eeg.set';
% filname2 = 'sub-001_ses-02_task-Rest_eeg.set';
% filname3 = 'sub-028_ses-01_task-Rest_eeg.set';
% 
% filepath1 = 'G:\filmuniversity\Master sharif\Master Project\data\parkinsons-oddball\sub-001\ses-01\eeg';
% filepath2 = 'G:\filmuniversity\Master sharif\Master Project\data\parkinsons-oddball\sub-001\ses-02\eeg';
% filepath3 = 'G:\filmuniversity\Master sharif\Master Project\data\parkinsons-oddball\sub-028\ses-01\eeg';
% 
% channellocationfile1 = 'G:\filmuniversity\Master sharif\Master Project\data\parkinsons-oddball\sub-001\ses-01\eeg\sub-001_ses-01_task-Rest_electrodes.tsv';
% channellocationfile2 = 'G:\filmuniversity\Master sharif\Master Project\data\parkinsons-oddball\sub-001\ses-02\eeg\sub-001_ses-02_task-Rest_electrodes.tsv';
% channellocationfile3 = 'G:\filmuniversity\Master sharif\Master Project\data\parkinsons-oddball\sub-028\ses-01\eeg\sub-028_ses-01_task-Rest_electrodes.tsv';
% 
% channellocationfile = {channellocationfile1,channellocationfile2,channellocationfile3};
% filepath = {filepath1,filepath2,filepath3};
% filename = {filname1,filname2,filname3};

eeglab
for i=1:3
    EEG = pop_loadset(filename{i},filepath{i});
    EEG = pop_chanedit(EEG,'load',{channellocationfile{i} ,'filetype','autodetect'});
    EEG = pop_select( EEG, 'nochannel',{'X','Y','Z','VEOG'});
    EEG = pop_eegfiltnew(EEG, 'locutoff',Highpass_low,'hicutoff',Highpass_High);
%   figure; topoplot([],EEG.chanlocs, 'style', 'blank',  'electrodes', 'labelpoint', 'chaninfo', EEG.chaninfo);
    EEG1 = pop_clean_rawdata(EEG, 'FlatlineCriterion',5,'ChannelCriterion',0.8,'LineNoiseCriterion',12,'Highpass','off','BurstCriterion','off','WindowCriterion','off','BurstRejection','off','Distance','Euclidian');
    EEG = pop_interp(EEG1, EEG.chanlocs, 'spherical');
    EEG = pop_reref( EEG, []);
    EEG.setname=[filename{i},'basr'];
    pop_eegplot( EEG, 1, 1, 1);
    EEG = pop_clean_rawdata(EEG, 'FlatlineCriterion','off','ChannelCriterion','off','LineNoiseCriterion','off','Highpass','off','BurstCriterion',15,'WindowCriterion','off','BurstRejection','off','Distance','Euclidian');
    EEG.setname=[filename{i},'afterasr'];
    pop_eegplot( EEG, 1, 1, 1);
    EEG = pop_reref( EEG, []);
    EEG = pop_saveset( EEG, 'filename',filename{i},'filepath','G:\filmuniversity\Master sharif\Master Project\data\parkinsons-oddball\preprocess_check_data');
end

%% Make ERP
clear all

sub_num = [30,31,80];
temp = size(sub_num);
filename = cell(temp(1),temp(2));
filepath = cell(temp(1),temp(2));
channellocationfile = cell(temp(1),temp(2));

for i = 1:temp(2)
    filename{1,i} = sprintf('sub-%03d_ses-%02d_task-Rest_eeg.set',floor(sub_num(i)/2),mod(sub_num(i),2)+1);
%   filepath{1,i} = sprintf('G:\\filmuniversity\\Master sharif\\Master Project\\data\\parkinsons-oddball\\sub-%03d\\ses-%02d\\eeg',floor(sub_num(i)/2),mod(sub_num(i),2)+1);
%   channellocationfile{1,i} = sprintf('G:\\filmuniversity\\Master sharif\\Master Project\\data\\parkinsons-oddball\\sub-%03d\\ses-%02d\\eeg\\sub-%03d_ses-%02d_task-Rest_electrodes.tsv',floor(sub_num(i)/2),mod(sub_num(i),2)+1,floor(sub_num(i)/2),mod(sub_num(i),2)+1);
end

index_start = 122;

filepath1 = 'G:\filmuniversity\Master sharif\Master Project\data\parkinsons-oddball\preprocess_check_data';

number_samples = 1.2*500+1;
epochs = zeros(3,63,3,number_samples);

for i=1:3
    EEG = pop_loadset(filename{i},filepath1);
    temp1 = size(EEG.event);
    for j=1:63
        counter1 = 0; counter2= 0; counter3 = 0;
        for m=index_start:temp1(2)
            if strcmp(EEG.event(m).type,'S201')
                temp = zeros(1,number_samples);
                temp(:) = squeeze(epochs(i,j,1,:));
                epochs(i,j,1,:) = (temp*counter1 + EEG.data(j,EEG.event(m).latency-0.2*500:EEG.event(m).latency+500))/(counter1+1);
                counter1 = counter1 + 1;
            elseif strcmp(EEG.event(m).type,'S202') 
                temp = zeros(1,number_samples);
                temp(:) = squeeze(epochs(i,j,2,:));
                epochs(i,j,2,:) = (temp*counter2 + EEG.data(j,EEG.event(m).latency-0.2*500:EEG.event(m).latency+500))/(counter2+1);
                counter2 = counter2 + 1;
            elseif strcmp(EEG.event(m).type,'S200') 
                temp = zeros(1,number_samples);
                temp(:) = squeeze(epochs(i,j,3,:));
                epochs(i,j,3,:) = (temp*counter3 + EEG.data(j,EEG.event(m).latency-0.2*500:EEG.event(m).latency+500))/(counter3+1);
                counter3 = counter3 + 1;
            end    
        end   
    end    
end     
save('erp1.mat','epochs')
%% PAC
clear all

Amplitude_Frequency_Range = 50:4:150;
Phase_Frequency = 4:1:13;
temp1 = size(Amplitude_Frequency_Range);
temp2 = size(Phase_Frequency);
load('erp1.mat')
temp = size(epochs);
pacs=zeros(3,63,3,temp1(2),temp2(2));
% counter=0;
for a=1:temp(1)
    for b=3:3:63
        for c=1:3
            for i=1:temp1(2)
                for j = 1:temp2(2)
                    pacs(a,b,c,i,j) = tfMVL(epochs(a,b,c,:),Amplitude_Frequency_Range(i),Phase_Frequency(j),500);
                end
            end
            fprintf('person %d channel %d stim %d  done\n',a,b,c)
%           counter=counter + 1;
        end    
    end
end 
frequency_details.high_start_fre = Amplitude_Frequency_Range(1);
frequency_details.high_end_fre = Amplitude_Frequency_Range(end);
frequency_details.step_high = 4;
frequency_details.low_start_fre = Phase_Frequency(1);
frequency_details.low_end_fre = Phase_Frequency(end);
frequency_details.step_low = 1;

save('pacs1.mat','pacs','frequency_details')

pacs = mean(pacs,2);
%% Plot comodulogram
clear all
close all
load('pacs1.mat')
pacs = mean(pacs,2);
% pacs = mean(pacs,2);
high_range = [frequency_details.high_start_fre,frequency_details.high_end_fre];
low_range =  [frequency_details.low_start_fre,frequency_details.low_end_fre];
sub_num = [30,31,80];
subplot(3,3,1)
plot_comodulogram(squeeze(pacs(1,1,1,:,:)),high_range,low_range,sprintf('sub%d patient off medication S201 Standard Tone',floor(sub_num(1)/2)),frequency_details.step_high,frequency_details.step_low)
subplot(3,3,2)
plot_comodulogram(squeeze(pacs(2,1,1,:,:)),high_range,low_range,sprintf('sub%d patient on medication S201 Standard Tone ',floor(sub_num(2)/2)),frequency_details.step_high,frequency_details.step_low)
subplot(3,3,3)
plot_comodulogram(squeeze(pacs(3,1,1,:,:)),high_range,low_range,sprintf('sub%d healthy S201 Standard Tone ',floor(sub_num(3)/2)),frequency_details.step_high,frequency_details.step_low)
subplot(3,3,4)
plot_comodulogram(squeeze(pacs(1,1,2,:,:)),high_range,low_range,sprintf('sub%d patient off medication S202 Novel Tone ',floor(sub_num(1)/2)),frequency_details.step_high,frequency_details.step_low)
subplot(3,3,5)
plot_comodulogram(squeeze(pacs(2,1,2,:,:)),high_range,low_range,sprintf('sub%d patient on medication S202 Novel Tone ',floor(sub_num(2)/2)),frequency_details.step_high,frequency_details.step_low)
subplot(3,3,6)
plot_comodulogram(squeeze(pacs(3,1,2,:,:)),high_range,low_range,sprintf('sub%d healthy S202 Novel Tone ',floor(sub_num(3)/2)),frequency_details.step_high,frequency_details.step_low)
subplot(3,3,7)
plot_comodulogram(squeeze(pacs(1,1,3,:,:)),high_range,low_range,sprintf('sub%d patient off medication S200 Target Tone ',floor(sub_num(1)/2)),frequency_details.step_high,frequency_details.step_low)
subplot(3,3,8)
plot_comodulogram(squeeze(pacs(2,1,3,:,:)),high_range,low_range,sprintf('sub%d patient on medication S200 Target Tone ',floor(sub_num(2)/2)),frequency_details.step_high,frequency_details.step_low)
subplot(3,3,9)
plot_comodulogram(squeeze(pacs(3,1,3,:,:)),high_range,low_range,sprintf('sub%d healthy S200 Target Tone ',floor(sub_num(3)/2)),frequency_details.step_high,frequency_details.step_low)

%%
clear all
close all
load('channel.mat')
load('pacs1.mat')
for i=3:3:63
    % pacs = mean(pacs,2);
    figure;
    high_range = [frequency_details.high_start_fre,frequency_details.high_end_fre];
    low_range =  [frequency_details.low_start_fre,frequency_details.low_end_fre];
    sub_num = [30,31,80];
    subplot(3,3,1)
    plot_comodulogram(squeeze(pacs(1,i,1,:,:)),high_range,low_range,sprintf('sub%d patient off medication S201 Standard Tone channel %s',floor(sub_num(1)/2),a(1,i).labels),frequency_details.step_high,frequency_details.step_low)
    subplot(3,3,2)
    plot_comodulogram(squeeze(pacs(2,i,1,:,:)),high_range,low_range,sprintf('sub%d patient on medication S201 Standard Tone channel %s',floor(sub_num(2)/2),a(1,i).labels),frequency_details.step_high,frequency_details.step_low)
    subplot(3,3,3)
    plot_comodulogram(squeeze(pacs(3,i,1,:,:)),high_range,low_range,sprintf('sub%d healthy S201 Standard Tone channel %s',floor(sub_num(3)/2),a(1,i).labels),frequency_details.step_high,frequency_details.step_low)
    subplot(3,3,4)
    plot_comodulogram(squeeze(pacs(1,i,2,:,:)),high_range,low_range,sprintf('sub%d patient off medication S202 Novel Tone channel %s',floor(sub_num(1)/2),a(1,i).labels),frequency_details.step_high,frequency_details.step_low)
    subplot(3,3,5)
    plot_comodulogram(squeeze(pacs(2,i,2,:,:)),high_range,low_range,sprintf('sub%d patient on medication S202 Novel Tone channel %s',floor(sub_num(2)/2),a(1,i).labels),frequency_details.step_high,frequency_details.step_low)
    subplot(3,3,6)
    plot_comodulogram(squeeze(pacs(3,i,2,:,:)),high_range,low_range,sprintf('sub%d healthy S202 Novel Tone channel %s',floor(sub_num(3)/2),a(1,i).labels),frequency_details.step_high,frequency_details.step_low)
    subplot(3,3,7)
    plot_comodulogram(squeeze(pacs(1,i,3,:,:)),high_range,low_range,sprintf('sub%d patient off medication S200 Target Tone channel %s',floor(sub_num(1)/2),a(1,i).labels),frequency_details.step_high,frequency_details.step_low)
    subplot(3,3,8)
    plot_comodulogram(squeeze(pacs(2,i,3,:,:)),high_range,low_range,sprintf('sub%d patient on medication S200 Target Tone channel %s',floor(sub_num(2)/2),a(1,i).labels),frequency_details.step_high,frequency_details.step_low)
    subplot(3,3,9)
    plot_comodulogram(squeeze(pacs(3,i,3,:,:)),high_range,low_range,sprintf('sub%d healthy S200 Target Tone channel %s',floor(sub_num(3)/2),a(1,i).labels),frequency_details.step_high,frequency_details.step_low)
end




