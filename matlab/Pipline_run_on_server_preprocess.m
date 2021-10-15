%% Read Task File

clear
%eeglab

T = readtable('..\task_track_files\task_track_file_matlab.csv','Format','%d%s%s%s%s%s%s%s%s%s');

Path = {'/home/kiani/DS/ds003490-download'};
files_preprocessed = fullfile(Path,T{:,2},T{:,4});
flags = zeros(size(files_preprocessed));
for i = 1:size(files_preprocessed, 2)
    flags(i) = exist(files_preprocessed{i}, "file");
end

task_not_completed = 1:75;%find(flag==0)';
temp = size(task_not_completed);

%% Preprocess
%
Highpass_low = 1; % in Hz
Highpass_High = 150; % in Hz
Notch_low = 59.9; % in Hz
Notch_High = 60.1;% in Hz
Notch_order = 16500;

%Path = 'G:\\filmuniversity\\Master sharif\\MasterProject\\data\\parkinsons-oddball';
Path = '/home/kiani/DS/ds003490-download';
channellocationfile = fullfile({Path},T{1,2}{1},'sub-001_ses-01_task-Rest_electrodes.tsv');
channellocationfile = channellocationfile{1};
%delete(gcp('nocreate'))
%parpool(2)
a=T{:,2};
b=T{:,3};
c=T{:,4};
parfor k=1:size(task_not_completed, 2)
    i = task_not_completed(k);
    load_path = fullfile(Path, a{i}); %sprintf('%s/%s',Path,a{i});
    EEG = pop_loadset(b{i},load_path);
    EEG = pop_chanedit(EEG,'load',{channellocationfile ,'filetype','autodetect'});
    EEG = pop_select( EEG, 'nochannel',{'X','Y','Z','VEOG'});
    EEG = pop_eegfiltnew(EEG, 'locutoff',Highpass_low,'hicutoff',Highpass_High);
    EEG = pop_eegfiltnew(EEG, 'locutoff',Notch_low,'hicutoff',Notch_High,'filtorder',Notch_order,'revfilt',1);
    EEG1 = pop_clean_rawdata(EEG, 'FlatlineCriterion',5,'ChannelCriterion',0.8,'LineNoiseCriterion',12,'Highpass','off','BurstCriterion','off','WindowCriterion','off','BurstRejection','off','Distance','Euclidian');
    EEG = pop_interp(EEG1, EEG.chanlocs, 'spherical');
    EEG = pop_reref( EEG, []);
    EEG = pop_clean_rawdata(EEG, 'FlatlineCriterion','off','ChannelCriterion','off','LineNoiseCriterion','off','Highpass','off','BurstCriterion',15,'WindowCriterion','off','BurstRejection','off','Distance','Euclidian');
    EEG = pop_reref( EEG, []);
    EEG = pop_saveset( EEG, 'filename',c{i},'filepath',load_path);
end
