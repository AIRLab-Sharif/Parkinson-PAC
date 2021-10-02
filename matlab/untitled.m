cfg = [];
cfg.xlim = [0.3 0.5];
cfg.zlim = [0 6e-14];
cfg.layout = '/mnt/D/Mastersharif/semister2/neurosience/HW3/eeglab2021.0/plugins/Fieldtrip-lite20210418/template/layout/elec1020.lay';
cfg.parameter = 'individual'; % the default 'avg' is not present in the data
t=1:10;
figure; plot(t,sin(2*pi*t/100));