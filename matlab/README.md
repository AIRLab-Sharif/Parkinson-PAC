Preprocess
1. Remove x,y,z VEOG channel
2. Highpass filter from 1 to 150Hz
3. Notch Filter at 60 
4. clean_raw_data remove flat and noisy channel see parameters in pipeline.m
5. interpolate the removed channel
6. re-reference to average 
7. ASR, reconstruction of Artifacts
8. re-reference again
9. Make ERP


For runing Preprocess on server:
1. At first, MakeTaskFileDebug should be runned on the server
2. The path to task_track_file.csv should given to the Pipline_run_on_server_preprocess.mlx
3. The path to Pipline_run_on_server_preprocess.mlx should be set as path in matlab
4. Finally, Pipline_run_on_server_preprocess.mlx should be runned

Note: all the above steps are checked locally but not on the server!
