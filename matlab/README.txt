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