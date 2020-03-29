# Automatic-Local-Outlier-Factor-Tuning : 
This repository is an implementation of the algorithm presented in the attached reference paper. Thanks to the authors for sharing their ideas. Hope you find it helpful in your projects. 

Check this video out for a demo of the tuning in action!
[![Automatic LOF Tuning Demo](https://i.imgur.com/Io9GbXo.png)](https://youtu.be/kc1rCc_9Vms)
<div align="center">Automatic LOF Tuning Demo</div>

## Requirements
- Python 3
- Celluloid (https://pypi.org/project/celluloid/)
- Scipy, Numpy and Sklearn
- Matplotlib
- Tqdm (for fancy progress bars...)

Download lof_tuner.py to your working directory and import everything.

## Usage
Initialize the tuner with your input 'data' or 'n_samples'. If no data is provided, 'n_samples' of datapoints are sampled from an isotropic gaussian distribution for running the tuner test. The optimal configuration of 'K' and 'C' is one that maximizes 'Z' and is indicated by the frame with grey circles drawn around the datapoints in the attached video) -
 
```
#TEST LOF MODEL TUNER 
#-----------------
#init tuner
tuner = LOF_AutoTuner(n_samples = 500, k_max = 50, c_max = 0.1)

#view grid of hyperparameter values
print(tuner.k_grid) #neighbors
print(tuner.c_grid) #contamination

#test tuner
tuner.test()

#OR TUNE MODEL ON YOUR DATA 
#-----------------
#tune on your custom dataset -> Input 'data', array-like, shape : (n_samples, n_features)
tuner = LOF_AutoTuner(data = data, k_max = 50, c_max = 0.1)

#run tuner
tuner.run()

#TO VISUALISE 
#-----------------
tuner.visualise() #not required when running tuner.test()

#save tuning animation
tuner.animation.save('./lof_tuner_demo.mp4', fps = 2)

```

## Contact
To ask questions or report issues, please open an issue on the [issues tracker](https://github.com/vsatyakumar/automatic-local-outlier-factor-tuning/issues).

## References
Paper : Automatic Hyperparameter Tuning Method for Local Outlier Factor, with Applications to Anomaly Detection (https://arxiv.org/abs/1902.00567)
