#Automatic Hyperparameter Tuning Method for Local Outlier Factor, with Applications to Anomaly Detection
#https://arxiv.org/pdf/1902.00567v1.pdf

import numpy as np
import tqdm
import matplotlib 
import matplotlib.pyplot as plt
from celluloid import Camera
from collections import defaultdict
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import nct
np.set_printoptions(precision=5)


"""
Algorithm : Tuning algorithm for LOF
1: training data X ∈ R
n×p
2: a grid of feasible values gridc
for contamination c
3: a grid of feasible values gridk
for neighborhood size k
4: for each c ∈ gridc do
5: for each k ∈ gridk do
6: set Mc,k,out to be mean log LOF for the bcnc outliers
7: set Mc,k,in to be mean log LOF for the bcnc inliers
8: set Vc,k,out to be variance of log LOF for the bcnc outliers
9: set Vc,k,in to be variance of log LOF for the bcnc inliers
10: set Tc,k = √
Mc,k,out−Mc,k,in
1
bcnc (Vc,k,out+Vc,k,in)
11: end for
12: set Mc,out to be mean Mc,k,out over k ∈ gridk
13: set Mc,in to be mean Mc,k,in over k ∈ gridk
14: set Vc,out to be mean Vc,k,out over k ∈ gridk
15: set Vc,in to be mean Vc,k,in over k ∈ gridk
16: set ncpc = √
Mc,out−Mc,in
1
bcnc (Vc,out+Vc,in)
17: set dfc = 2bcnc − 2
18: set kc,opt = arg maxk Tc,k
19: end for
20: set copt = arg maxc P(Z < Tc,kc,opt ; d fc
, ncpc), where the random variable Z follows a noncentral
t distribution with dfc degrees of freedom and ncpc noncentrality parameter
"""


class LOF_AutoTuner(object):
    def __init__(self, n_samples = 500, data = None, c_max = 0.1, k_max = 100):
    
        if data is None:
            self.n_samples = n_samples
            print("Input 'data', array-like, shape : (n_samples, n_features).")
        else:
            self.data = data
            self.n_samples = self.data.shape[0]
        
        self.eps = 1e-8
        self.c_max = c_max
        self.k_max = k_max
        self.c_steps = 100
        self.k_grid = np.arange(1,self.k_max + 1) #neighbors
        self.c_grid = np.linspace(0.005, self.c_max, self.c_steps) #contamination
        
    def test(self):
        #sample random gaussian data
        self.data = np.random.standard_normal(size=(self.n_samples,2)) 
        
        #run tuner
        self.run()
        
        #visualize tuning
        self.visualise()
        
    def visualise(self):
        #set inlier threshold. i.e - Any point with Log-LOF score < thresh is considered an inlier.
        thresh = 0.2
        
        fig, ax = plt.subplots(2,2,dpi= 100)
        cam = Camera(fig)
        c_list = [c[3] for c in self.collector]
        k_list = [c[0] for c in self.collector]
        z_list = [c[2] for c in self.collector]


        for i, v in tqdm.tqdm(enumerate(self.collector)):
            Kopt, Topt, Z, contamination = v
            clf = LocalOutlierFactor(n_neighbors=Kopt, 
                                     contamination=contamination)
            clf.fit_predict(self.data)
            X_scores = clf.negative_outlier_factor_
            log_lof = np.log(-X_scores).flatten()

            #viz--->  
            ax[0,1].hist(log_lof, density = True, bins = 100)
            ax[0,1].text(0.05, 0.85, 'Log-LOF :', transform=ax[0,1].transAxes)   

            c_lis = c_list[:i+1]
            k_lis = k_list[:i+1]
            z_lis = z_list[:i+1]
            
            ax[0,0].scatter(c_lis, z_lis, c = 'b', s = 5.)
            ax[0,0].text(0.05, 0.85, 'Z :' + str(Z), c = 'b', transform=ax[0,0].transAxes)  

            ax[1,0].scatter(c_lis, k_lis, c = 'r', s = 5.)
            ax[1,0].text(0.05, 0.85, 'K :' + str(Kopt), c = 'r', transform=ax[1,0].transAxes)     

            #set axes limits
            ax[1,0].set_xlim(0,self.c_max)
            ax[1,0].set_ylim(0,self.k_max)
            
            ax[0,0].set_xlim(0,self.c_max)
            ax[0,0].set_ylim(min(z_list),max(z_list))
            

            if Kopt == self.tuned_params['k'] and contamination == self.tuned_params['c']:
                ax[1,1].scatter(self.data[:, 0], self.data[:, 1], facecolors = 'none', s=1000 * log_lof, edgecolors = 'darkgray')

            ax[1,1].scatter(np.ma.masked_where(log_lof < thresh, self.data[:, 0]), np.ma.masked_where(log_lof < thresh, self.data[:, 1]), c = 'orange', s=5.0)
            ax[1,1].scatter(np.ma.masked_where(log_lof > thresh, self.data[:, 0]), np.ma.masked_where(log_lof > thresh, self.data[:, 1]), c = 'green', s=5.0)
            ax[1,1].text(0.05, 0.85, 'C :' + str(contamination), c = 'darkgray', transform=ax[1,1].transAxes)     
            cam.snap()
        self.animation = cam.animate()
        return
    
    def run(self):
        self.collector = []
        #main op
        for contamination in tqdm.tqdm(self.c_grid):
            samps = int(contamination * self.n_samples)
            if samps < 2:
                continue

            #init running metrics
            running_metrics = defaultdict(list)
            for k in self.k_grid:
                clf = LocalOutlierFactor(n_neighbors=k, contamination=contamination)
                clf.fit_predict(self.data)
                X_scores = np.log(- clf.negative_outlier_factor_)
                t0 = X_scores.argsort()#[::-1]
                top_k = t0[-samps:]
                min_k = t0[:samps]

                x_out = X_scores[top_k]
                x_in = X_scores[min_k]

                mc_out = np.mean(x_out)
                mc_in = np.mean(x_in)
                vc_out = np.var(x_out)
                vc_in = np.var(x_in)
                Tck = (mc_out - mc_in)/np.sqrt((self.eps + ((1/samps)*(vc_out +vc_in))))

                running_metrics['tck'].append(Tck)
                running_metrics['mck_out'].append(mc_out)
                running_metrics['mck_in'].append(mc_in)
                running_metrics['vck_in'].append(vc_in)
                running_metrics['vck_out'].append(vc_out)

            largest_idx = np.array(running_metrics['tck']).argsort()[-1]
            mean_mc_out = np.mean(running_metrics['mck_out'])
            mean_mc_in = np.mean(running_metrics['mck_in'])
            mean_vc_out = np.mean(running_metrics['vck_out'])
            mean_vc_in = np.mean(running_metrics['vck_in'])

            #ncpc - non-centrality parameter
            ncpc = (mean_mc_out - mean_mc_in)/np.sqrt((self.eps + ((1/samps)*(mean_vc_out 
                                                                         + mean_vc_in))))
            #dfc - degrees of freedom
            dfc = (2*samps) - 2

            if dfc <= 0:
                continue

            Z = nct(dfc, ncpc) #non-central t-distribution
            Kopt = self.k_grid[largest_idx]
            Topt = running_metrics['tck'][largest_idx]
            Z = Z.cdf(Topt)
            self.collector.append([Kopt, Topt, Z, contamination])      

        max_cdf = 0.
        self.tuned_params = {}
        for v in self.collector:
            Kopt, Topt, Z, contamination = v
            if Z > max_cdf:
                max_cdf = Z

            if max_cdf == Z:
                self.tuned_params['k'] = Kopt
                self.tuned_params['c'] = contamination

        print("\nTuned LOF Parameters : {}".format(self.tuned_params))
        return
    
