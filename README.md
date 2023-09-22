# Dynamical System Optimization in Linear Parameter Varying Formulation

This module is a updated rendition of the optimization implementation from:
https://github.com/penn-figueroa-lab/ds-opt-py

The module has been adapted and integrated into a comprehensive real robot pipeline: Directionality-aware Mixture Model (DAMM) for Dynamical System Learning. Please refer to: https://github.com/SunannnSun/damm_lpvds for complete usage.


---
### Input
To utilize it on your algorithm:

The data should be formulated as a dictionary:
```
data = {
    "Data": Data,         # Dimension x Number of Datapoints
    "Data_sh": Data_sh,   # Shifted attractor to 0 for 'Data' (used in Lyapunov function learning)
    "att": att,           # Dimension x 1
    "x0_all": x0_all,     # Dimension x Number of demonstrated trajectory
                          # This is the start points for all demonstrated trajectories
    "dt": dt,             # Sample time
}
```

For the clustering result, save this dictionary as a json file:
```
json_output = {
    "name": "Clustering Result",
    "K": # Number of clusters,
    "M": # Dimension,
    "Priors": # List of Prior,
    "Mu": # ravel K x M shape Mu to a list,
    "Sigma": #ravel K x M x M Sigma to a list,
}
```
---
### Usage
import DsOpt class and initialize the object:
```
ds_opt = DsOpt(#Your data dictionary, #Your json output directory)
```

Train:
```
ds_opt.begin()
```

Evaluate:
Return the rmse, e_dot, dwtd for learned trajectory
```
ds_opt.evaluate()
```

Plot: Make plots for Lyapunov derivative, value, and reproduced streamlines
```
ds_opt.make_plot()
```
![21931692138054_ pic](https://github.com/HuiTakami/ds_opt_ood/assets/97799818/7207f6f9-a93c-494d-84a3-bb691609160e)

