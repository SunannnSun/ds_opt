import numpy as np
import json, sys, os

from .util.math_tools import ds_tools, optimization_tools
from .util.data_tools import plot_tools, structures, rearrange_clusters


def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def write_json(data, path):
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=4)


def read_param(data):
    K = data['K']
    M = data['M']
    Priors = np.array(data['Priors'])
    Mu = np.array(data['Mu']).reshape(K, -1)
    Sigma = np.array(data['Sigma']).reshape(K, M, M)
    
    return K, M, Priors, Mu, Sigma


def read_data(data):
    return data["Data"], data["Data_sh"], data["att"], data["x0_all"], data["dt"], data["traj_length"]


class ds_opt:
    def __init__(self, data, Priors, Mu, Sigma):


        # self.js_path = js_path
        self.Data, self.Data_sh, self.att, self.x0_all, self.dt, self.traj_length = read_data(data)
        # self.original_js = read_json(js_path)


        # self.K, self.M, self.Priors, self.Mu, self.Sigma = read_param(self.original_js)
        self.Priors = Priors
        self.Mu = Mu
        self.Sigma = Sigma

        self.K = Priors.shape[0]
        self.M = Mu.shape[1]

        self.ds_struct = rearrange_clusters.rearrange_clusters(self.Priors, self.Mu, self.Sigma, self.att)




    def begin(self):
        """
        A: k x M x M
        b: M x k
        """
        
        self.P_opt = optimization_tools.optimize_P(self.Data_sh)
        self.A_k, self.b_k = optimization_tools.optimize_lpv_ds_from_data(self.Data, self.att, 2, self.ds_struct, self.P_opt, 0)
        

        return self.A_k, self.b_k


    def evaluate(self):
        rmse, e_dot, dwtd = ds_tools.reproduction_metrics(self.Data, self.A_k, self.b_k,
                                                 self.traj_length, self.x0_all, self.ds_struct)
        print("the reproduced RMSE is ", rmse)
        print("the reproduced e_dot is", e_dot)
        print("the reproduced dwtd is ", dwtd)




    def plot(self, *args_):
        Data_dim = self.M
        ds_handle = lambda x_velo: ds_tools.lpv_ds(x_velo, self.ds_struct, self.A_k, self.b_k)
        ds_opt_plot_option = structures.ds_plot_options()
        ds_opt_plot_option.attractor = self.att
        ds_opt_plot_option.x0_all = self.x0_all

        # print(self.x0_all.shape)
        # The plotting function for lyapunov only valid for data with 2 dimension
        if Data_dim == 2:
            # plot_tools.plot_lyapunov_and_derivatives(self.Data, ds_handle, self.att, self.P_opt)
            plot_tools.visualize_DS_2D(self.Data, ds_handle, ds_opt_plot_option)


        # Visualized the reproduced trajectories
        # if len(args_)==0:
        #     ds_opt_plot_option.x0_all = self.x0_all
        #     plot_tools.VisualizeEstimatedDS(self.Data[:Data_dim], ds_handle, ds_opt_plot_option)
        # else:
        #     # ds_opt_plot_option.x0_all = self.x0_all
        #     # ds_opt_plot_option.x0_all = np.hstack((self.x0_all, args_[2]))
        #     ds_opt_plot_option.x0_all = np.hstack(args_[1])
        else:
            plot_tools.visualize_DS_3D(self.Data, ds_handle, ds_opt_plot_option)
        
    

    def step(self, x, dt):
        ds_handle = lambda x_velo: ds_tools.lpv_ds(x_velo, self.ds_struct, self.A_k, self.b_k)
        
        xd = ds_handle(x)

        x_next = x + xd * dt   
        
        return x_next




    def logOut(self, js_path=[]):
        """
        If json file exists, overwrite; if not create a new one

        A: K,M,M
        b: M,K
        """    
        if len(js_path) == 0:
            js_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'output.json')
        self.original_js = read_json(js_path)

        # new_A_k = np.copy(self.A_k)
        # new_b_k = np.copy(self.b_k)
        # new_Sig = np.copy(self.Sigma)    

        # for k in range(self.K):
        #     new_A_k[k, :, :] = new_A_k[k, :, :].T
        #     new_Sig[k] = new_Sig[k].T

        # Mu_trans = self.ds_struct.Mu.T
        # new_A_k = new_A_k.reshape(-1).tolist()

        # self.original_js['Sigma'] = new_Sig.reshape(-1).tolist()
        # self.original_js['Mu'] = Mu_trans.reshape(-1).tolist()
        # self.original_js['Priors'] = self.ds_struct.Priors.tolist()
        # self.original_js['A'] = new_A_k
        # self.original_js['b'] = new_b_k.reshape(-1).tolist()
            
        self.original_js['A'] = self.A_k.ravel().tolist()
        self.original_js['b'] = self.b_k.ravel().tolist()
        self.original_js['attractor']= self.att.ravel().tolist()
        self.original_js['att_all']= self.att.ravel().tolist()
        self.original_js["dt"] = self.dt
        self.original_js["gripper_open"] = 0

        write_json(self.original_js, js_path)

        pass