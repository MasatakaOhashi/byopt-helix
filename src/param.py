import numpy as np

class ByoptParam(object):
    def __init__(self):
        self.project_dir = r''
        self.inputfile   = ''
        self.referencefile = ''
        self.folder_name = r'class3d'
        self.script_name = r'3dclass.sh'
        self.relion_gpu = "0"
        self.relion_thread=1
        self.relion_iter = 10
        self.relion_K = 1
        self.relion_angpix=3.0
        self.relion_particle_diameter=270.0
        self.relion_helical_outer_diameter=240.0
        self.relion_helical_inner_diameter=10.0
        self.relion_helical_z_percentage = 0.1
        self.relion_helical_nr_asu = 1
        self.relion_sym = 'C1'
        self.relion_tau2_fudge = 4.0
        self.relion_ini_high = 60 
        self.model_type = 'GP_MCMC'
        self.acquisition_type = 'EI_MCMC'
        self.search_type = 0
        self.bayesian_opt_normalize_Y = True
        self.bayesian_opt_acquisition_weight = 2.0
        self.bayesian_opt_acquisition_jitter = 0.01
        #    self.bayesian_opt_lengthscale        = np.array([1.0, 1.0])
        self.bayesian_opt_lengthscale = np.array([0.1, 0.5])
        self.bayesian_opt_ARD = True
        self.bayesian_opt_eps = 1e-3
        self.bayesian_opt_1d_search_lengthscale = 1.0
        self.bayesian_opt_1d_search_ARD = False

        self.bayesian_opt_dp_start = 48.0
        self.bayesian_opt_dp_end = 52.0
        self.bayesian_opt_dp_num = 2
        self.bayesian_opt_dz_start = 8.5
        self.bayesian_opt_dz_end = 9.5
        self.bayesian_opt_dz_num = 2
        self.bayesian_opt_t_num = 2
        self.bayesian_opt_1d_search_margin=0.1
        self.bayesian_opt_1d_search_execute_1 = True
        self.bayesian_opt_1d_search_execute_2 = True
        self.bayesian_opt_1d_search_execute_3 = True
        self.bayesian_opt_1d_search_load_1 = False
        self.bayesian_opt_1d_search_load_2 = False
        self.bayesian_opt_1d_search_load_3 = False
        self.bayesian_opt_1d_search_load_folder1 = ''
        self.bayesian_opt_1d_search_load_folder2 = ''
        self.bayesian_opt_1d_search_load_folder3 = ''
        self.bayesian_opt_2d_search_load = False
        self.bayesian_opt_2d_search_load_folder=''
        self.bayesian_opt_1d_search_first_max_itr = 100
        self.bayesian_opt_1d_search_second_max_itr = 100
        self.bayesian_opt_2d_search_max_itr = 100
