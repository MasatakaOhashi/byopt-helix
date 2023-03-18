import numpy as np
import os
import sys
import subprocess
import re
import glob
import GPy
import GPyOpt
import shutil
import time
from config import Reader,ByOptInitReader,ByOptMinWriter,ByOptMinReader

SAVE_AQ_CV=False

def get_application_folder():
    appli_path = sys.argv[0]
    appli_dir,filename = os.path.split(appli_path)
    return appli_dir

def get_model_files(data_folder):
    """
    get model files in data_folder
    :param data_folder:
    :return:
    """
    return sorted(glob.glob(os.path.join(data_folder, 'run_it[0-9][0-9][0-9]_model.star')))
def get_matched_line_from_file(log_file_full_path,phrase):
    """
    get matched line from file using regular representation
    :param log_file_full_path:
    :param phrase:
    :return:
    """
    m=[]
    with open(log_file_full_path) as f:
        for line in f:
            if re.match('%s'%phrase, line):
                m.append(line)
    return m
def get_folder_list(folder):
    return sorted([f for f in glob.glob(os.path.join(folder,'*')) if os.path.isdir(f)])
def split_jobindex(bn):
    m = re.search(r'(.*)([0-9]{3})', bn)
    if m:
        index = m.group(2)
        return int(index)
    else:
        return 0

class BYoptHelix(object):
    def __init__(self):
        self.project_folder= r'/home/ohashi/data/testdata_20181128_031_noise_5_ctf'
        self.inputfile = ''
        self.referencefile = ''
        self.folder_name = r'class3d'
        self.script_name = '3dclass.sh'
        self.relion_iter = 10
        self.relion_gpu=''
        self.relion_K = 1
        self.relion_angpix=3.0
        self.relion_particle_diameter=270.0
        self.relion_helical_outer_diameter=240.0
        self.relion_helical_inner_diameter=10.0
        self.relion_helical_z_percentage = 0.1
        self.relion_helical_nr_asu = 1
        self.relion_sym = 'C1'
        self.relion_tau2_fudge = 4.0
        self.relion_ini_high=60
        self.output_dir = os.path.join(self.project_folder, self.folder_name)
        self.script_path = os.path.join(self.project_folder,self.script_name)
        self.bayesian_opt_1d_search_margin = 1
        self.dp = 0
        self.min_dp = 0
        self.min_dz = 0
        self.min_t  = 0
        self.initX = None
        self.initY = None
        self.linear_position=np.zeros((2,2))
        self.prefix=''
        self.ResultList=[]
    def set_param(self,prm):
        """
        get parameters from struct prm
        :param prm:
        :return:
        """
        self.project_folder               = prm.project_dir
        self.inputfile                    = prm.inputfile
        self.referencefile                = prm.referencefile
        self.folder_name                  = prm.folder_name
        self.script_name                  = prm.script_name
        self.relion_angpix                = prm.relion_angpix                
        self.relion_particle_diameter     = prm.relion_particle_diameter     
        self.relion_helical_outer_diameter= prm.relion_helical_outer_diameter
        self.relion_helical_inner_diameter= prm.relion_helical_inner_diameter
        self.relion_helical_z_percentage  = prm.relion_helical_z_percentage
        self.relion_helical_nr_asu        = prm.relion_helical_nr_asu
        self.relion_tau2_fudge            = prm.relion_tau2_fudge
        self.relion_sym                   = prm.relion_sym
        self.relion_K                     = prm.relion_K
        self.relion_iter                  = prm.relion_iter
        self.relion_gpu                   = prm.relion_gpu
        self.relion_thread                = prm.relion_thread
        self.relion_ini_high              = prm.relion_ini_high
        self.output_dir                   = os.path.join(self.project_folder, self.folder_name)
        self.script_path                  = os.path.join(self.project_folder,self.script_name)
        self.bayesian_opt_1d_search_margin = prm.bayesian_opt_1d_search_margin
    def set_prefix(self,prefix):
        self.prefix = prefix
    def get_prefix(self):
        return self.prefix
    def clear_result_list(self):
        self.ResultList=[]
    def get_result_filepath(self,prm,num=0):
        if num == 1:
            return os.path.join(prm.bayesian_opt_1d_search_load_folder1,'byopt_list_{}.txt'.format(num))
        elif num == 2:
            return os.path.join(prm.bayesian_opt_1d_search_load_folder2, 'byopt_list_{}.txt'.format(num))
        elif num == 3:
            return os.path.join(prm.bayesian_opt_1d_search_load_folder3, 'byopt_list_{}.txt'.format(num))
        else:
            return os.path.join(prm.bayesian_opt_2d_search_load_folder,'byopt_list.txt')
    def get_min_filepath(self,prm,num=0):
        if num == 1:
            return os.path.join(prm.bayesian_opt_1d_search_load_folder1,'byopt_min_{}.txt'.format(num))
        elif num == 2:
            return os.path.join(prm.bayesian_opt_1d_search_load_folder2, 'byopt_min_{}.txt'.format(num))
        elif num == 3:
            return os.path.join(prm.bayesian_opt_1d_search_load_folder3, 'byopt_min_{}.txt'.format(num))
        else:
            return os.path.join(prm.bayesian_opt_2d_search_load_folder,'byopt_min.txt')

    def load_byopt_list(self,prm,num=0):
        filepath = self.get_result_filepath(prm,num)
        reader = ByOptInitReader()
        reader.open(filepath)
        ret = reader.read()
        result = np.array(ret).T

        if num == 1 or num == 2:
            X = result[:,1]
            Y = result[:,2]
        elif num == 3:
            X = result[:,3:]
            Y = result[:,2]
        else:
            X = result[:,0:2]
            Y = result[:,2]
        self.initX = X if X.ndim > 1 else X[:,np.newaxis]
        self.initY = Y[:,np.newaxis]
        self.ResultList=result.tolist()

    def load_byopt_min(self,prm,num=0):
        filepath = self.get_min_filepath(prm,num)
        reader = ByOptMinReader()
        data = reader.read(filepath)
        self.linear_position=data
    def save_byopt_min(self,filepath):
        writer = ByOptMinWriter()
        writer.write(self.linear_position,filepath)

    def clear_byopt_list(self):
        self.initX = None
        self.initY = None
        self.ResultList=[]

    def convert_t2dpdz(self,t):
        """

        :param t:
        :return: [dp, dz]
        """
        p1 = self.linear_position[0, :]
        p2 = self.linear_position[1, :]
        dp = p1[0] + (p2[0] - p1[0]) * t
        dz = p1[1] + (p2[1] - p1[1]) * t
        return dp, dz
    def convert_ts2dpdz(self,t,s):
        """

        :param t:
        :param s:
        :return: [dp, dz]
        """
        p1 = self.linear_position[0, :]
        p2 = self.linear_position[1, :]
        p2p1=p2-p1
        p2p1norm=np.sqrt(np.sum(p2p1**2))
        cs = p2p1[0]/p2p1norm
        sn = p2p1[1]/p2p1norm
        dp=t*cs - s*sn + p1[0]
        dz=t*sn + s*cs + p1[1]
        return dp, dz
    def get_t_range(self,pos,dp_start,dp_end):
        p1 = pos[0,:]
        p2 = pos[1,:]
        tmin=(dp_start-p1[0])/(p2[0]-p1[0])
        tmax=(dp_end-p1[0])/(p2[0]-p1[0])
        return tmin,tmax
    def get_ts_range(self,pos,dp_start,dp_end,dz_start,dz_end):
        p1 = pos[0,:]
        p2 = pos[1,:]
        p2p1=p2-p1
        p2p1norm = np.sqrt(np.sum(p2p1**2))

        tmin=(dp_start-p1[0])/p2p1[0]*p2p1norm
        tmax=(dp_end-p1[0])/p2p1[0]*p2p1norm
        smin= dz_start
        smax= dz_end

        return tmin,tmax,smin,smax
    def get_fixed_dp(self):
        return self.dp
    def set_fixed_dp(self,dp):
        self.dp = dp
    def set_min_dp_and_dz(self,min_dp,min_dz):
        self.min_dp = min_dp
        self.min_dz = min_dz
    def get_min_dp_and_dz(self):
        return self.min_dp, self.min_dz
    def save_result_log(self,filepath):
        if len(self.ResultList) == 0:
            return
        numOfData = len(self.ResultList[0])

        with open(filepath,'w') as f:
            if numOfData == 3:
                print('dp,dz,lb',file=f)
            elif numOfData == 4:
                print('dp,dz,lb,t',file=f)
            elif numOfData == 5:
                print('dp,dz,lb,t,s',file=f)

            for byv in self.ResultList:
                if len(byv) == 3:
                    print('{0:.8e},{1:.8e},{2:.8e}'.format(byv[0],byv[1],byv[2]),file=f)
                elif len(byv)==4:
                    print('{0:.8e},{1:.8e},{2:.8e},{3:.8e}'.format(byv[0],byv[1],byv[2],byv[3]),file=f)
                elif len(byv)==5:
                    print('{0:.8e},{1:.8e},{2:.8e},{3:.8e},{4:.8e}'.format(byv[0],byv[1],byv[2],byv[3],byv[4]),file=f)


    def get_loglikelihood_1d_first(self, input):
        X = input
        if X.ndim == 2:
            dz = X[0, 0]
        elif X.ndim == 1:
            dz = X[0]
        else:
            dz = X
        dp = self.get_fixed_dp()

        _, _, ll = self.get_loglikelihood(np.array([[dp, dz]]))
        self.ResultList.append([dp, dz, ll])
        return ll
    def get_loglikelihood_1d_second(self,input):
        X = input
        if X.ndim == 2:
            t = X[0, 0]
        elif X.ndim == 1:
            t = X[0]
        else:
            t = X
        dp, dz = self.convert_t2dpdz(t)

        _, _, ll = self.get_loglikelihood(np.array([[dp, dz]]))
        self.ResultList.append([dp,dz,ll,t])
        return ll
    def get_loglikelihood_1d_second_2d(self,input):
        X = input
        if X.ndim == 2:
            t = X[0, 0]
            s = X[0, 1]
        elif X.ndim == 1:
            t = X[0]
            s = X[1]
        else:
            raise RuntimeError('unsupported ndim {}'.format(X.ndim))

        dp, dz = self.convert_ts2dpdz(t,s* self.bayesian_opt_1d_search_margin)

        _, _, ll = self.get_loglikelihood(np.array([[dp, dz]]))
        self.ResultList.append([dp,dz,ll,t,s])
        return ll

    def get_loglikelihood_2d(self,input):
        dp, dz, ll = self.get_loglikelihood(input)
        self.ResultList.append([dp,dz,ll])
        return ll

    def get_loglikelihood(self,input):
        X = input
        if X.ndim == 2:
            X0=X[0,0]
            X1=X[0,1]
        elif X.ndim == 1:
            X0=X[0]
            X1=X[1]
        else:
            raise RuntimeError('unsupported ndim {}'.format(X.ndim))
        dp = X0
        dz = X1

        project_folder=self.project_folder
        folder_name = self.folder_name
        output_dir = self.output_dir

        flist=get_folder_list(output_dir)
        if len(flist)==0:
            idx = 0
        else:
            flist[-1]
            bname = os.path.basename(flist[-1])
            idx = split_jobindex(bname) + 1

        script_path = self.script_path
        relion_iter = self.relion_iter
        relion_gpu  = self.relion_gpu
        relion_thread = self.relion_thread
        relion_angpix                = self.relion_angpix                
        relion_particle_diameter     = self.relion_particle_diameter     
        relion_helical_outer_diameter= self.relion_helical_outer_diameter
        relion_helical_inner_diameter= self.relion_helical_inner_diameter
        relion_helical_z_percentage  = self.relion_helical_z_percentage
        relion_helical_nr_asu        = self.relion_helical_nr_asu
        relion_K                     = self.relion_K
        relion_sym                   = self.relion_sym
        relion_tau2_fudge            = self.relion_tau2_fudge
        relion_ini_high              = self.relion_ini_high

        dp_str                     = '{}'.format(dp)
        dz_str                     = '{}'.format(dz)
        jobno                      = '{:03d}'.format(idx)
        iter_str                   = '{:d}'.format(relion_iter)
        if relion_angpix == 0:
            angpix_str = '0'
        else:
            angpix_str = '{:f}'.format(relion_angpix)
        particle_diameter_str      = '{:f}'.format(relion_particle_diameter)
        helical_outer_diameter_str = '{:f}'.format(relion_helical_outer_diameter)
        helical_inner_diameter_str = '{:f}'.format(relion_helical_inner_diameter)
        relion_helical_z_percentage_str = '{:f}'.format(relion_helical_z_percentage)
        relion_helical_nr_asu_str  = '{:d}'.format(relion_helical_nr_asu)
        K_str                      = '{:d}'.format(relion_K)
        tau2_fudge_str             = '{:f}'.format(relion_tau2_fudge)
        num_of_thread_str          = '{:d}'.format(relion_thread)
        ini_high_str               = '{:d}'.format(relion_ini_high)

        args = [script_path, jobno, dp_str, dz_str, angpix_str, particle_diameter_str, helical_outer_diameter_str, helical_inner_diameter_str, relion_helical_z_percentage_str, relion_helical_nr_asu_str, relion_sym, tau2_fudge_str, K_str, iter_str, relion_gpu, num_of_thread_str, self.folder_name, self.inputfile, self.referencefile, ini_high_str]
        print(args)
        try:
            res = subprocess.run(args, stdout=subprocess.PIPE)
            sys.stdout.buffer.write(res.stdout)
        except:
            print("Error!!")
        print('done!')

        job_dir = os.path.join(output_dir, 'job' + jobno)
        model_files = get_model_files(job_dir)
        if len(model_files) == 0:
            raise RuntimeError('not found model file')

        lines = get_matched_line_from_file(model_files[-1], '_rlnLogLikelihood(.*)')
        if len(lines) == 0:
            raise RuntimeError('not found loglikelihood')
        sp = lines[0].split()
        ll = -float(sp[1])
        print(dp,dz,ll)

        filepath = os.path.join(output_dir,'byopt_list%s_temp.txt'%self.get_prefix())
        self.save_result_log(filepath)

        return dp, dz, ll


    def grid_search(self,prm):
        self.set_param(prm)
        dp_start = prm.bayesian_opt_dp_start
        dp_end   = prm.bayesian_opt_dp_end
        dp_num   = prm.bayesian_opt_dp_num
        dz_start = prm.bayesian_opt_dz_start
        dz_end   = prm.bayesian_opt_dz_end
        dz_num   = prm.bayesian_opt_dz_num

        dzlist = np.linspace(dz_start,dz_end,dz_num)
        dplist = np.linspace(dp_start,dp_end,dp_num)

        lls = []
        dp_dz_lists=[]
        for dp in dplist:
            for dz  in dzlist:

                X=np.array([[dp,dz]])
                ll=self.get_loglikelihood_2d(X)

                lls.append(ll)
                dp_dz_lists.append({'dz':dz,'dp':dp})


        print('number of loglikelihood=',len(lls))
        indexMax = np.argmax(lls)
        print('max loglikelihood = ',lls[indexMax])
        print('dz={},dp={}'.format(dp_dz_lists[indexMax]['dz'],dp_dz_lists[indexMax]['dp']))
        print(lls)

    def byopt2d_search(self,prm):
        if prm.bayesian_opt_2d_search_load:
            self.load_byopt_list(prm)

        model_type                      = prm.model_type
        acquisition_type                = prm.acquisition_type
        bayesian_opt_normalize_Y        = prm.bayesian_opt_normalize_Y
        bayesian_opt_acquisition_weight = prm.bayesian_opt_acquisition_weight
        bayesian_opt_acquisition_jitter = prm.bayesian_opt_acquisition_jitter
        bayesian_opt_lengthscale        = prm.bayesian_opt_lengthscale
        bayesian_opt_ARD                = prm.bayesian_opt_ARD
        bayesian_opt_2d_search_max_itr  = prm.bayesian_opt_2d_search_max_itr
        bayesian_opt_eps                = prm.bayesian_opt_eps


        self.set_param(prm)
        output_dir       = self.output_dir

        ''' bounds '''
        dp_start = prm.bayesian_opt_dp_start
        dp_end   = prm.bayesian_opt_dp_end
        dp_num   = prm.bayesian_opt_dp_num
        dz_start = prm.bayesian_opt_dz_start
        dz_end   = prm.bayesian_opt_dz_end
        dz_num   = prm.bayesian_opt_dz_num

        if self.initX is None or self.initY is None:
            dzlist = np.linspace(dz_start, dz_end, dz_num)
            dplist = np.linspace(dp_start, dp_end, dp_num)
            self.initX = []
            self.initY = []
            for dp in dplist:
                for dz in dzlist:
                    X = np.array([[dp, dz]])
                    ll = self.get_loglikelihood_2d(X)
                    self.initX.append([dp,dz])
                    self.initY.append(ll)
            self.initX = np.array(self.initX)
            self.initY = np.array(self.initY)[:,np.newaxis]

        bounds = [{'name': 'dp', 'type': 'continuous', 'domain': (dp_start, dp_end)},
                  {'name': 'dz', 'type': 'continuous', 'domain': (dz_start, dz_end)}]

        if bayesian_opt_2d_search_max_itr != 0:
            krn = GPy.kern.Matern52(input_dim=2, lengthscale=bayesian_opt_lengthscale,ARD=bayesian_opt_ARD)
            myBopt = GPyOpt.methods.BayesianOptimization(self.get_loglikelihood_2d,
                                                       X = self.initX,
                                                       Y = self.initY,
                                                       domain=bounds,
                                                       model_type=model_type,
                                                       acquisition_type=acquisition_type,
                                                       normalize_Y=bayesian_opt_normalize_Y,
                                                       acquisition_weight=bayesian_opt_acquisition_weight,
                                                       acquisition_jitter=bayesian_opt_acquisition_jitter,
                                                       kernel=krn,
                                                       maximize=False,
                                                       verbosity=True)
            myBopt.run_optimization(bayesian_opt_2d_search_max_itr, eps=bayesian_opt_eps)
            min_x = myBopt.x_opt

            min_dp = min_x[0]
            min_dz = min_x[1]
            self.set_min_dp_and_dz(min_dp, min_dz)

            print('number of iteration=',len(self.ResultList))
            print('min_x', min_x)
            print('min_dp=', min_dp,'min_dz=',min_dz)
            if SAVE_AQ_CV:
                myBopt.plot_acquisition(filename=os.path.join(output_dir, 'byresult-plot_acquisition%s.png'%self.get_prefix()))
                myBopt.plot_convergence(filename=os.path.join(output_dir, 'byresult-plot_convergence%s.png'%self.get_prefix()))

        filepath = os.path.join(output_dir,'byopt_list%s.txt'%self.get_prefix())
        self.save_result_log(filepath)

    def byopt1d_search(self,prm):
        if prm.bayesian_opt_1d_search_load_1:
            self.load_byopt_list(prm,1)
            self.load_byopt_min(prm, 1)
        if prm.bayesian_opt_1d_search_execute_1:
            dp = prm.bayesian_opt_dp_start
            self.set_fixed_dp(dp)
            self.set_prefix('_1')
            self.byopt1d_search_first(prm)
            min_dp, min_dz = self.get_min_dp_and_dz()
            self.linear_position[0,0]=min_dp
            self.linear_position[0,1]=min_dz
            self.save_byopt_min(os.path.join(self.output_dir,'byopt_min_1.txt'))
        self.clear_result_list()
        self.clear_byopt_list()

        if prm.bayesian_opt_1d_search_load_2:
            self.load_byopt_list(prm,2)
            self.load_byopt_min(prm,2)
        if prm.bayesian_opt_1d_search_execute_2:
            dp = prm.bayesian_opt_dp_end
            self.set_fixed_dp(dp)
            self.set_prefix('_2')
            self.byopt1d_search_first(prm)
            min_dp, min_dz = self.get_min_dp_and_dz()
            self.linear_position[1,0]=min_dp
            self.linear_position[1,1]=min_dz
            self.save_byopt_min(os.path.join(self.output_dir,'byopt_min_2.txt'))
        self.clear_result_list()
        self.clear_byopt_list()

        if prm.bayesian_opt_1d_search_load_3:
            self.load_byopt_list(prm,3)

        if prm.bayesian_opt_1d_search_execute_3:
            self.set_prefix('_3')
            if prm.bayesian_opt_1d_search_margin == 0:
                self.byopt1d_search_second(prm)
            else:
                self.byopt1d_search_second_2d(prm)

    def byopt1d_search_first(self,prm):

        model_type                      = prm.model_type
        acquisition_type                = prm.acquisition_type
        bayesian_opt_normalize_Y        = prm.bayesian_opt_normalize_Y
        bayesian_opt_acquisition_weight = prm.bayesian_opt_acquisition_weight
        bayesian_opt_acquisition_jitter = prm.bayesian_opt_acquisition_jitter
        bayesian_opt_lengthscale        = prm.bayesian_opt_1d_search_lengthscale
        bayesian_opt_ARD                = prm.bayesian_opt_1d_search_ARD
        bayesian_opt_1d_search_first_max_itr = prm.bayesian_opt_1d_search_first_max_itr
        bayesian_opt_eps                = prm.bayesian_opt_eps


        self.set_param(prm)
        output_dir       = self.output_dir

        ''' bounds '''
        dp_start = prm.bayesian_opt_dp_start
        dp_end   = prm.bayesian_opt_dp_end
        dp_num   = prm.bayesian_opt_dp_num
        dz_start = prm.bayesian_opt_dz_start
        dz_end   = prm.bayesian_opt_dz_end
        dz_num   = prm.bayesian_opt_dz_num
        bounds = [{'name': 'dz', 'type': 'continuous', 'domain': (dz_start, dz_end)}]

        if self.initX is None or self.initY is None:
            init_list = np.linspace(dz_start,dz_end,dz_num)
            self.initX = []
            self.initY = []
            for dz in init_list:
                ll=self.get_loglikelihood_1d_first(np.array([[dz]]))
                self.initX.append(dz)
                self.initY.append(ll)
            self.initX = np.array(self.initX)[:,np.newaxis]
            self.initY = np.array(self.initY)[:,np.newaxis]

        if bayesian_opt_1d_search_first_max_itr != 0:
            krn = GPy.kern.Matern52(input_dim=1,lengthscale=bayesian_opt_lengthscale,ARD=bayesian_opt_ARD)
            myBopt = GPyOpt.methods.BayesianOptimization(self.get_loglikelihood_1d_first,
                                                       domain=bounds,
                                                       X = self.initX,
                                                       Y = self.initY,
                                                       model_type=model_type,
                                                       acquisition_type=acquisition_type,
                                                       normalize_Y=bayesian_opt_normalize_Y,
                                                       acquisition_weight=bayesian_opt_acquisition_weight,
                                                       acquisition_jitter=bayesian_opt_acquisition_jitter,
                                                       kernel=krn,
                                                       maximize=False,
                                                       verbosity=True)
            myBopt.run_optimization(bayesian_opt_1d_search_first_max_itr, eps=bayesian_opt_eps)
            min_x = myBopt.x_opt

            min_dz = min_x[0]
            min_dp = self.get_fixed_dp()
            self.set_min_dp_and_dz(min_dp, min_dz)

            print('number of iteration=',len(self.ResultList))
            print('min_x', min_x)
            print('min_dp=', min_dp,'min_dz=',min_dz)
            if SAVE_AQ_CV:
                myBopt.plot_acquisition(filename=os.path.join(output_dir, 'byresult-plot_acquisition%s.png'%self.get_prefix()))
                myBopt.plot_convergence(filename=os.path.join(output_dir, 'byresult-plot_convergence%s.png'%self.get_prefix()))

        filepath = os.path.join(output_dir,'byopt_list%s.txt'%self.get_prefix())
        self.save_result_log(filepath)


    def byopt1d_search_second(self,prm):

        model_type                      = prm.model_type
        acquisition_type                = prm.acquisition_type
        bayesian_opt_normalize_Y        = prm.bayesian_opt_normalize_Y
        bayesian_opt_acquisition_weight = prm.bayesian_opt_acquisition_weight
        bayesian_opt_acquisition_jitter = prm.bayesian_opt_acquisition_jitter
        bayesian_opt_lengthscale        = prm.bayesian_opt_1d_search_lengthscale
        bayesian_opt_ARD                = prm.bayesian_opt_1d_search_ARD
        bayesian_opt_1d_search_second_max_itr = prm.bayesian_opt_1d_search_second_max_itr
        bayesian_opt_eps                = prm.bayesian_opt_eps


        self.set_param(prm)
        output_dir       = self.output_dir

        ''' bounds '''
        dp_start = prm.bayesian_opt_dp_start
        dp_end   = prm.bayesian_opt_dp_end
        dz_start = prm.bayesian_opt_dz_start
        dz_end   = prm.bayesian_opt_dz_end
        t_num = prm.bayesian_opt_t_num

        print('linear_position',self.linear_position)
        t_start,t_end=self.get_t_range(self.linear_position,dp_start,dp_end)
        print('t_start={},t_end={}'.format(t_start,t_end))
        bounds = [{'name': 't', 'type': 'continuous', 'domain': (t_start, t_end)}]

        if self.initX is None or self.initY is None:
            init_list = np.linspace(t_start, t_end,t_num)
            self.initX = []
            self.initY = []
            for t in init_list:
                ll=self.get_loglikelihood_1d_second(np.array([[t]]))
                self.initX.append(t)
                self.initY.append(ll)
            self.initX = np.array(self.initX)[:,np.newaxis]
            self.initY = np.array(self.initY)[:,np.newaxis]

        krn = GPy.kern.Matern52(input_dim=1,lengthscale=bayesian_opt_lengthscale,ARD=bayesian_opt_ARD)
        myBopt = GPyOpt.methods.BayesianOptimization(self.get_loglikelihood_1d_second,
                                                       domain=bounds,
                                                       X = self.initX,
                                                       Y = self.initY,
                                                       model_type=model_type,
                                                       acquisition_type=acquisition_type,
                                                       normalize_Y=bayesian_opt_normalize_Y,
                                                       acquisition_weight=bayesian_opt_acquisition_weight,
                                                       acquisition_jitter=bayesian_opt_acquisition_jitter,
                                                       kernel=krn,
                                                       maximize=False,
                                                       verbosity=True)
        myBopt.run_optimization(bayesian_opt_1d_search_second_max_itr, eps=bayesian_opt_eps)
        min_x = myBopt.x_opt
        min_t  = min_x[0]
        min_dp, min_dz = self.convert_t2dpdz(min_t)

        self.set_min_dp_and_dz(min_dp,min_dz)

        print('number of iteration=',len(self.ResultList))
        print('min_x', min_x)
        print('min_dp=', min_dp,'min_dz=',min_dz)
        if SAVE_AQ_CV:
            myBopt.plot_acquisition(filename=os.path.join(output_dir, 'byresult-plot_acquisition%s.png'%self.get_prefix()))
            myBopt.plot_convergence(filename=os.path.join(output_dir, 'byresult-plot_convergence%s.png'%self.get_prefix()))

        filepath = os.path.join(output_dir,'byopt_list%s.txt'%self.get_prefix())
        self.save_result_log(filepath)
    def byopt1d_search_second_2d(self,prm):

        model_type                      = prm.model_type
        acquisition_type                = prm.acquisition_type
        bayesian_opt_normalize_Y        = prm.bayesian_opt_normalize_Y
        bayesian_opt_acquisition_weight = prm.bayesian_opt_acquisition_weight
        bayesian_opt_acquisition_jitter = prm.bayesian_opt_acquisition_jitter
        bayesian_opt_lengthscale        = prm.bayesian_opt_lengthscale
        bayesian_opt_ARD                = prm.bayesian_opt_ARD
        bayesian_opt_1d_search_second_max_itr = prm.bayesian_opt_1d_search_second_max_itr
        bayesian_opt_eps                = prm.bayesian_opt_eps


        self.set_param(prm)
        output_dir       = self.output_dir

        ''' bounds '''
        dp_start = prm.bayesian_opt_dp_start
        dp_end   = prm.bayesian_opt_dp_end
        dz_start = prm.bayesian_opt_dz_start
        dz_end   = prm.bayesian_opt_dz_end
        t_num = prm.bayesian_opt_t_num
        margin = prm.bayesian_opt_1d_search_margin

        print('linear_position',self.linear_position)
#        t_start,t_end, s_start, s_end =self.get_ts_range(self.linear_position,dp_start,dp_end,-margin,margin)
        t_start,t_end, s_start, s_end =self.get_ts_range(self.linear_position,dp_start,dp_end,-1.0,1.0)
        print('t_start={},t_end={}'.format(t_start,t_end))
        print('s_start={},s_end={}'.format(s_start,s_end))
        bounds = [{'name': 't', 'type': 'continuous', 'domain': (t_start, t_end)},
                  {'name': 's', 'type': 'continuous', 'domain': (s_start, s_end)}
                  ]
        if self.initX is None or self.initY is None:
            init_list = np.linspace(t_start, t_end,t_num)
            self.initX = []
            self.initY = []
            s = 0.0
            for t in init_list:
                ll=self.get_loglikelihood_1d_second_2d(np.array([[t,s]]))
                self.initX.append([t,s])
                self.initY.append(ll)
            self.initX = np.array(self.initX)
            self.initY = np.array(self.initY)[:,np.newaxis]

        if bayesian_opt_1d_search_second_max_itr != 0:
            krn = GPy.kern.Matern52(input_dim=2, lengthscale=bayesian_opt_lengthscale,ARD=bayesian_opt_ARD)
            myBopt = GPyOpt.methods.BayesianOptimization(self.get_loglikelihood_1d_second_2d,
                                                       domain=bounds,
                                                       X = self.initX,
                                                       Y = self.initY,
                                                       model_type=model_type,
                                                       acquisition_type=acquisition_type,
                                                       normalize_Y=bayesian_opt_normalize_Y,
                                                       acquisition_weight=bayesian_opt_acquisition_weight,
                                                       acquisition_jitter=bayesian_opt_acquisition_jitter,
                                                       kernel=krn,
                                                       maximize=False,
                                                       verbosity=True)
            myBopt.run_optimization(bayesian_opt_1d_search_second_max_itr, eps=bayesian_opt_eps)
            min_x = myBopt.x_opt
            min_t  = min_x[0]
            min_s  = min_x[1]
            min_dp, min_dz = self.convert_ts2dpdz(min_t,min_s * self.bayesian_opt_1d_search_margin)

            self.set_min_dp_and_dz(min_dp,min_dz)

            print('number of iteration=',len(self.ResultList))
            print('min_x', min_x)
            print('min_dp=', min_dp,'min_dz=',min_dz)
            if SAVE_AQ_CV:
                myBopt.plot_acquisition(filename=os.path.join(output_dir, 'byresult-plot_acquisition%s.png'%self.get_prefix()))
                myBopt.plot_convergence(filename=os.path.join(output_dir, 'byresult-plot_convergence%s.png'%self.get_prefix()))

        filepath = os.path.join(output_dir,'byopt_list%s.txt'%self.get_prefix())
        self.save_result_log(filepath)

def preprocess(prm,setting_file):
    script_fullpath = os.path.join(get_application_folder(),prm.script_name)
    shutil.copy(script_fullpath,prm.project_dir)

    class_dir = os.path.join(prm.project_dir,prm.folder_name)
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)
    shutil.copy(setting_file,class_dir)

if __name__ == '__main__':
    print('byopt_helix')
    argvs = sys.argv
    if len(argvs) == 2:
        setting_filepath=argvs[1]
    else:
        setting_filepath='setting.txt'

    byopt = BYoptHelix()

    reader = Reader()
    prm= reader.read(setting_filepath)
    if prm is None:
        raise RuntimeError('invalid setting file')

    print('********** start **********')
    print(time.strftime("%a %b %d %H:%M:%S %Y"))

    preprocess(prm,setting_filepath)

    if prm.search_type == 0:
        byopt.grid_search(prm)
    elif prm.search_type == 1:
        byopt.byopt1d_search(prm)
    elif prm.search_type == 2:
        byopt.byopt2d_search(prm)

    print(time.strftime("%a %b %d %H:%M:%S %Y"))
    print('********** end **********')
