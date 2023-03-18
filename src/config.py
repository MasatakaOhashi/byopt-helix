import numpy as np
import pandas as pd
import ast
import os
from distutils.util import strtobool
from param import ByoptParam

class ConfigReader(object):
    """
    self.dt=
    {'prm.common.archive_mode': ['s', 'xz'],
     'prm.common.code_archive_folder': ['s', 'archives'],
     'prm.common.enable_code_archive': ['b', 'False'],
     'prm.common.enable_double_precision': ['b', 'False'],
     'prm.common.enable_max_samples_per_device': ['b', 'True'],
     'prm.common.enable_refuge_result': ['b', 'False'],
     'prm.common.enabled_devices': ['list', '[1,2]'],
     'prm.common.estimate_next_index_from_temp_folder': ['b', 'False'],
     'prm.common.factory_name': ['s', 'Examples.helix.factory.factory_test_21'],
     'prm.common.get_next_index_from_config_file': ['b', 'False'],
     'prm.common.max_samples_per_device_estep': ['i', '0'],
     'prm.common.max_samples_per_device_mstep': ['i', '0'],
    """

    def __init__(self, type=1):
        self.type = type
    def exists(self,key):
        return key in self.dt.keys()

    def read_config_file(self, filename):
        if os.path.exists(filename):
            if self.type == 0:
                column_names = ['tag', 'param']
            else:
                column_names = ['tag', 'type', 'param']

            self.df = pd.read_csv(filename, header = None, names = column_names,comment='#',skipinitialspace=True)
            self.dt = self.df.set_index('tag').T.to_dict('list')

    def get_value(self, key,type='s'):
        if key in self.dt.keys():
            value_idx = 0 if self.type == 0 else 1
            val = self.dt[key][value_idx]
            if val is None:
                return val
            elif type == 'i':
                return int(val)
            elif type == 'b':
                return bool(strtobool(val))
            elif type == 'f':
                return float(val)
            elif type == 'dict':
                return ast.literal_eval(val)
            elif type == 'list':
                return ast.literal_eval(val)
            elif type == 'n1i':
                return np.fromstring(val,dtype=np.int, sep=' ')
            elif type == 'n1f':
                return np.fromstring(val, dtype=np.float, sep=' ')
            else:
                return val
        else:
            return None

    def get_type(self, key):
        if self.type != 0:
            if key in self.dt.keys():
                type_idx = 0
                val = self.dt[key][type_idx]
                return val
            else:
                return None
        else:
            return None


class Reader(object):
    def __init__(self):
        pass

    def read(self, fullpath):
        """

        :param fullpath:
        :return: ByoptParam
        """
        if not os.path.exists(fullpath):
            raise RuntimeError('not found {}'.format(fullpath))
            return None
        prm = ByoptParam()

        cnfReader = ConfigReader(type=1)
        cnfReader.read_config_file(fullpath)
        for key in prm.__dict__.keys():
            tag = '%s' % key
            if cnfReader.exists(tag):
                val = cnfReader.get_value(tag, type=cnfReader.get_type(tag))
                prm.__dict__[key] = val
                print(tag, '=', val, type(val))

        print('***** parameters *****\n')
        for key in prm.__dict__.keys():
            print(key, '=', prm.__dict__[key])

        return prm

class ByOptInitReader(object):
    def __init__(self):
        self.df=None
        self.filepath=None

    def open(self,filepath):
        self.filepath=filepath
        self.df = pd.read_csv(filepath,comment='#')

    def get_num_of_column(self):
        return len(self.df.keys())

    def read(self):
        init_dp = self.df['dp']
        init_dz = self.df['dz']
        init_lb = self.df['lb']
        if self.get_num_of_column() == 4:
            init_t = self.df['t']
            return init_dp,init_dz,init_lb,init_t
        elif self.get_num_of_column() == 5:
            init_t = self.df['t']
            init_s = self.df['s']
            return init_dp, init_dz, init_lb, init_t,init_s
        else:
            return init_dp,init_dz,init_lb

    def close(self):
        self.df=None
        self.filepath=None

class ByOptMinReader(object):
    def __init__(self):
        self.df=None
        self.filepath=None

    def open(self,filepath):
        self.filepath=filepath
        self.df = pd.read_csv(filepath,comment='#')


    def read(self,filepath):
        self.open(filepath)
        return self.df.values
    def close(self):
        self.df=None
        self.filepath=None

class ByOptMinWriter(object):
    def __init__(self):
        pass
    def write(self,data,filepath):

        with open(filepath,'w') as f:
            print('min_dp,min_dz',file=f)

            for index in range(data.shape[0]):
                print('{0:.8e},{1:.8e}'.format(data[index,0],data[index,1]),file=f)
