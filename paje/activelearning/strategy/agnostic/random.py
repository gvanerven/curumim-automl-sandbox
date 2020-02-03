# -*- coding: utf-8 -*-

from paje.activelearning.strategy.base import Strategy
from paje.base.data import Data
import pandas as pd
import random as rd
import numpy as np

class RandomSampling(Strategy):
    def __init__(self, datasource, ds_type, replacement=False, seed=None, **kargs):
        super().__init__(datasource, ds_type)
        self.set_type()

        self.replacement = replacement
        
        if replacement:
            self.to_query = None
        else:
            self.to_query = list(range(self.features_shape[0]))
            
        self.queried = []
        self.nr_samples = 0
        rd.seed(seed)

    def set_type(self):
        if self.ds_type == 'paje_data':
            self.features_shape = self.datasource.X.shape
            self.query_function = self.query_paje_data
            self.columns = list(self.datasource.columns)
        else:
            print("Warning: type not defined!")

    @classmethod
    def suported_types(cls):
        return ['paje_data', 'sql']

    def query(self, nr_records:int) -> Data:
        self.nr_samples += 1
        return self.query_function(nr_records)

    def query_paje_data(self, nr_records:int) -> Data:
        insts = []
        arr_inst = np.empty((0,self.features_shape[1]), self.datasource.X.dtype)
        
        if len(self.queried) == self.features_shape[0]:
            return None
        
        if self.replacement:
            insts += rd.sample(range(self.features_shape[0]), k=nr_records)
            self.queried += insts
        else:
            aux_insts = rd.sample(range(len(self.to_query)), k=nr_records)
            
            for j in aux_insts:
                insts.append(self.to_query[j])
                del self.to_query[j]
            self.queried += insts
            
        for record in insts:
            arr_inst = np.append(arr_inst, [self.datasource.X[record,:]], axis=0)

        return Data(name='sample' + str(self.nr_samples), X=arr_inst, columns=self.columns, history=None)
            