# -*- coding: utf-8 -*-

from paje.activelearning.oracle.base import Oracle
from paje.base.data import Data
import numpy as np

class SKLOracle(Oracle):
    def __init__(self, data: Data):
        self.query_count = 0
        self.data = data

    def label_instances(self, instances):
        arr_inst = np.empty((0, self.data.Y.shape[1]), self.data.Y.dtype)

        for item in instances.X:
            for i, row in enumerate(self.data.X):
                if self.__check_row(item, row):
                    arr_inst = np.append(arr_inst, [self.data.Y[i,:]], axis=0)

        self.query_count += 1
        return Data(name='SKLOracle' + str(self.query_count), X=instances.X, 
                      Y=arr_inst, 
                      columns=list(self.data.columns), 
                      history=None)

    def __check_row(self, row1, row2):
        if row1.shape != row2.shape:
            return False
            
        for i, item in enumerate(row1):
            if item != row2[i]:
                return False
        return True