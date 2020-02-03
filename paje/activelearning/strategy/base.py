# -*- coding: utf-8 -*-
from paje.base.data import Data

class Strategy:
    def __init__(self, datasource, ds_type, **kargs):
        self.datasource = datasource
        self.ds_type = ds_type.lower()
    
    def query(self, nr_instances:int) -> Data:
        pass
    
    @classmethod
    def suported_types(cls):
        return []