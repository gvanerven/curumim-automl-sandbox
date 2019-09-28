from imblearn.over_sampling import RandomOverSampler

from paje.searchspace.configspace import HPTree, ConfigSpace
from paje.ml.element.preprocessing.supervised.instance.sampler.resampler import \
    Resampler
from paje.searchspace.hp import CatHP
from paje.util.distributions import choice


class RanOverSampler(Resampler):
    def build_impl(self):
        self.model = RandomOverSampler(**self.config)

    @classmethod
    def cs_impl(cls, data=None):
        hps = {'sampling_strategy': CatHP(
            choice,
            items=['not minority', 'not majority', 'all']
        )}
        return ConfigSpace(name=cls.__name__, hps=hps)
