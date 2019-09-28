from imblearn.under_sampling import RandomUnderSampler

from paje.searchspace.configspace import HPTree, ConfigSpace
from paje.ml.element.preprocessing.supervised.instance.sampler.resampler \
    import Resampler
from paje.searchspace.hp import CatHP
from paje.util.distributions import choice


class RanUnderSampler(Resampler):
    def build_impl(self):
        self.model = RandomUnderSampler(**self.config)

    @classmethod
    def cs_impl(cls, data=None):
        hps = {'sampling_strategy': CatHP(
            choice,
            items=['majority', 'not minority', 'not majority', 'all']
        )}
        return ConfigSpace(name=cls.__name__, hps=hps)
