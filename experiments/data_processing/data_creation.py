from distutils import core

from divisiveclustering.coresetsUtils import Coreset
from divisiveclustering.datautils import DataUtils

data_creation = DataUtils("/Users/yogi/libraries/Kmeans_NISQ")

raw_data = data_creation.create_new_data_set(1000, 4, 2, 10, True)

coresetutil = Coreset()

cv, cw = coresetutil.get_coresets(
    data_vectors=raw_data, number_of_runs=10, coreset_numbers=5, size_vec_list=10
)

bcv, bcw = coresetutil.get_best_coresets(raw_data, cv, cw)

data_creation.save_coresets(bcv, bcw, 5, 4)
