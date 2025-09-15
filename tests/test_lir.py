import matplotlib.pyplot as plt
import numpy as np
from mbb import ModifiedBlackbody as MBB
from multiprocessing import freeze_support
import time



if __name__ == "__main__":
    freeze_support()
    
    m1 = MBB(z=2.5,L=12.6,T=35,beta=2.0,pl=True,opthin=False)
    m1.fit(phot=([450, 850,1200],[0.005, 0.0021,0.00078],[0.0006,0.00032,0.00025]),niter=1000,params=['L','T','beta'],restframe=False)
    
    print('after fit:')
    print('LIR', m1.L)

    # start = time.time()
    # print('percentiles', m1.post_percentile('L',q=(16,50,84)))
    # end = time.time()
    # print('Time:',(end-start)/60,'min')

    from cProfile import Profile
    from pstats import SortKey, Stats

    with Profile() as profile:
        start = time.time()
        print('percentiles', m1.post_percentile('L',q=(16,50,84),sample_by=5))
        end = time.time()
    print('Time:',(end-start)/60,'min')
    print(Stats(profile)
        .strip_dirs()
        .sort_stats("tottime")
        .print_stats(25))

    # start = time.time()
    # print('percentiles', m1.post_percentile('L',q=(16,50,84),sample_by=10))
    # end = time.time()
    # print('Time:',(end-start)/60,'min')

    # start = time.time()
    # print('percentiles', m1.post_percentile('L',q=(16,50,84),sample_by=20))
    # end = time.time()
    # print('Time:',(end-start)/60,'min')

    # start = time.time()
    # print('percentiles', m1.post_percentile('L',q=(16,50,84),sample_by=100))
    # end = time.time()
    # print('Time:',(end-start)/60,'min')