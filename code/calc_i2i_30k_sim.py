import numpy as np
import pandas as pd
import time
import multiprocessing as mp
from tqdm import tqdm
import pickle
import math
from os.path import isfile
from const import CACHE_FOLDER

def _i2i_30k_sim_core(job_id, user_id_list, dataset):
    _item_counts_dic = {}
    _i2i_30k_sim = {}

    start_time = time.time()
    for user_id in user_id_list:
        item_dt = dataset[user_id]
        ts_list = pd.Series([ts for _, ts in item_dt])
        idx_list = [idx for idx, val in dict(ts_list - ts_list.shift(1) == 30000).items() if val]

        for idx in idx_list:
            i_art_id, _ = item_dt[idx]
            j_art_id, _ = item_dt[idx - 1]

            _i2i_30k_sim.setdefault(i_art_id, {})
            _i2i_30k_sim[i_art_id].setdefault(j_art_id, 0)
            _i2i_30k_sim[i_art_id][j_art_id] += 1

            _i2i_30k_sim.setdefault(j_art_id, {})
            _i2i_30k_sim[j_art_id].setdefault(i_art_id, 0)
            _i2i_30k_sim[j_art_id][i_art_id] += 1

    print('子任务[{}]: 完成i2i_30k相似度的计算。({}秒)'.format(job_id, '%.2f' % (time.time() - start_time)))

    return _i2i_30k_sim

def i2i_30k_sim(dataset, n_cpu, offline, max_related=50):
    filename = 'i2i_30k_sim_{}.pkl'.format('offline' if offline else 'online')
    if isfile(CACHE_FOLDER + filename):
        print('直接从文件{}中读取计算好的i2i_30k相似度'.format(filename))
        return pickle.load(open(CACHE_FOLDER + filename, 'rb'))

    # 计算相似度
    start_time = time.time()
    print('开始计算i2i_30k相似度')
    i2i_sim_3k = {}
    n_block = (len(dataset.keys()) - 1) // n_cpu + 1
    keys = list(dataset.keys())
    pool = mp.Pool(processes=n_cpu)
    results = [pool.apply_async(_i2i_30k_sim_core, args=(i, keys[i * n_block:(i + 1) * n_block], dataset)) for i in range(0, n_cpu)]
    pool.close()
    pool.join()

    for result in results:
        _i2i_sim_3k = result.get()

        for art_id, related_art_id_dic in _i2i_sim_3k.items():
            i2i_sim_3k.setdefault(art_id, {})
            for related_art_id, value in related_art_id_dic.items():
                i2i_sim_3k[art_id].setdefault(related_art_id, 0)
                i2i_sim_3k[art_id][related_art_id] += value

    print('逆序排序')
    for art_id, related_arts in tqdm(i2i_sim_3k.items()):
        sorted_and_topK = sorted(related_arts.items(), key=lambda x: x[1], reverse=True)
        i2i_sim_3k[art_id] = {
            'sorted_keys': [art_id for art_id, _ in sorted_and_topK],
            'related_arts': dict(sorted_and_topK)
        } 

    print('i2i_30k相似度计算完毕({}秒)'.format('%.2f' % (time.time() - start_time)))
    print('保存i2i_30k相似度数据至文件{}中'.format(filename))
    pickle.dump(i2i_sim_3k, open(CACHE_FOLDER + filename, 'wb'))
    return i2i_sim_3k

