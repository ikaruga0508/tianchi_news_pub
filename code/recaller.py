from calc_i2i_30k_sim import i2i_30k_sim
from tqdm import tqdm
import multiprocessing as mp
import time
import math
import pandas as pd
import numpy as np

def get_clicked_items(items):
    return { art_id for art_id, _ in items }

def _calc_sim(dataset, articles_dic, cpu_cores, offline):
    # 计算各种相似度
    num = len([i2i_30k_sim])

    start_time = time.time()
    print('召回前的计算处理开始({}件)'.format(num))

    sims = {}
    sims['i2i_30k_sim'] = i2i_30k_sim(dataset, cpu_cores, offline)

    print('召回前的计算处理结束({}秒)'.format('%.2f' % (time.time() - start_time)))

    return sims


def _is_recall_target(last_clicked_timestamp, art_id, articles_dic, lag_hour_max=27, lag_hour_min=3):
    # 热度文章在用户最后一次点击时刻起，前3小时~27小时内的文章
    lag_max = lag_hour_max * 60 * 60 * 1000
    lag_min = lag_hour_min * 60 * 60 * 1000
    if articles_dic[art_id]['created_at_ts'] < (last_clicked_timestamp - lag_max):
        return False

    if articles_dic[art_id]['created_at_ts'] > (last_clicked_timestamp - lag_min):
        return False

    return True

def _recall_hot_items(dataset, train_dataset, test_users, articles_dic, topK=10):
    result = {}
    start_time = time.time()
    lag_hour_min = 3
    lag_hour_max = 27

    hot_items = {}
    for _, items in tqdm(dataset.items()):
        for art_id, _ in items:
            hot_items.setdefault(art_id, 0)
            hot_items[art_id] += 1

    sorted_hot_items = sorted(hot_items.items(), key=lambda x: x[1], reverse=True)

    for user_id, ts_set in tqdm(test_users.items()):
        for last_clicked_timestamp in ts_set:
            items = train_dataset[user_id][last_clicked_timestamp]
            clicked_items = get_clicked_items(items)
            recommend_items = []

            for art_id, _ in sorted_hot_items:
                if art_id in clicked_items:
                    continue

                if not _is_recall_target(last_clicked_timestamp, art_id, articles_dic, lag_hour_min=lag_hour_min, lag_hour_max=lag_hour_max):
                    continue

                recommend_items.append(art_id)

                if len(recommend_items) >= topK:
                    break

            result.setdefault(user_id, {})
            result[user_id][last_clicked_timestamp] = recommend_items

    print('hot召回处理完毕({}秒) 限制：[{}-{}]'.format('%.2f' % (time.time() - start_time), lag_hour_min, lag_hour_max))
    return result

def _recall_i2i_30k_sim_items(dataset, test_users, articles_dic, i2i_30k_sim, topK=25):
    result = {}
    start_time = time.time()
    lag_hour_min = 0
    lag_hour_max = 27

    for user_id, ts_set in tqdm(test_users.items()):
        for last_clicked_timestamp in ts_set:
            items = dataset[user_id][last_clicked_timestamp]
            clicked_items = get_clicked_items(items)
            recommend_items = {}

            for art_id, _ in items:
                if art_id not in i2i_30k_sim:
                    break

                recommand_art_id_list = i2i_30k_sim[art_id]['sorted_keys']
                for recommend_art_id in recommand_art_id_list:
                    if recommend_art_id in clicked_items:
                        continue

                    if not _is_recall_target(last_clicked_timestamp, art_id, articles_dic, lag_hour_min=lag_hour_min, lag_hour_max=lag_hour_max):
                        continue

                    if i2i_30k_sim[art_id]['related_arts'][recommend_art_id] < 2:
                        break

                    recommend_items.setdefault(recommend_art_id, 0)
                    recommend_items[recommend_art_id] += (i2i_30k_sim[art_id]['related_arts'][recommend_art_id])
            
            result.setdefault(user_id, {})
            result[user_id][last_clicked_timestamp] = [art_id for art_id, _ in sorted(recommend_items.items(), key=lambda x: x[1], reverse=True)[:topK]]

    print('i2i_30k_sim召回处理完毕({}秒) 限制：[{}-{}]'.format('%.2f' % (time.time() - start_time), lag_hour_min, lag_hour_max))
    return result

def calc_and_recall(dataset, train_dataset, test_users, articles_dic, cpu_cores, offline, answers=None):
    sims = _calc_sim(dataset, articles_dic, cpu_cores, offline)
    num = len([_recall_hot_items, _recall_i2i_30k_sim_items])

    start_time = time.time()
    print('召回处理开始({}件)'.format(num))

    recalls = {}
    recalls['hot'] = _recall_hot_items(dataset, train_dataset, test_users, articles_dic)
    recalls['i2i_30k_sim'] = _recall_i2i_30k_sim_items(train_dataset, test_users, articles_dic, sims['i2i_30k_sim'])

    if offline and answers is not None:
        test_users_count = np.sum([len(ts_list) for _, ts_list in test_users.items()])
        for recall_name, result in recalls.items():
            accuracy = 0
            recall_counts = np.repeat(0, np.max([len(items) for _, ts_list in result.items() for _, items in ts_list.items()]))
            for user_id, ts_list in result.items():
                for last_clicked_timestamp, items in ts_list.items():
                    if answers[user_id][last_clicked_timestamp] in items:
                        accuracy += 1 
                        recall_counts[items.index(answers[user_id][last_clicked_timestamp])] += 1
            
            print('召回处理[{}]的召回率为{}%'.format(recall_name, '%.2f' % (accuracy * 100 / test_users_count)))
            print('召回处理[{}]的详细召回命中计数: {}'.format(recall_name, recall_counts))

        total_accuracy = 0
        for user_id, ts_list in test_users.items():
            for last_clicked_timestamp in ts_list:
                for _, result in recalls.items():
                    if answers[user_id][last_clicked_timestamp] in result[user_id][last_clicked_timestamp]:
                        total_accuracy += 1
                        break

        print('所有召回处理的总召回率为{}%'.format('%.2f' % (total_accuracy * 100 / test_users_count)))

    print('召回处理结束({}秒)'.format('%.2f' % (time.time() - start_time)))

    return recalls
