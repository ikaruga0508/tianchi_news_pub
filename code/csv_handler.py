import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from const import CACHE_FOLDER

def neg_sampling(ds, min=1, max=5):
    start_time = time.time()
    pos_ds = ds.loc[ds['answer'] == 1]
    neg_ds = ds.loc[ds['answer'] == 0]

    def _neg_sampling_func(x):
        n_sampling = len(x)
        n_sampling = min if n_sampling < min else (max if n_sampling > max else n_sampling)
        return x.sample(n=n_sampling, replace=False)

    neg_ds = pd.concat([
        neg_ds.groupby(['user_id', 'last_clicked_timestamp']).apply(_neg_sampling_func),
        neg_ds.groupby('article_id').apply(_neg_sampling_func),
        ]).drop_duplicates()

    ret = pd.concat([pos_ds, neg_ds]).reset_index(drop=True)
    print('负采样处理完毕({}秒, {}->{}件)'.format('%.2f' % (time.time() - start_time), len(ds), len(ret)))
    return ret

def get_user_features(raw_data, train_dataset, test_users, articles_dic):
    def calc_avg_words_count(items):
        return np.average([articles_dic[item[0]]['words_count'] for item in items])

    def calc_min_words_count(items):
        return np.min([articles_dic[item[0]]['words_count'] for item in items])

    def calc_max_words_count(items):
        return np.max([articles_dic[item[0]]['words_count'] for item in items])

    def calc_lag_between_created_at_ts_and_clicked_ts(items, articles_dic):
        item = items[-1]
        return (item[1] - articles_dic[item[0]]['created_at_ts']) / (1000 * 60 * 60 * 24)

    def calc_lag_between_two_click(items):
        if len(items) > 1:
            return (items[-1][1] - items[-2][1]) / (1000 * 60 * 60 * 24)
        else:
            return np.nan

    def calc_lag_between_two_articles(items, articles_dic):
        if len(items) > 1:
            return (articles_dic[items[-1][0]]['created_at_ts'] - articles_dic[items[-2][0]]['created_at_ts']) / (1000 * 60 * 60 * 24)
        else:
            return np.nan

    df_users = pd.DataFrame(list(test_users.keys()), columns=['user_id'])

    # 计算
    # 1. 用户看新闻的平均字数
    _data = []
    for user_id, ts_set in tqdm(test_users.items()):
        for last_clicked_timestamp in ts_set:
            _data.append((
                user_id,
                last_clicked_timestamp,
                calc_avg_words_count(train_dataset[user_id][last_clicked_timestamp]),
                calc_min_words_count(train_dataset[user_id][last_clicked_timestamp]),
                calc_max_words_count(train_dataset[user_id][last_clicked_timestamp]),
                calc_lag_between_created_at_ts_and_clicked_ts(train_dataset[user_id][last_clicked_timestamp], articles_dic),
                calc_lag_between_two_click(train_dataset[user_id][last_clicked_timestamp]),
                calc_lag_between_two_articles(train_dataset[user_id][last_clicked_timestamp], articles_dic),
                ))

    df1 = pd.DataFrame(_data, columns=['user_id', 'last_clicked_timestamp', 'avg_words_count', 'min_words_count', 'max_words_count', 'lag_between_created_at_ts_and_clicked_ts', 'lag_between_two_click', 'lag_between_two_articles'])

    # 计算用户使用设备，环境等的众数
    columns = ['user_id','click_environment','click_deviceGroup','click_os','click_country','click_region','click_referrer_type']
    df2 = df_users.merge(raw_data.get_all_click_log())[columns].groupby('user_id').agg(lambda x: x.value_counts().index[0]).reset_index()

    return df1.merge(df2)
    
def create_train_data(raw_data, train_dataset, test_users, articles_dic, recall_results, offline, y_answer):
    start_time = time.time()
    keys_ds = []

    for user_id, ts_set in test_users.items():
        for last_clicked_timestamp in ts_set:
            items = np.concatenate([result[user_id][last_clicked_timestamp] for _, result in recall_results.items()])
            keys_ds.append(list(zip(np.repeat(user_id, len(items)), np.repeat(last_clicked_timestamp, len(items)), items)))

    ds = pd.DataFrame(np.concatenate(keys_ds), columns=['user_id', 'last_clicked_timestamp', 'article_id'], dtype=np.int64).drop_duplicates()

    if offline:
        answer_keys_ds = []
        # 拼接正确答案标签
        for user_id, ts_list in y_answer.items():
            for last_clicked_timestamp, art_id in ts_list.items():
                answer_keys_ds.append((user_id, last_clicked_timestamp, art_id))

        answers = pd.DataFrame(answer_keys_ds, columns=['user_id', 'last_clicked_timestamp', 'article_id'], dtype=np.int64)
        # 将正确答案融合进数据集
        answers['answer'] = 1
        ds = ds.merge(answers, how='left').fillna({'answer': 0})
        ds['answer'] = ds['answer'].astype(np.int8)

        # 负采样
        ds = neg_sampling(ds)

    ds = ds.merge(raw_data.get_articles()).merge(get_user_features(raw_data, train_dataset, test_users, articles_dic))

    # 新特征
    ds['lag_period_last_article'] = ds['last_clicked_timestamp'] - ds['created_at_ts']
    ds['diff_words_last_article'] = ds['avg_words_count'] - ds['words_count']
    ds.to_csv(CACHE_FOLDER + '{}.csv'.format('train' if offline else 'test'), index=False)
    print('{}用的csv文件生成完毕({}秒, {}件)'.format('训练' if offline else '测试', '%.2f' % (time.time() - start_time), len(ds)))