import pandas as pd
import numpy as np
import sys
from data_holder import DataHolder
from tqdm import tqdm
import pickle
import math
import time
import multiprocessing as mp
import pickle
import lightgbm as lgb
from const import RAW_DATA_FOLDER, OUTPUT_FOLDER, CACHE_FOLDER
from recaller import calc_and_recall
from csv_handler import create_train_data
from sklearn.model_selection import train_test_split

def read_raw_data(filename, cb=None):
    data = pd.read_csv(RAW_DATA_FOLDER + filename)
    return cb(data) if cb is not None else data

def read_all_raw_data(filenames=['articles.csv', 'train_click_log.csv', 'testB_click_log_Test_B.csv', 'testA_click_log.csv']):
    return DataHolder(*[read_raw_data(filename) for filename in filenames])

def calc_mrr_and_hit(recommend_dict, y, k=5):
    #assert len(recommend_dict) == len(y)
    sum_mrr = 0.0
    sum_hit = 0.0
    sum_hit_detail = np.repeat(0.0, 5)
    user_cnt = len(recommend_dict.keys())

    for user_id, recommend_items in recommend_dict.items():
        answer = y[user_id] if user_id in y else -1
        if (answer in recommend_items) and (recommend_items.index(answer) < k):
            sum_hit += 1
            sum_mrr += 1 / (recommend_items.index(answer) + 1)
            sum_hit_detail[recommend_items.index(answer)] += 1

    return (sum_mrr / user_cnt), (sum_hit / user_cnt), (sum_hit_detail / user_cnt)

def create_submission(recommend_dict):
    _data = [{'user_id': user_id,
        'article_1': art_id_list[0],
        'article_2': art_id_list[1],
        'article_3': art_id_list[2],
        'article_4': art_id_list[3],
        'article_5': art_id_list[4]} for user_id, art_id_list in tqdm(recommend_dict.items())]
    _t = pd.DataFrame(_data)
    _t.sort_values('user_id', inplace=True)
    _t.to_csv(OUTPUT_FOLDER + 'result.csv', index=False)

def handler(offline=True):
    cpu_cores = mp.cpu_count()
    print('使用CPU核心数: {}'.format(cpu_cores))
    print('开始{}数据验证处理'.format('线下' if offline else '线上'))
    raw_data = read_all_raw_data()
    test_users = raw_data.get_test_users(offline)

    _user_id_list = list(test_users.keys())
    user_id_min = np.min(_user_id_list)
    user_id_max = np.max(_user_id_list)
    print('获得{}用户集合{}件 [{} ~ {}]'.format('验证' if offline else '测试', len(test_users), user_id_min, user_id_max))

    dataset = raw_data.get_item_dt_groupby_user()

    if offline:
        train_dataset, y_answer = raw_data.get_train_dataset_and_answers(test_users)
    else:
        train_dataset = raw_data.get_train_dataset_for_online(test_users)
        y_answer = None

    print('训练数据({}件)'.format(np.sum([len(ts_list) for user_id, ts_list in train_dataset.items()])))

    articles_dic = dict(list(raw_data.get_articles().apply(lambda x: (x['article_id'], dict(x)), axis=1)))
    print('获得文章字典({}件)'.format(len(articles_dic.keys())))

    recall_results = calc_and_recall(dataset, train_dataset, test_users, articles_dic, cpu_cores, offline, y_answer)
    create_train_data(raw_data, train_dataset, test_users, articles_dic, recall_results, offline, y_answer)

def make_train_data():
    handler()

def make_test_data():
    handler(False)

def prepare_dataset(df):
    agg_column = [column for column in df.columns if column != 'user_id'][0]
    df.sort_values('user_id', inplace=True)
    grp_info = df.groupby('user_id', as_index=False).count()[agg_column].values
    y = df['answer'] if 'answer' in df.columns else None
    return df.drop(columns=['answer']) if 'answer' in df.columns else df, grp_info, y

def make_recommend_dict(X_val, y_pred):
    X_val['pred'] = y_pred
    _t = X_val.groupby('user_id')\
        .apply(lambda x: list(x.sort_values('pred', ascending=False)['article_id'].head(5)))\
        .reset_index()\
        .rename(columns={0: 'item_list'})

    recommend_dict = dict(zip(_t['user_id'], _t['item_list']))    
    return recommend_dict

def test():
    df_train = pd.read_csv(CACHE_FOLDER + 'train.csv')

    clf = lgb.LGBMRanker(random_state=777, n_estimators=1000)

    users = df_train['user_id'].unique()
    train_users, _test_users = train_test_split(users, test_size=0.2, random_state=98)
    test_users, val_users = train_test_split(_test_users, test_size=0.5, random_state=38)
    df_new_train = df_train.merge(pd.DataFrame(train_users, columns=['user_id']))
    df_test = df_train.merge(pd.DataFrame(test_users, columns=['user_id']))
    df_val = df_train.merge(pd.DataFrame(val_users, columns=['user_id']))

    X_train, X_grp_train, y_train = prepare_dataset(df_new_train)
    X_test, X_grp_test, y_test = prepare_dataset(df_test)
    X_val, X_grp_val, _ = prepare_dataset(df_val)

    def handle_columns(X):
        return X.drop(columns=['user_id', 'article_id'])

    _X_train = handle_columns(X_train)

    clf.fit(_X_train, y_train, group=X_grp_train, eval_set=[(handle_columns(X_test), y_test)], eval_group=[X_grp_test], eval_at=[1, 2, 3, 4, 5], eval_metric=['ndcg', ], early_stopping_rounds=50, verbose=False)
    print('Best iteration: {}'.format(clf.best_iteration_))


    for X, X_grp, df, title in [(X_test, X_grp_test, df_test, 'Test Set'), (X_val, X_grp_val, df_val, 'Validation Set')]:
        print('[{}]'.format(title))
        y_pred = clf.predict(handle_columns(X), group=X_grp, num_iteration=clf.best_iteration_)
        recommend_dict = make_recommend_dict(X, y_pred)
        answers = dict(df.loc[df['answer'] == 1, ['user_id', 'article_id']].values)
        mrr, hit, details = calc_mrr_and_hit(recommend_dict, answers)
        print('MRR: {} / HIT: {}'.format(mrr, hit))
        print(' / '.join(['%.2f' % detail for detail in details]))

    for column, score in sorted(zip(_X_train.columns, clf.feature_importances_), key=lambda x: x[1], reverse=True):
        print('{}: {}'.format(column, score))

def run():
    df_train = pd.read_csv(CACHE_FOLDER + 'train.csv')
    df_test = pd.read_csv(CACHE_FOLDER + 'test.csv')

    clf = lgb.LGBMRanker(random_state=777, n_estimators=1000)

    users = df_train['user_id'].unique()
    train_users, eval_users = train_test_split(users, test_size=0.2, random_state=77)
    df_new_train = df_train.merge(pd.DataFrame(train_users, columns=['user_id']))
    df_eval = df_train.merge(pd.DataFrame(eval_users, columns=['user_id']))

    X_train, X_grp_train, y_train = prepare_dataset(df_new_train)
    X_eval, X_grp_eval, y_eval = prepare_dataset(df_eval)
    X_test, X_grp_test, _ = prepare_dataset(df_test)

    def handle_columns(X):
        return X.drop(columns=['user_id', 'article_id'])

    _X_train = handle_columns(X_train)

    clf.fit(_X_train, y_train, group=X_grp_train, eval_set=[(handle_columns(X_eval), y_eval)], eval_group=[X_grp_eval], eval_at=[1, 2, 3, 4, 5], eval_metric=['ndcg', ], early_stopping_rounds=50, verbose=False)
    print('Best iteration: {}'.format(clf.best_iteration_))
    y_pred = clf.predict(handle_columns(X_test), group=X_grp_test, num_iteration=clf.best_iteration_)
    
    for column, score in sorted(zip(_X_train.columns, clf.feature_importances_), key=lambda x: x[1], reverse=True):
        print('{}: {}'.format(column, score))

    recommend_dict = make_recommend_dict(X_test, y_pred)

    create_submission(recommend_dict)

if __name__ == "__main__":
    make_train_data()
    # test()
    make_test_data()
    run()
