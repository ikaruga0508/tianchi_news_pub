import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import time
from os.path import isfile
from const import CACHE_FOLDER

class DataHolder:
    def __init__(self, articles, train_click_log, test_click_log, trainB_click_log=None):
        self.articles = articles
        self.train_click_log = train_click_log
        self.test_click_log = test_click_log
        self.trainB_click_log = trainB_click_log
        self.all_click_log = self.train_click_log.append(self.trainB_click_log).append(self.test_click_log) if self.trainB_click_log is not None else self.train_click_log.append(self.test_click_log)
        self.all_click_log.drop_duplicates(['user_id', 'click_article_id', 'click_timestamp'], inplace=True)

        print('从train_click_log读取{}件(UserId=[{},{}])'.format(len(self.train_click_log), self.train_click_log['user_id'].min(), self.train_click_log['user_id'].max()))
        print('从test_click_log读取{}件(UserId=[{},{}])'.format(len(self.test_click_log), self.test_click_log['user_id'].min(), self.test_click_log['user_id'].max()))

        if self.trainB_click_log is not None:
            print('从trainB_click_log读取{}件(UserId=[{},{}])'.format(len(self.trainB_click_log), self.trainB_click_log['user_id'].min(), self.trainB_click_log['user_id'].max()))

        print('使用训练集all_click_log共{}件(UserId=[{},{}])'.format(len(self.all_click_log), self.all_click_log['user_id'].min(), self.all_click_log['user_id'].max()))

        # DataFrame对象转换成字典
        filename = 'dataset.pkl'
        if isfile(CACHE_FOLDER + filename):
            print('直接从文件{}中读取dataset'.format(filename))
            self.dataset = pickle.load(open(CACHE_FOLDER + filename, 'rb'))
        else:
            start_time = time.time()
            _t = self.all_click_log.sort_values('click_timestamp').groupby('user_id')\
                .apply(lambda x: list(zip(x['click_article_id'], x['click_timestamp'])))\
                .reset_index()\
                .rename(columns={0: 'item_dt_list'})

            self.dataset = dict(zip(_t['user_id'], _t['item_dt_list']))
            print('dataset对象完毕({}秒)'.format('%.2f' % (time.time() - start_time)))

            print('保存dataset至文件{}中'.format(filename))
            pickle.dump(self.dataset, open(CACHE_FOLDER + filename, 'wb'))

        # 生成可供训练用的(user_id, timestamp)字典
        filename = 'train_users_dic.pkl'
        if isfile(CACHE_FOLDER + filename):
            print('直接从文件{}中读取train_users_dic'.format(filename))
            self.train_users_dic = pickle.load(open(CACHE_FOLDER + filename, 'rb'))
        else:
            start_time = time.time()
            self.train_users_dic = {}
            for user_id, items in tqdm(self.dataset.items()):
                ts_list = pd.Series([item[1] for item in items])
                self.train_users_dic[user_id] = list(ts_list.loc[ts_list.shift(-1) - ts_list == 30000])

            print('train_users_dic对象完毕({}秒)'.format('%.2f' % (time.time() - start_time)))

            print('保存train_users_dic至文件{}中'.format(filename))
            pickle.dump(self.train_users_dic, open(CACHE_FOLDER + filename, 'wb'))

    def get_articles(self):
        return self.articles

    def get_train_click_log(self):
        return self.train_click_log

    def get_test_click_log(self):
        return self.test_click_log

    def get_all_click_log(self):
        return self.all_click_log

    def get_user_list(self):
        return self.train_click_log['user_id'].unique()

    def get_item_dt_groupby_user(self):
        return self.dataset

    def users_df2dic(self, df_users):
        _t = df_users.sort_values('click_timestamp').groupby('user_id')\
            .apply(lambda x: set(x['click_timestamp']))\
            .reset_index()\
            .rename(columns={0: 'ts_set'})

        return dict(zip(_t['user_id'], _t['ts_set']))

    def get_test_users(self, offline, samples=100000):
        if offline:
            # 一维数组化
            users = []
            for user_id, ts_list in self.train_users_dic.items():
                # for ts in ts_list:
                #     users.append((user_id, ts))
                if len(ts_list) > 0:
                    users.append((user_id, ts_list[-1]))
                
            np.random.seed(42)
            idx_list = np.random.choice(len(users), samples, replace=False)
            selected_users = [users[idx] for idx in idx_list]

            # 字典化
            return self.users_df2dic(pd.DataFrame(selected_users, columns=['user_id', 'click_timestamp']))
        else:
            return self.users_df2dic(self.test_click_log.groupby('user_id').max('click_timestamp').reset_index()[['user_id', 'click_timestamp']])

    def take_last(self, items, last=1):
        if len(items) <= last:
            return items.copy(), items[0]
        else:
            return items[:-last], items[-last]

    def get_train_dataset_and_answers(self, test_users):
        start_time = time.time()
        train_dataset = {}
        y_answer = {}

        for user_id, ts_set in tqdm(test_users.items()):
            items = self.dataset[user_id]
            for last_clicked_timestamp in ts_set:
                idx = [item[1] for item in items].index(last_clicked_timestamp)
                train_dataset.setdefault(user_id, {})
                train_dataset[user_id][last_clicked_timestamp] = items[0:idx+1]
                y_answer.setdefault(user_id, {})
                y_answer[user_id][last_clicked_timestamp] = items[idx+1][0]

        print('训练集和答案分割完毕({}秒)'.format('%.2f' % (time.time() - start_time)))
        return train_dataset, y_answer

    def get_train_dataset_for_online(self, test_users):
        start_time = time.time()
        train_dataset = {}

        for user_id, ts_set in tqdm(test_users.items()):
            items = self.dataset[user_id]
            for last_clicked_timestamp in ts_set:
                train_dataset.setdefault(user_id, {})
                train_dataset[user_id][last_clicked_timestamp] = items

        print('测试集制作完毕({}秒)'.format('%.2f' % (time.time() - start_time)))
        return train_dataset
