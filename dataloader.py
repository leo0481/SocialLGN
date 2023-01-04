import gc
import json
import os
import re
from time import time
import logging
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import torch
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split, GroupShuffleSplit

import world


def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))


class PairDataset:
    def __init__(self, src="lastfm"):
        self.src = src
        try:
            self.train_set = pd.read_csv(f'./data/preprocessed/{src}/train_set.txt')
            self.test_set = pd.read_csv(f'./data/preprocessed/{src}/test_set.txt')
            self.n_user = pd.concat([self.train_set, self.test_set])['user'].nunique()
            self.m_item = pd.concat([self.train_set, self.test_set])['item'].nunique()
        except IOError:
            self.interactionNet, self.n_user, self.m_item = loadInteraction(src)
            self.train_set, self.test_set = splitDataset(self.interactionNet)
            self.train_set.to_csv(f'./data/preprocessed/{src}/train_set.txt', index=False)
            self.test_set.to_csv(f'./data/preprocessed/{src}/test_set.txt', index=False)

        self.trainUser = np.array(self.train_set['user'])
        self.trainUniqueUser = np.unique(self.train_set['user'])
        self.trainItem = np.array(self.train_set['item'])
        self._trainDataSize = len(self.train_set)
        self._testDataSize = len(self.test_set)
        print(f"{self._trainDataSize} interactions for training")
        print(f"{self._testDataSize} interactions for testing")
        print(f"Number of users: {self.n_user}\n Number of items: {self.m_item}")
        print(f"Number of Ratings: {self._trainDataSize + self._testDataSize}")
        print(f"{world.dataset} Rating Density: {(self._trainDataSize + self._testDataSize) / self.n_user / self.m_item}")

        # build (users,items), bipartite graph
        self.interactionGraph = None
        self.UserItemNet = csr_matrix((np.ones(len(self.train_set)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        #  user's history interacted items
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        # get test dictionary
        self._testDic = self.__build_test()
        self._coldTestDic = self.__build_cold_test()
        self._userDic, self._itemDic = self._getInteractionDic()

    @property
    def userDic(self):
        return self._userDic

    @property
    def itemDic(self):
        return self._itemDic

    @property
    def testDict(self):
        return self._testDic

    @property
    def coldTestDict(self):
        return self._coldTestDic

    @property
    def allPos(self):
        return self._allPos

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self._trainDataSize

    def getUserPosItems(self, users):
        """
        Method of get user all positive items
        Returns
        -------
        [ndarray0,...,ndarray_users]
        """
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
            # item_u = self.UserItemNet[self.UserItemNet['user'] == user]
            # item_u = item_u['item'].values
            # posItems.append(item_u)
        return posItems

    def __build_test(self):
        """
        Method of build test dictionary
        Returns
        -------
            dict: {user: [items]}
        """
        test_data = {}
        for i in range(len(self.test_set)):
            user = self.test_set['user'][i]
            item = self.test_set['item'][i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def __build_cold_test(self):
        test_data = self._testDic.copy()
        for i in list(test_data.keys()):
            try:
                if self.train_set['user'].value_counts()[i] > 20:
                    del test_data[i]
            except:
                pass
        return test_data

    def _getInteractionDic(self):
        user_interaction = {}
        item_interaction = {}

        def getDict(_set):
            for i in range(len(_set)):
                user = _set['user'][i]
                item = _set['item'][i]
                if user_interaction.get(user):
                    user_interaction[user].append(item)
                else:
                    user_interaction[user] = [item]
                if item_interaction.get(item):
                    item_interaction[item].append(user)
                else:
                    item_interaction[item] = [user]

        getDict(self.train_set)
        getDict(self.test_set)
        return user_interaction, item_interaction


class GraphDataset(PairDataset):
    def __init__(self, src):
        super(GraphDataset, self).__init__(src)
        # build (users,items), bipartite graph
        self.interactionGraph = None

    def getInteractionGraph(self):
        if self.interactionGraph is None:
            try:
                norm_adj = sp.load_npz(f'./data/preprocessed/{self.src}/interaction_adj_mat.npz')
                logging.debug("successfully loaded normalized interaction adjacency matrix")
            except IOError:
                logging.debug("generating adjacency matrix")
                start = time()
                adj_mat = sp.dok_matrix((self.n_user + self.m_item, self.n_user + self.m_item), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_user, self.n_user:] = R
                adj_mat[self.n_user:, :self.n_user] = R.T
                adj_mat = adj_mat.todok()

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                sp.save_npz(f'./data/preprocessed/{self.src}/interaction_adj_mat.npz', norm_adj)
                logging.debug(f"costing {time() - start}s, saved normalized interaction adjacency matrix")

            self.interactionGraph = _convert_sp_mat_to_sp_tensor(norm_adj)
            self.interactionGraph = self.interactionGraph.coalesce().to(world.device)
        return self.interactionGraph


class SocialGraphDataset(GraphDataset):
    def __init__(self, src):
        super(SocialGraphDataset, self).__init__(src)
        trustPath = f'./data/preprocessed/{src}/trust.txt'
        if os.path.exists(trustPath):
            self.friendNet = pd.read_csv(trustPath)
        else:
            self.friendNet = loadFriend(src)
        self.socialNet = csr_matrix((np.ones(len(self.friendNet)), (self.friendNet['user'], self.friendNet['friend'])),
                                    shape=(self.n_user, self.n_user))
        self.socialGraph = None
        logging.info(f"Number of Links: {len(self.friendNet)}")
        logging.info(f"{world.dataset} Link Density: {len(self.friendNet) / self.n_user / self.n_user}")

    def getSocialGraph(self):
        if self.socialGraph is None:
            try:
                pre_adj_mat = sp.load_npz(f'./data/preprocessed/{self.src}/social_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except IOError:
                print("generating adjacency matrix")
                start = time()
                adj_mat = self.socialNet.tolil()

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                print(f"costing {time() - start}s, saved norm_mat...")
                sp.save_npz(f'./data/preprocessed/{self.src}/social_adj_mat.npz', norm_adj)

            self.socialGraph = _convert_sp_mat_to_sp_tensor(norm_adj)
            self.socialGraph = self.socialGraph.coalesce().to(world.device)
        return self.socialGraph

    def getDenseSocialGraph(self):
        if self.socialGraph is None:
            self.socialGraph = self.getSocialGraph().to_dense()
        else:
            pass
        return self.socialGraph


def loadInteraction(src='lastfm', prepro='origin', binary=True, posThreshold=None, level='ui'):
    """
        Method of loading certain raw data
        Parameters
        ----------
        src : str, the name of dataset
        prepro : str, way to pre-process raw data input, expect 'origin', f'{N}core', f'{N}filter', N is integer value
        binary : boolean, whether to transform rating to binary label as CTR or not as Regression
        posThreshold : float, if not None, treat rating larger than this threshold as positive sample
        level : str, which level to do with f'{N}core' or f'{N}filter' operation
        (it only works when prepro contains 'core' or 'filter')

        Returns
        -------
        df : pd.DataFrame, rating information with columns: user, item, rating, (options: timestamp)
        user_num : int, the number of users
        item_num : int, the number of items
        """
    df = pd.DataFrame()
    # which dataset will use
    if src == 'lastfm':
        # user_artists.dat
        df = pd.read_csv(f'./data/raw/{src}/user_artists.dat', sep='\t')
        df.rename(columns={'userID': 'user', 'artistID': 'item', 'weight': 'rating'}, inplace=True)

    elif src == 'ciao':
        d = sio.loadmat(f'./data/raw/{src}/rating_with_timestamp.mat')
        prime = []
        for val in d['rating']:
            user, item, rating, timestamp = val[0], val[1], val[3], val[5]
            prime.append([user, item, rating, timestamp])
        df = pd.DataFrame(prime, columns=['user', 'item', 'rating', 'timestamp'])
        del prime, d, user, item, rating, timestamp
        gc.collect()
    else:
        raise ValueError('Invalid Dataset Error')

    # set rating >= threshold as positive samples
    if posThreshold is not None:
        df = df.query(f'rating >= {posThreshold}').reset_index(drop=True)

    # reset rating to interaction, here just treat all rating as 1
    if binary:
        df['rating'] = 1.0

    # which type of pre-dataset will use
    if prepro == 'origin':
        pass

    elif prepro.endswith('filter'):
        pattern = re.compile(r'\d+')
        filter_num = int(pattern.findall(prepro)[0])

        # count user's item
        tmp1 = df.groupby(['user'], as_index=False)['item'].count()
        tmp1.rename(columns={'item': 'cnt_item'}, inplace=True)
        # count item's user
        tmp2 = df.groupby(['item'], as_index=False)['user'].count()
        tmp2.rename(columns={'user': 'cnt_user'}, inplace=True)
        # add column of count numbers
        df = df.merge(tmp1, on=['user']).merge(tmp2, on=['item'])
        if level == 'ui':
            df = df.query(f'cnt_item >= {filter_num} and cnt_user >= {filter_num}').reset_index(drop=True)
        elif level == 'u':
            df = df.query(f'cnt_item >= {filter_num}').reset_index(drop=True)
        elif level == 'i':
            df = df.query(f'cnt_user >= {filter_num}').reset_index(drop=True)
        else:
            raise ValueError(f'Invalid level value: {level}')

        df.drop(['cnt_item', 'cnt_user'], axis=1, inplace=True)
        del tmp1, tmp2
        gc.collect()

    elif prepro.endswith('core'):
        pattern = re.compile(r'\d+')
        core_num = int(pattern.findall(prepro)[0])

        def filter_user(df):
            tmp = df.groupby(['user'], as_index=False)['item'].count()
            tmp.rename(columns={'item': 'cnt_item'}, inplace=True)
            df = df.merge(tmp, on=['user'])
            df = df.query(f'cnt_item >= {core_num}').reset_index(drop=True).copy()
            df.drop(['cnt_item'], axis=1, inplace=True)

            return df

        def filter_item(df):
            tmp = df.groupby(['item'], as_index=False)['user'].count()
            tmp.rename(columns={'user': 'cnt_user'}, inplace=True)
            df = df.merge(tmp, on=['item'])
            df = df.query(f'cnt_user >= {core_num}').reset_index(drop=True).copy()
            df.drop(['cnt_user'], axis=1, inplace=True)

            return df

        if level == 'ui':
            while 1:
                df = filter_user(df)
                df = filter_item(df)
                chk_u = df.groupby('user')['item'].count()
                chk_i = df.groupby('item')['user'].count()
                if len(chk_i[chk_i < core_num]) <= 0 and len(chk_u[chk_u < core_num]) <= 0:
                    break
        elif level == 'u':
            df = filter_user(df)
        elif level == 'i':
            df = filter_item(df)
        else:
            raise ValueError(f'Invalid level value: {level}')

        gc.collect()

    else:
        raise ValueError('Invalid dataset preprocess type, origin/Ncore/Nfilter (N is int number) expected')

    # encoding user_id and item_id
    userId = pd.Categorical(df['user'])
    itemId = pd.Categorical(df['item'])
    df['user'] = userId.codes
    df['item'] = itemId.codes

    userCodeDict = {int(value): code for code, value in enumerate(userId.categories.values)}
    itemCodeDict = {int(value): code for code, value in enumerate(itemId.categories.values)}

    outputPath = f"./data/preprocessed/{src}"
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    with open(f"./data/preprocessed/{src}/userReindex.json", "w") as f:
        f.write(json.dumps(userCodeDict))
    with open(f"./data/preprocessed/{src}/itemReindex.json", "w") as f:
        f.write(json.dumps(itemCodeDict))

    userNum = df['user'].nunique()
    itemNum = df['item'].nunique()

    logging.info(f'Finish loading [{src}]-[{prepro}] dataset')
    return df[['user', 'item']], userNum, itemNum


def loadFriend(src):
    path = f'./data/preprocessed/{src}/trust.txt'
    if os.path.exists(path):
        return pd.read_csv(path)

    if src == 'lastfm':
        friendNet = pd.read_csv(f'./data/raw/{src}/user_friends.dat', sep='\t')
        friendNet.rename(columns={'userID': 'user', 'friendID': 'friend'}, inplace=True)

    elif src == 'ciao':
        d = sio.loadmat(f'./data/raw/{src}/trust.mat')
        prime = []
        for val in d['trust']:
            user, friend = val[0], val[1]
            prime.append([user, friend])
        friendNet = pd.DataFrame(prime, columns=['user', 'friend'])
        del prime, d, user, friend, val
        gc.collect()
    else:
        raise ValueError('Invalid Dataset Error')

    gc.collect()
    # remove self-connection
    friendNet = friendNet[friendNet['user'] != friendNet['friend']].reset_index(drop=True)

    friendNet = renameFriendID(src, friendNet)
    friendNet.to_csv(path, index=False)
    return friendNet


def renameFriendID(src: str, friendNet: pd.DataFrame):
    with open(f"./data/preprocessed/{src}/userReindex.json") as f:
        userReindex = json.load(f)
    friendNet['user'] = friendNet.apply(lambda x: reIndex(x.user, userReindex), axis=1)
    friendNet['friend'] = friendNet.apply(lambda x: reIndex(x.friend, userReindex), axis=1)
    friendNet = friendNet.drop(friendNet[(friendNet['user'] == -1) | (friendNet['friend'] == -1)].index)
    return friendNet


def reIndex(x, userReindex):
    if str(x) in userReindex.keys():
        return userReindex[str(x)]
    else:
        return -1


def splitDataset(df, testMethod='fo', testSize=.2):
    """
    this is from https://github.com/recsys-benchmark/DaisyRec-v2.0
    method of splitting data into training data and test data
    Parameters
    ----------
    df : pd.DataFrame raw data waiting for test set splitting
    testMethod : str, way to split test set
                    'fo': split by ratio
                    'tfo': split by ratio with timestamp
                    'tloo': leave one out with timestamp
                    'loo': leave one out
                    'ufo': split by ratio in user level
                    'utfo': time-aware split by ratio in user level
    testSize : float, size of test set
    valSize : float, size of validate set

    Returns
    -------
    train_set : pd.DataFrame training dataset
    test_set : pd.DataFrame test dataset

    """

    train_set, test_set = pd.DataFrame(), pd.DataFrame()
    if testMethod == 'ufo':
        driver_ids = df['user']
        _, driver_indices = np.unique(np.array(driver_ids), return_inverse=True)
        gss = GroupShuffleSplit(n_splits=1, test_size=testSize, random_state=2020)
        for train_idx, test_idx in gss.split(df, groups=driver_indices):
            train_set, test_set = df.loc[train_idx, :].copy(), df.loc[test_idx, :].copy()

    elif testMethod == 'utfo':
        df = df.sort_values(['user', 'timestamp']).reset_index(drop=True)

        def time_split(grp):
            start_idx = grp.index[0]
            split_len = int(np.ceil(len(grp) * (1 - testSize)))
            split_idx = start_idx + split_len
            end_idx = grp.index[-1]
            return list(range(split_idx, end_idx + 1))

        test_index = df.groupby('user').apply(time_split).explode().values
        test_set = df.loc[test_index, :]
        train_set = df[~df.index.isin(test_index)]

    elif testMethod == 'tfo':
        # df = df.sample(frac=1)
        df = df.sort_values(['timestamp']).reset_index(drop=True)
        split_idx = int(np.ceil(len(df) * (1 - testSize)))
        train_set, test_set = df.iloc[:split_idx, :].copy(), df.iloc[split_idx:, :].copy()

    elif testMethod == 'fo':
        train_set, test_set = train_test_split(df, test_size=testSize, random_state=2020)

    elif testMethod == 'tloo':
        # df = df.sample(frac=1)
        df = df.sort_values(['timestamp']).reset_index(drop=True)
        df['rank_latest'] = df.groupby(['user'])['timestamp'].rank(method='first', ascending=False)
        train_set, test_set = df[df['rank_latest'] > 1].copy(), df[df['rank_latest'] == 1].copy()
        del train_set['rank_latest'], test_set['rank_latest']

    elif testMethod == 'loo':
        # # slow method
        # test_set = df.groupby(['user']).apply(pd.DataFrame.sample, n=1).reset_index(drop=True)
        # test_key = test_set[['user', 'item']].copy()
        # train_set = df.set_index(['user', 'item']).drop(pd.MultiIndex.from_frame(test_key)).reset_index().copy()

        # # quick method
        test_index = df.groupby(['user']).apply(lambda grp: np.random.choice(grp.index))
        test_set = df.loc[test_index, :].copy()
        train_set = df[~df.index.isin(test_index)].copy()

    else:
        raise ValueError('Invalid data_split value, expect: loo, fo, tloo, tfo')

    train_set, test_set = train_set.reset_index(drop=True), test_set.reset_index(drop=True)

    return train_set, test_set
