import gc
import json
import os
import re
from time import time
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

        self.train_set = pd.read_csv(f'./data/preprocessed/{src}/train_set.txt')
        self.test_set = pd.read_csv(f'./data/preprocessed/{src}/test_set.txt')

        self.n_user = pd.concat([self.train_set, self.test_set])['user'].nunique()
        self.m_item = pd.concat([self.train_set, self.test_set])['item'].nunique()
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
        test_data = {}
        for i in range(len(self.test_set)):
            user = self.test_set['user'][i]
            item = self.test_set['item'][i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        for i in list(test_data.keys()):
            if self.train_set['user'].value_counts()[i] > 20:
                del test_data[i]
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


class GraphDataset:
    def __init__(self, src="lastfm"):
        self.src = src
        self.train_set = pd.read_csv(f'./data/preprocessed/{src}/train_set.txt')
        self.test_set = pd.read_csv(f'./data/preprocessed/{src}/test_set.txt')
        self.friendNet = pd.read_csv(f'./data/preprocessed/{src}/trust.txt')
        self.n_user = pd.concat([self.train_set, self.test_set])['user'].nunique()
        self.m_item = pd.concat([self.train_set, self.test_set])['item'].nunique()
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
        test_data = {}
        for i in range(len(self.test_set)):
            user = self.test_set['user'][i]
            item = self.test_set['item'][i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        for i in list(test_data.keys()):
            if self.train_set['user'].value_counts()[i] > 20:
                del test_data[i]
        return test_data

    def getInteractionGraph(self):
        print("loading adjacency matrix")
        if self.interactionGraph is None:
            try:
                pre_adj_mat = sp.load_npz(f'./data/preprocessed/{self.src}/interaction_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except IOError:
                print("generating adjacency matrix")
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
                print(f"costing {time() - start}s, saved norm_mat...")
                sp.save_npz(f'./data/preprocessed/{self.src}/interaction_adj_mat.npz', norm_adj)

            self.interactionGraph = _convert_sp_mat_to_sp_tensor(norm_adj)
            self.interactionGraph = self.interactionGraph.coalesce().to(world.device)
        return self.interactionGraph

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


class SocialGraphDataset(GraphDataset):
    def __init__(self, src):
        super(SocialGraphDataset, self).__init__(src)
        self.friendNet = pd.read_csv(f'./data/preprocessed/{src}/trust.txt')
        self.socialNet = csr_matrix((np.ones(len(self.friendNet)), (self.friendNet['user'], self.friendNet['friend'])),
                                    shape=(self.n_user, self.n_user))
        self.socialGraph = None
        print(f"Number of Links: {len(self.friendNet)}")
        print(f"{world.dataset} Link Density: {len(self.friendNet) / self.n_user / self.n_user}")

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
