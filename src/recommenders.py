import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix #, coo_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight #, tfidf_weight

# from src.utils import prefilter_items
# from src.metrics import precision_at_k

class MainRecommender:
    """Рекоммендации, которые можно получить из ALS
    
    Input
    -----
    data: pd.DataFrame
        Матрица взаимодействий user-item (факта покупки)
        
    weighting: bool, default True
        Флаг для выполнения взвешивания данных (BM25)
    """
    
    def __init__(self, data:pd.DataFrame, weighting:bool=True):
        # Топ покупок по всему датасету
        self.overall_top_purchases = data.groupby('item_id')['quantity'].count().reset_index()
        self.overall_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.overall_top_purchases = self.overall_top_purchases[self.overall_top_purchases['item_id'] != 999999]
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()
        
        self.user_item_matrix = self._prepare_matrix(data)  # pd.DataFrame
        
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self._prepare_dicts(self.user_item_matrix)
        self.sparse_user_item = csr_matrix(self.user_item_matrix) # csr-matrix
        
        if weighting:
            self.sparse_user_item = bm25_weight(self.sparse_user_item.T).T.tocsr()

        self.model = self.fit(self.sparse_user_item)
        
        self.own_recommender = self.fit_own_recommender(self.sparse_user_item)

    @staticmethod
    def _prepare_matrix(data:pd.DataFrame) -> pd.DataFrame:
        user_item_matrix = pd.pivot_table(data, index='user_id', columns='item_id',
                                          values='quantity', # Можно пробовать другие варианты
                                          aggfunc='count', 
                                          fill_value=0)

        user_item_matrix = user_item_matrix.astype(float) # необходимый тип матрицы для implicit
        
        return user_item_matrix
    
    @staticmethod
    def _prepare_dicts(user_item_matrix:pd.DataFrame) -> (dict, dict, dict, dict):
        """Подготавливает вспомогательные словари"""
        
        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))
        
        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id
     
    def _extend_with_top_popular(self, recommendations, N=5):
        """Если кол-во рекоммендаций < N, то дополняем их топ-популярными"""

        if len(recommendations) < N:
            recommendations.extend(self.overall_top_purchases[:N])
            recommendations = recommendations[:N]

        return recommendations
        
    def get_top_n_users_items(self, user_id, N=5):
        df = self.user_item_matrix.loc[user_id, :].reset_index().groupby('item_id').sum()
        df = df.loc[df.index != 999999, :]
        if N > 0:
            df = df.sort_values(by=user_id, ascending=False).head(N).reset_index()
            return df['item_id'].tolist()
        elif N == -1:
            df = df.sort_values(by=user_id, ascending=False).reset_index()
            df = df[df[user_id] > 0]
            return df['item_id'].tolist()
    
    @staticmethod
    def fit(sparse_user_item:csr_matrix,
            n_factors:int=100, regularization:float=0.05,
            iterations:int=20, num_threads:int=4) -> 'ALS_Model':
        """Обучает ALS"""
        
        model = AlternatingLeastSquares(factors=n_factors, 
                                        regularization=regularization,
                                        iterations=iterations,  
                                        num_threads=num_threads, # calculate_training_loss=True,
                                        random_state=1234)
        
        model.fit(sparse_user_item.T.tocsr(),
                  show_progress=False)
        
        return model
    
    @staticmethod
    def fit_own_recommender(sparse_user_item:csr_matrix) -> 'ItemItem_Model':
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
    
        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(sparse_user_item.T.tocsr(),
                            show_progress=False)
        
        return own_recommender
    
    def _update_dict(self, user_id):
        """Если появился новый user / items, то нужно обновить словари"""
        
        if user_id not in self.userid_to_id.keys():
            max_id = max(list(self.userid_to_id.values()))
            max_id += 1
            
            self.userid_to_id.update({user_id: max_id})
            self.id_to_userid.update({max_id: user_id})
    
    def _get_recommendations(self, user_id, model, N:int=5):
        self._update_dict(user_id=user_id)
        recs = [self.id_to_itemid[rec[0]] for rec in model.recommend(userid=self.userid_to_id[user_id],
                                                                    user_items=self.sparse_user_item,
                                                                    N=N,
                                                                    filter_already_liked_items=False,
                                                                    filter_items=[self.itemid_to_id[999999]],
                                                                    recalculate_user=False)]
        recs = self._extend_with_top_popular(recs, N=N)
        assert len(recs) == N, 'Количество рекомендаций != {}'.format(N)
        return recs
        
    def get_als_recommendations(self, user_id:int, N:int=5) -> list:
        """Рекомендации через стандартные библиотеки implicit"""
        
        self._update_dict(user_id=user_id)
        return self._get_recommendations(user_id, model=self.model, N=N)
    
    def get_own_recommendations(self, user_id:int, N:int=5) -> list:
        """Рекомендуем товары среди тех, которые юзер уже купил"""
        
        self._update_dict(user_id=user_id)
        return self._get_recommendations(user_id, model=self.own_recommender, N=N)
    
    def get_similar_items_recommendation(self, user_id:int, N:int=5) -> list:
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        item_ids = self.get_top_n_users_items(user_id, N=N)
        recs = []
        for item in item_ids:
            recs.append(self.id_to_itemid[self.model.similar_items(self.itemid_to_id[item], N=2)[1][0]])

        assert len(recs) == N, 'Количество рекомендаций != {}'.format(N)
        return recs
    
    def get_similar_users_recommendation(self, user_id:int, N:int=5) -> list:
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        top_n_nearest_users = [self.id_to_userid[uidx] for uidx, _ in self.model.similar_users(self.userid_to_id[user_id], N=5+1)[1:]]
    #     print(top_n_nearest_users)

        top_items_nearest_users = []
        for uidx in top_n_nearest_users:
            top_items_nearest_users.append(self.get_top_n_users_items(uidx, N=-1))
    #     print([len(lst) for lst in top_items_nearest_users])

        recs = []
#         cycles = 0
        while len(recs) < N and top_items_nearest_users:            
            cycles = len(top_items_nearest_users)
            for i in range(cycles):
                if top_items_nearest_users[i]:
                    item = top_items_nearest_users[i].pop(0)
                    if len(recs) < N and item not in recs:
                        recs.append(item)
            
            top_items_nearest_users = [lst for lst in top_items_nearest_users if lst]
        
        recs = self._extend_with_top_popular(recs, N=N)

        assert len(recs) == N, 'Количество рекомендаций != {}'.format(N)
        return recs
