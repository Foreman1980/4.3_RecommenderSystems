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
        
        # your_code. Это не обязательная часть. Но если вам удобно что-либо посчитать тут - можно это сделать

        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.sparse_user_item = csr_matrix(self.user_item_matrix)
        
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)

        if weighting:
            self.bm25_user_item_matrix = bm25_weight(self.sparse_user_item.T).T # csr-matrix

        self.model = self.fit(self.bm25_user_item_matrix)
        
        self.own_recommender = self.fit_own_recommender(self.bm25_user_item_matrix)

    @staticmethod
    def prepare_matrix(data:pd.DataFrame) -> pd.DataFrame:
        # your_code
        user_item_matrix = pd.pivot_table(data, index='user_id', columns='item_id',
                                          values='quantity', # Можно пробовать другие варианты
                                          aggfunc='count', 
                                          fill_value=0
                                         )

        user_item_matrix = user_item_matrix.astype(float) # необходимый тип матрицы для implicit
        
        return user_item_matrix
    
    @staticmethod
    def prepare_dicts(user_item_matrix:pd.DataFrame) -> (dict, dict, dict, dict):
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
     
    def get_top_n_users_items(self, user_id, N=5):
        df = self.user_item_matrix.loc[user_id, :].reset_index().groupby('item_id').sum()
        if N > 0:
            df = df.sort_values(by=user_id, ascending=False).head(N).reset_index()
            return df['item_id'].tolist()
        elif N == -1:
            df = df.sort_values(by=user_id, ascending=False).reset_index()
            df = df[df[user_id] > 0]
            return df['item_id'].tolist()
    
    @staticmethod
    def fit(bm25_user_item_matrix:csr_matrix,
            n_factors:int=20, regularization:float=0.001,
            iterations:int=15, num_threads:int=4
           ) -> 'ALS_Model':
        
        """Обучает ALS"""
        
        model = AlternatingLeastSquares(factors=n_factors, 
                                        regularization=regularization,
                                        iterations=iterations,  
                                        num_threads=num_threads,
                                        calculate_training_loss=True,
                                        random_state=1234)
        
        model.fit(bm25_user_item_matrix.T.tocsr(),
                 show_progress=False)
        
        return model

    @staticmethod
    def fit_own_recommender(bm25_user_item_matrix:csr_matrix) -> 'ItemItem_Model':
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
    
        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(bm25_user_item_matrix.T.tocsr(),
                 show_progress=False)
        
        return own_recommender
    
    def get_similar_items_recommendation(self, user_id:int, N:int=5) -> list:
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
        # your_code
        items_id = self.get_top_n_users_items(user_id, N=N)
        rec = []
        for item in items_id:
            rec.append(self.id_to_itemid[self.model.similar_items(self.itemid_to_id[item], N=2)[1][0]])

        assert len(rec) == N, 'Количество рекомендаций != {}'.format(N)
        return rec
    
    def get_similar_users_recommendation(self, user_id:int, N:int=5) -> list:
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
        # your_code
        top_n_nearest_users = [self.id_to_userid[uidx] for uidx, _ in self.model.similar_users(self.userid_to_id[user_id], N=5+1)[1:]]
    #     print(top_n_nearest_users)

        top_items_nearest_users = []
        for uidx in top_n_nearest_users:
            top_items_nearest_users.append(self.get_top_n_users_items(uidx, N=-1))
    #     print([len(lst) for lst in top_items_nearest_users])

        rec = []
        while len(rec) < N:
            for i in range(len(top_items_nearest_users)):
                item = top_items_nearest_users[i].pop(0)
                if len(rec) < N and item not in rec:
                    rec.append(item)

        assert len(rec) == N, 'Количество рекомендаций != {}'.format(N)
        return rec

