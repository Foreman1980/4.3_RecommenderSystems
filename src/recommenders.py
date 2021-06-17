import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight

# from src.utils import prefilter_items
# from src.metrics import precision_at_k

class MainRecommender:
    """Рекоммендации, которые можно получить из ALS
    
    Input
    -----
    user_item_matrix: pd.DataFrame !!! возможно ошибка - нужен просто датасет взаимодействий (первоначальный - data)
        Матрица взаимодействий user-item
        
    weighting: bool, default True
        Флаг для выполнения взвешивания данных (BM25)
    """
    
    def __init__(self, data:pd.DataFrame, weighting:bool=True):
        
        # your_code. Это не обязательная часть. Но если вам удобно что-либо посчитать тут - можно это сделать

        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        
        self.id_to_itemid, self.id_to_userid,
        self.itemid_to_id, self.userid_to_id = prepare_dicts(self.user_item_matrix)

        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T 

        self.model = self.fit(self.user_item_matrix)
        
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

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
     
    @staticmethod    
    def get_top_n_users_items(user_id:int, N:int=5) -> list:
        df = self.user_item_matrix.loc[user_id, :].reset_index().groupby('item_id').sum().sort_values(by=user_id, ascending=False).head(N)
        return df.reset_index()['item_id'].tolist() 
    
    @staticmethod
    def fit(user_item_matrix:pd.DataFrame, n_factors:int=20, regularization:float=0.001, iterations:int=15, num_threads:int=4) -> 'ALS_Model':
        """Обучает ALS"""
        
        model = AlternatingLeastSquares(factors=factors, 
                                        regularization=regularization,
                                        iterations=iterations,  
                                        num_threads=num_threads)
        
        model.fit(csr_matrix(self.user_item_matrix).T.tocsr())
        
        return model

    @staticmethod
    def fit_own_recommender(user_item_matrix:pd.DataFrame) -> 'ItemItem_Model':
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
    
        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return own_recommender
    
    def get_similar_items_recommendation(self, user_id:int, N:int=5) -> list:
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
        # your_code
        items_id = self.get_top_n_users_items(user_id, N=N)
        rec = []
        for item in items_id:
            rec.append(self.model.similar_items(itemid_to_id[item], N=2)[1][0])
#         rec = [id_to_itemid[model.similar_items(itemid_to_id[item], N=2)[1][0]] for item in items_id]

        assert len(rec) == N, 'Количество рекомендаций != {}'.format(N)
        return rec
    
    def get_similar_users_recommendation(self, user_id:int, N:int=5) -> list:
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
    
        # your_code

        assert len(rec) == N, 'Количество рекомендаций != {}'.format(N)
        return rec
