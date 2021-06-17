import numpy as np
import pandas as pd


def prefilter_items(data):

    # исключим из данных продажи с нулевым "quantity"
    data = data[data['quantity'] >= 1]
    
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)
    popularity['share_unique_users'] = popularity['share_unique_users'] / data['user_id'].nunique()
    
    # Уберем самые популярные товары (их и так купят), а также
    # самые НЕ популярные товары (их и так НЕ купят)
    popular = popularity[popularity['share_unique_users'].between(.03, .5)].item_id.tolist()
    data = data[~data['item_id'].isin(popular)]
    
    # Уберем товары, которые не продавались за последние 12 месяцев (последние 52 недели)
    old_item = list(set(data[data['week_no'] < data['week_no'].max() - 52].item_id.unique().tolist())\
                    - set(data[data['week_no'] >= data['week_no'].max() - 52].item_id.unique().tolist()))
    data = data[~data['item_id'].isin(old_item)] # 140 позиций
    
    # Уберем не интересные для рекоммендаций категории (department)
    # пока ничего не будем убирать    
    
    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    # И слишком дорогие товары
    products_price = data[['item_id', 'quantity', 'sales_value']]
    products_price['price'] = products_price['sales_value'] / products_price['quantity']
    products_price = products_price.groupby('item_id')['price'].mean().reset_index()
    interest_items = products_price[products_price['price'].between(1, 150)].item_id.unique().tolist()
    data = data[data['item_id'].isin(interest_items)]
    
    # ...
    return data
    
def postfilter_items(user_id, recommednations):
    pass
