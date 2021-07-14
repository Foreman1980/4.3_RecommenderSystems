import numpy as np
import pandas as pd


def prefilter_items(data, take_n_popular=5000, item_features=None):

    set_of_excluded_items = set()
    
    # определяем все продажи с нулевым "quantity"
#     products_quantity = data.groupby('item_id')['quantity'].sum().reset_index()
#     zero_quantity_sales_items = products_quantity[products_quantity['quantity'] < 1]['item_id'].unique().tolist()
#     set_of_excluded_items.update(zero_quantity_sales_items)
    
    # определим товары, которые не продавались за последние 12 месяцев (последние 52 недели)
    old_item = list(set(data[data['week_no'] < data['week_no'].max() - 52].item_id.unique().tolist())
                    - set(data[data['week_no'] >= data['week_no'].max() - 52].item_id.unique().tolist()))
    set_of_excluded_items.update(old_item)
    
    # определим слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    # и слишком дорогие товары
#     products_price = data[['item_id', 'quantity', 'sales_value']].copy()

#     products_price['price'] = products_price['sales_value'] / np.maximum(products_price['quantity'], 1)
#     products_price = products_price.groupby('item_id')['price'].mean().reset_index()

#     noninterest_items = products_price[~products_price['price'].between(2, 100)].item_id.unique().tolist()
#     set_of_excluded_items.update(noninterest_items)
    
    # Уберем самые популярные товары (их и так купят), а также
    # самые НЕ популярные товары (их и так НЕ купят)
#     popular_items = data[~data['item_id'].isin(set_of_excluded_items)].groupby('item_id')['user_id'].nunique().reset_index()
#     popular_items.rename(columns={'user_id': 'n_users'}, inplace=True)
#     popular_items['n_users'] = popular_items['n_users'] / data[~data['item_id'].isin(set_of_excluded_items)]['user_id'].nunique()
    
#     nonpopular_items = popular_items[popular_items['n_users'].between(.03, .4)].item_id.tolist()
#     set_of_excluded_items.update(nonpopular_items)
    
    # # Уберем не интересные для рекоммендаций категории (department)
    # if item_features is not None:
    #     department_size = pd.DataFrame(item_features. \
    #                                    groupby('department')['item_id'].nunique(). \
    #                                    sort_values(ascending=False)).reset_index()

    #     department_size.columns = ['department', 'n_items']
    #     rare_departments = department_size[department_size['n_items'] < 150].department.tolist()
    #     items_in_rare_departments = item_features[
    #         item_features['department'].isin(rare_departments)].item_id.unique().tolist()

    #     data = data[~data['item_id'].isin(items_in_rare_departments)]
    
    # Возбмем топ по популярности
    popularity = data[~data['item_id'].isin(set_of_excluded_items)].groupby('item_id')['user_id'].sum().reset_index()
    popularity.rename(columns={'user_id': 'n_users'}, inplace=True)

    nontop_items = popularity.sort_values('n_users', ascending=False)[take_n_popular:].item_id.tolist()
    set_of_excluded_items.update(nontop_items)
    
    # data[~data['item_id'].isin(set_of_excluded_items)]['item_id'].nunique() == 5000 => True
    
    # Заведем фиктивный item_id (если юзер покупал товары не из топ-5000, то он "купил" такой товар)
    data.loc[data['item_id'].isin(set_of_excluded_items), 'item_id'] = 999999
    
    # ...
    return data
    
def postfilter_items(user_id, recommednations):
    pass
