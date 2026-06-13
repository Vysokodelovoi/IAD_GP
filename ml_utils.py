from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from catboost import CatBoostClassifier
import time
cat_cols = [
    'hotel', 'arrival_date_month', 'meal', 'country',
    'market_segment', 'distribution_channel', 'reserved_room_type',
    'assigned_room_type', 'deposit_type', 'customer_type', 'reservation_status',
    'country_full'
]

baseline_pipeline = Pipeline(
    [('catboost_cls', CatBoostClassifier(cat_features=cat_cols))]
)

def measure_classification(cls_pipeline, method):
    df = pd.read_parquet('main_df_edited.parquet')
    X = df.drop(columns=['is_canceled']).copy()
    y = df.is_canceled
    result = cross_val_score(cls_pipeline, X, y, cv=1, random_state=22)
    return (result, method, time.time())

# Линейные модели logistic_regression / linear, Даниэль
#  svm Игорь 
#  boosting Игорь
#  random trees Даниэль
#  нейронка Игорь
#  изотоническая регрессия Даниэль
#  ранжирование попробовать? Игорь

#  Для каждой затюнить гиперпараметры через GridSearchCV или optuna
#  Результаты по проведенным экспериментам собрать в pandas датасет и отправить
#  Сделать выводы по performance моделю, посмотреть на важность признаков через shap или feature_importances


#  Графики с дубликатами и без сравнить
