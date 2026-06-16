from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np


class FeatureEngineer(BaseEstimator, TransformerMixin):

    def __init__(self, country_threshold=0.01):
        self.country_threshold = country_threshold
        

    def fit(self, X, y=None):
        country_freq = X['country'].value_counts(normalize=True)
        self.feature_names_in_ = X.columns
        self.top_countries_ = set(
            country_freq[country_freq >= self.country_threshold].index
        )
        X_out = self.transform(X.head(1))
        self.feature_names_out_ = X_out.columns.to_numpy()
        return self

    def transform(self, X):
        X = X.copy()

        # stay features
        X['total_nights'] = (
            X['stays_in_weekend_nights']
            + X['stays_in_week_nights']
        )

        X['is_weekend_stay'] = (
            X['stays_in_weekend_nights'] > 0
        ).astype(int)

        # guest features
        X['total_guests'] = (
            X['adults']
            + X['children']
            + X['babies']
        )

        X['has_children'] = (
            (X['children'] + X['babies']) > 0
        ).astype(int)

        X['booking_intensity'] = (
            X['total_guests']
            * X['total_nights']
        )

        # missing indicators
        X['has_agent'] = X['agent'].notna().astype(int)
        X['has_company'] = X['company'].notna().astype(int)

        # country grouping
        X['country_group'] = np.where(
            X['country'].isin(self.top_countries_),
            X['country'],
            'Other'
        )

        # drop columns
        X = X.drop(
            columns=[
                'agent',
                'company',
                'country',
                'reservation_status',
                'reservation_status_date',
                'assigned_room_type'
            ],
            errors='ignore'
        )
        return X
    

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_
    

num_features = [
    'lead_time', 'arrival_date_year', 'arrival_date_week_number',
    'arrival_date_day_of_month', 'stays_in_weekend_nights',
    'stays_in_week_nights', 'adults', 'children', 'babies',
    'is_repeated_guest', 'previous_cancellations',
    'previous_bookings_not_canceled', 'booking_changes',
    'days_in_waiting_list', 'required_car_parking_spaces',
    'total_of_special_requests'
]
# After FE
num_features += [
    'total_nights',
    'is_weekend_stay',
    'total_guests',
    'has_children',
    'booking_intensity',
    'has_agent',
    'has_company'
]

cat_features = [
    'hotel', 'arrival_date_month', 'meal', 
    'market_segment', 'distribution_channel', 'reserved_room_type', 'deposit_type', 'customer_type'
]
# After FE
cat_features += ['country_group']

encoding_and_impute = ColumnTransformer(
    [
        ('categorial_oh', OneHotEncoder(handle_unknown='ignore'), cat_features),
        ('numeric', 
            Pipeline([
                ('cat_impute', KNNImputer()),
                ('scaler', StandardScaler())
                ]), num_features)
    ]
)
default_data_preparation_pipeline = Pipeline([
    ('Feature Engineering', FeatureEngineer()),
    ('feature encoding', encoding_and_impute)] 
)