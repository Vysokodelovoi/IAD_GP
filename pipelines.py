from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np


class FeatureEngineer(BaseEstimator, TransformerMixin):

    def __init__(self, country_threshold=0.01):
        self.country_threshold = country_threshold

    def fit(self, X, y=None):
        country_freq = X['country'].value_counts(normalize=True)

        self.top_countries_ = set(
            country_freq[country_freq >= self.country_threshold].index
        )

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
    
impute_pipieline = Pipeline([]) # Not implemented

default_data_preparation_pipeline = Pipeline(
    [('Impute part', 'pipelines'), # change to impute, when implemented
    ('Feature Engineering', FeatureEngineer()),
    ('feature encoding', 'passthrough')] # change to encoding when implemented
)