"""
Data cleaning script for Sydney Housing Price Prediction project.
Suburbs: Mosman, Epping, Blacktown
"""
import pandas as pd
import numpy as np
import re

def parse_money(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace('$', '').replace(',', '').strip()
    if s == '' or s.lower() == 'nan':
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan

def clean_yesno(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in ('yes', 'y', 'true', '1'):
        return 1
    if s in ('no', 'n', 'false', '0'):
        return 0
    return np.nan

def load_and_clean(path='data/raw_suburbs.csv'):
    df = pd.read_csv(path, encoding='latin-1')
    df = df.dropna(subset=['suburb'])
    df = df[df['suburb'].isin(['Mosman', 'Epping', 'Blacktown'])].copy()

    df['sale_price'] = df['sale_price'].apply(parse_money)
    df['prev_sale_price'] = df['prev_sale_price'].apply(parse_money)

    def parse_date(x):
        if pd.isna(x):
            return pd.NaT
        s = str(x).strip()
        for fmt in ('%d-%b-%y', '%d-%b-%Y', '%m/%d/%Y', '%m/%d/%y'):
            try:
                return pd.to_datetime(s, format=fmt)
            except (ValueError, TypeError):
                continue
        return pd.to_datetime(s, errors='coerce', dayfirst=True)

    df['sale_date'] = df['sale_date'].apply(parse_date)
    df['prev_sale_date'] = df['prev_sale_date'].apply(parse_date)

    for col in ['bedrooms', 'bathrooms', 'car_spaces', 'land_size_m2',
                'building_size_m2', 'year_built', 'Storeys']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['Pool'] = df['Pool'].apply(clean_yesno)
    df['renovated'] = df['renovated'].apply(clean_yesno)

    # Drop rows with no sale price (target) - can't use for supervised learning
    df = df.dropna(subset=['sale_price'])

    # Feature engineering
    df['sale_year'] = df['sale_date'].dt.year
    df['sale_month'] = df['sale_date'].dt.month
    df['property_age'] = df['sale_year'] - df['year_built']
    df['property_age'] = df['property_age'].where(df['property_age'] >= 0)

    df['years_since_prev_sale'] = (df['sale_date'] - df['prev_sale_date']).dt.days / 365.25
    df['price_growth_pct'] = np.where(
        df['prev_sale_price'].notna() & (df['prev_sale_price'] > 0),
        (df['sale_price'] - df['prev_sale_price']) / df['prev_sale_price'] * 100,
        np.nan
    )

    df['price_per_m2'] = df['sale_price'] / df['land_size_m2']
    df['desc_word_count'] = df['listing_description'].fillna('').apply(lambda s: len(str(s).split()))
    df['desc_length'] = df['listing_description'].fillna('').apply(len)

    # Simple sentiment/keyword proxy features from listing description
    luxury_words = ['luxury', 'stunning', 'exceptional', 'rare', 'premier', 'elegant',
                     'exquisite', 'spectacular', 'grand', 'prestigious', 'resort']
    df['luxury_word_count'] = df['listing_description'].fillna('').apply(
        lambda s: sum(w in str(s).lower() for w in luxury_words))

    # Impute remaining numeric NAs with suburb-level median (avoid leakage across suburbs)
    for col in ['bedrooms', 'bathrooms', 'car_spaces', 'building_size_m2',
                'property_age', 'Storeys']:
        df[col] = df.groupby('suburb')[col].transform(lambda s: s.fillna(s.median()))

    df['Pool'] = df['Pool'].fillna(0)
    df['renovated'] = df['renovated'].fillna(0)

    return df.reset_index(drop=True)

if __name__ == '__main__':
    df = load_and_clean()
    print(df.shape)
    print(df.isna().sum())
    df.to_csv('data/cleaned_suburbs.csv', index=False)
