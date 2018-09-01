import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

col_clean = ['body_length', 'channels', 'delivery_method', 'fb_published', 'gts',
       'has_analytics', 'has_logo', 'listed', 'name_length', 'num_order',
       'num_payouts', 'org_facebook', 'org_twitter', 'sale_duration',
       'sale_duration2', 'show_map', 'user_age', 'sold', 'currency_AUD',
       'currency_CAD', 'currency_EUR', 'currency_GBP', 'currency_MXN',
       'currency_NZD', 'currency_USD', 'payout_type_ACH', 'payout_type_CHECK',
       'payout_type_undefined', 'public_domain_False', 'public_domain_True']

cols = ['approx_payout_date', 'body_length', 'channels', 'country',
       'currency', 'delivery_method', 'description', 'email_domain',
       'event_created', 'event_end', 'event_published', 'event_start',
       'fb_published', 'gts', 'has_analytics', 'has_header', 'has_logo',
       'listed', 'name', 'name_length', 'num_order', 'num_payouts',
       'object_id', 'org_desc', 'org_facebook', 'org_name', 'org_twitter',
       'payee_name', 'payout_type', 'previous_payouts', 'sale_duration',
       'sale_duration2', 'show_map', 'ticket_types', 'user_age',
       'user_created', 'user_type', 'venue_address', 'venue_country',
       'venue_latitude', 'venue_longitude', 'venue_name', 'venue_state',
       'sold', 'public_domain']

money = ['currency_AUD','currency_CAD', 'currency_EUR', 'currency_GBP',
             'currency_MXN', 'currency_NZD', 'currency_USD']


label = {'label': {'fraudster_event': True, 'premium': False, 'spammer_warn': False, 'fraudster': True,
                   'spammer_limited': False, 'spammer_noinvite': False, 'locked': False, 'tos_lock': False,
                   'tos_warn': False, 'fraudster_att': True, 'spammer_web': False, 'spammer': False}}

columns = ['approx_payout_date', 'country', 'description', 'email_domain', 'event_created', 'event_end',
           'event_published', 'event_start', 'name', 'object_id', 'org_desc',
           'org_name', 'payee_name', 'previous_payouts', 'ticket_types', 'user_created', 'user_type',
           'venue_address', 'venue_country',
           'venue_latitude', 'venue_longitude', 'venue_name', 'venue_state', 'has_header']

def metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    matrix = np.array([[tp, fp], [fn, tn]])
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    accuracy = (tp + tn)/(tn+fp+fn+tp)
    return precision, recall, accuracy, matrix

def total_sold(series):
    """
    This function will return the total quantity
    sold by a user
    """
    total_sold = 0
    for dic in series:
        total_sold += dic['quantity_sold']
    return total_sold


def initialize_cols(df, cols):

    for col in cols:
        lst = df.columns
        if col not in lst:
            df[col] = 0
    return df


def currency(df):
    for country in money:
        try:
            df[country]
        except:
            df[country] = 0

    return df

def clean_data(df):
    """
    Creates a clean table with all numeric values and boolean
    input:

    df: DataFrame
    label: dict of dict with labels as keys and values as dict where keys
    are values of that the column and values are the categorical conversions.
    drop_list: list of column names to be dropped

    return
    DataFrame of numeric and boolean values
    """
    df = initialize_cols(df, cols)
    df.rename(index=str, columns={"acct_type": "label"}, inplace=True)
    try:
        df['sold'] = list(map(total_sold, df['ticket_types']))
    except:
        df['sold'] = 0
    try:
        df['public_domain'] = (df['email_domain'] == 'gmail.com') | (df['email_domain'] == 'yahoo.com') | (df['email_domain'] == 'hotmail.com')
    except:
        df['public_domain'] = 0
    try:
        df.replace(label, inplace=True)
    except:
        df['label'] = 0
    df.replace({'listed': {'y': 1, 'n': 0}}, inplace=True)
    df.replace({"payout_type": {"": "undefined"}}, inplace=True)
    df = pd.get_dummies(df, columns=["currency", "payout_type", "public_domain"])
    df = currency(df)
    df.drop(columns=columns, axis='columns', inplace=True)
    df.dropna(axis="index", inplace=True)
    df = initialize_cols(df, col_clean)
    return df
