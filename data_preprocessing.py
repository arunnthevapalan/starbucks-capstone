import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def prepare_portfolio(portfolio):
    """ 
    - It makes columns for the channels
    - Changes the name of the id column to offer_id
    Input:
    - portfolio: original dataset
    Returns:
    - portfolio_clean
    """
    portfolio_clean = portfolio.copy()
    # Create dummy columns for the channels column
    d_chann = pd.get_dummies(portfolio_clean.channels.apply(pd.Series).stack(),
                             prefix="channel").sum(level=0)
    portfolio_clean = pd.concat([portfolio_clean, d_chann], axis=1, sort=False)
    portfolio_clean.drop(columns='channels', inplace=True)
    # Change column name
    portfolio_clean.rename(columns={'id':'offer_id'}, inplace=True)

    return portfolio_clean


def prepare_profile(profile):
    """ 
    - Fix the date format
    - Change the column name id to customer_id
    - Create column to identify customers with demographic data
    - Add dummy columns for gender
    Input:
    - profile: original dataset
    Returns:
    - profile_clean
    """
    profile_clean = profile.copy()
    # Transform date from int to datetime
    date = lambda x: pd.to_datetime(str(x), format='%Y%m%d')
    profile_clean.became_member_on = profile_clean.became_member_on.apply(date)
    # Create column that separates customers with valida data
    profile_clean['valid'] = (profile_clean.age != 118).astype(int)
    # Change the name of id column to customer_id
    profile_clean.rename(columns={'id':'customer_id'}, inplace=True)
    # Create dummy columns for the gender column
    dummy_gender = pd.get_dummies(profile_clean.gender, prefix="gender")
    profile_clean = pd.concat([profile_clean, dummy_gender], axis=1, sort=False)
    return profile_clean


def prepare_transcript(transcript):
    """ .
    - Split value in several columns for offers and transactions
    - Split event column into sevelar columns
    - Change column name person to customer_id
    Input:
    - transcript: original dataset
    Returns:
    - transcript_clean
    """
    transcript_clean = transcript.copy()
    # Split event into several dummy columns
    transcript_clean.event = transcript_clean.event.str.replace(' ', '_')
    dummy_event = pd.get_dummies(transcript_clean.event, prefix="event")
    transcript_clean = pd.concat([transcript_clean, dummy_event], axis=1,
                                 sort=False)
    transcript_clean.drop(columns='event', inplace=True)
    # Get the offer_id data from the value column
    transcript_clean['offer_id'] = [[*v.values()][0]
                                    if [*v.keys()][0] in ['offer id',
                                                          'offer_id'] else None
                                    for v in transcript_clean.value]
    # Get the transaction amount data from the value column
    transcript_clean['amount'] = [np.round([*v.values()][0], decimals=2)
                                  if [*v.keys()][0] == 'amount' else None
                                  for v in transcript_clean.value]
    transcript_clean.drop(columns='value', inplace=True)
    # Change the name of person column to customer_id
    transcript_clean.rename(columns={'person':'customer_id'}, inplace=True)
    return transcript_clean

def merge_datasets(portfolio_clean, profile_clean, transcript_clean):
    """ Merge the three data sets into one
    Input:
    - portfolio_clean
    - profile_clean
    - transcript_clean
    Output:
    - df: merged dataframe
    """
    trans_prof = pd.merge(transcript_clean, profile_clean, on='customer_id',
                          how="left")
    df = pd.merge(trans_prof, portfolio_clean, on='offer_id', how='left')
    # Change the offer ids to a simplied form
    offer_id = {'ae264e3637204a6fb9bb56bc8210ddfd': 'B1',
                '4d5c57ea9a6940dd891ad53e9dbe8da0': 'B2',
                '9b98b8c7a33c4b65b9aebfe6a799e6d9': 'B3',
                'f19421c1d4aa40978ebb69ca19b0e20d': 'B4',
                '0b1e1539f2cc45b7b9fa7c272da2e1d7': 'D1',
                '2298d6c36e964ae4a3e7e9706d1fb8c2': 'D2',
                'fafdcd668e3743c1bb461111dcafc2a4': 'D3',
                '2906b810c7d4411798c6938adc9daaa5': 'D4',
                '3f207df678b143eea3cee63160fa8bed': 'I1',
                '5a8bc65990b245e5a138643cd4eb9837': 'I2'}
    df.offer_id = df.offer_id.apply(lambda x: offer_id[x] if x else None)

    return df