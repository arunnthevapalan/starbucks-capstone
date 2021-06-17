import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from eda import *

def get_most_popular_offers(customers, n_top=2, q=0.5, offers=None):
    """ Sort offers based on the ones that result in the highest net_expense
    Input:
    - customers: dataframe with aggregated data of the offers
    - n_top: number of offers to be returned (default: 2)
    - q: quantile used for sorting
    - offers: list of offers to be sorted
    Returns:
    - sorted list of offers, in descending order according to the median net_expense
    """
    if not offers:
        offers = ['I1', 'I2', 'B1', 'B2', 'B3',
                  'B4', 'D1', 'D2', 'D3', 'D4']
    offers.sort(key=lambda x: get_net_expense(customers, x, q), reverse=True)
    offers_dict = {o: get_net_expense(customers, o, q) for o in offers}
    return offers[:n_top], offers_dict


def get_most_popular_offers_filtered(customers, n_top=2, q=0.5, income=None,
                                     age=None, gender=None):
    """ Sort offers based on the ones that result in the highest net_expense
    Input:
    - customers: dataframe with aggregated data of the offers
    - n_top: number of offers to be returned (default: 2)
    - income_range: tuple with min and max income
    - age_range: tuple with min and max age
    - gender:  'M', 'F', or 'O'
    Returns:
    - sorted list of offers, in descending order according to the
    median net_expense
    """
    flag = (customers.valid == 1)
    if income:
        income_gr = round_income(income)
        if income_gr > 0:
            flag = flag & (customers.income_group == income_gr)
    if age:
        age_gr = round_age(age)
        if age_gr > 0:
            flag = flag & (customers.age_group == age_gr)
    if gender:
        flag = flag & (customers.gender == gender)
    return get_most_popular_offers(customers[flag], n_top, q)

def get_net_expense(customers, offer, q=0.5):
    """ Get the net_expense for customers that viewed and completed and offer
    Input:
    - offer: offer of interest
    - q: quantile to be used
    Returns:
    - net_expense median
    """
    flag = (customers['{}_viewed'.format(offer)] > 0)
    flag = flag & (customers.net_expense > 0)
    flag = flag & (customers.total_transactions >= 5)
    if offer not in ['I1', 'I2']:
        flag = flag & (customers['{}_completed'.format(offer)] > 0)
    return customers[flag].net_expense.quantile(q)



