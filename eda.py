import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def per_customer_data(df, profile):
    """ Build a dataframe with aggregated purchase and offer data and demographics
    Input:
    - df: merged dataframe with transactions, customer and offer data
    Output:
    - customer: dataframe with aggregated data
    """
    cust_dict = dict()
    # Get total transaction data
    transactions = df[df.event_transaction == 1].groupby('customer_id')
    cust_dict['total_expense'] = transactions.amount.sum()
    cust_dict['total_transactions'] = transactions.amount.count()
    # Get  aggr offer data
    cust_dict.update(get_offer_cust(df))
    # Get offer type data
    for ot in ['bogo', 'discount', 'informational']:
        cust_dict.update(get_offer_cust(df, ot))
    # Get offer id data
    for oi in ['B1', 'B2', 'B3', 'B4', 'D1', 'D2', 'D3', 'D4', 'I1', 'I2']:
        cust_dict.update(get_offer_id_cust(df, oi))

    customers = pd.concat(cust_dict.values(), axis=1, sort=False);
    customers.columns = cust_dict.keys()
    customers.fillna(0, inplace=True)

    # Add demographic data
    customers = pd.merge(customers, profile.set_index('customer_id'),
                         left_index=True, right_index=True)
    customers['age_group'] = customers.age.apply(round_age)
    customers['income_group'] = customers.income.apply(round_income)
    customers['net_expense'] = customers['total_expense'] - customers['reward']

    return customers

def get_offer_cust(df, offer_type=None):
    """
    Get offer data (received, viewed and completed) per customer and
    offer type
    Inputs:
    - df: dataframe of merged transactions, portfolio and profile data
    - offer_type: informational, bogo or discount
    Output:
    - aggregated data per customer and offer type
    """
    data = dict()
    for e in ['received', 'viewed', 'completed']:
        # Informational offers don't have completed data
        if offer_type == 'informational' and e == 'completed':
            continue
        flag = (df['event_offer_{}'.format(e)] == 1)
        key = e
        if offer_type:
            flag = flag & (df.offer_type == offer_type)
            key = '{}_'.format(offer_type) + key
        data[key] = df[flag].groupby('customer_id').offer_id.count()
    # Informational offers don't have reward data
    flag = (df.event_offer_completed == 1)
    if offer_type != 'informational':
        key = 'reward'
        if offer_type:
            flag = flag & (df.offer_type == offer_type)
            key = '{}_'.format(offer_type) + key
        data[key] = df[flag].groupby('customer_id').reward.sum()

    return data


def get_offer_id_cust(df, offer_id):
    """
    Get offer data (received, viewed and completed) per customer
    and offer id
    Inputs:
    - df: dataframe of merged transactions, portfolio and profile data
    - offer_id: B1, B2, ...
    Output:
    - aggregated data per customer and offer id
    """
    data = dict()

    for e in ['received', 'viewed', 'completed']:
        # Informational offers don't have completed data
        if offer_id in ['I1', 'I2'] and e == 'completed':
            continue
        event = 'event_offer_{}'.format(e)
        flag = (df[event] == 1) & (df.offer_id == offer_id)
        key = '{}_{}'.format(offer_id, e)
        data[key] = df[flag].groupby('customer_id').offer_id.count()

    # Informational offers don't have reward data
    flag = (df.event_offer_completed == 1) & (df.offer_id == offer_id)
    if offer_id not in ['I1', 'I2']:
        key = '{}_reward'.format(offer_id)
        data[key] = df[flag].groupby('customer_id').reward.sum()

    return data

def plot_offer_expense(customers, offer):
    """ Plot the histograms of the total expense and the average
    expense per transaction incurred by customers that have received,
    viewed and completed an offer.
    Input:
    - customers: dataframe with aggregated data of the offers
    - offer: offer of interest
    """
    rcv, vwd, cpd = get_offer_stat(customers, 'total_expense', offer)
    rcv_avg, vwd_avg, cpd_avg = get_average_expense(customers, offer)

    plt.figure(figsize=(16, 5))
    bins = 100

    plt.subplot(121)
    plt.hist(rcv, bins, alpha=0.5, label='{}-received'.format(offer))
    plt.hist(vwd, bins, alpha=0.5, label='{}-viewed'.format(offer))
    if offer not in ['informational', 'I1', 'I2']:
        plt.hist(cpd, bins, alpha=0.5, label='{}-completed'.format(offer))
    plt.legend(loc='best')
    ax = plt.gca();
    ax.set_xlim(0, 600);
    plt.title('Total Transaction ($)')
    plt.grid();

    plt.subplot(122)
    plt.hist(rcv_avg, bins, alpha=0.5, label='{}-received'.format(offer))
    plt.hist(vwd_avg, bins, alpha=0.5, label='{}-viewed'.format(offer))
    if offer not in ['informational', 'I1', 'I2']:
        plt.hist(cpd_avg, bins, alpha=0.5, label='{}-completed'.format(offer))
    plt.legend(loc='best')
    ax = plt.gca();
    ax.set_xlim(0, 50);
    plt.title('Average Transaction ($)')
    plt.grid();

def plot_offer_expense_by(customers, offer):
    """ Plot the total expense and the average expense per transaction
    incurred by customers that have received, viewed and completed an offer.
    The plots are separated by age, income and gender.
    Input:
    - customers: dataframe with aggregated data of the offers
    - offer: offer of interest
    """
    rcv_by = dict()
    vwd_by = dict()
    cpd_by = dict()
    rcv_avg_by = dict()
    vwd_avg_by = dict()
    cpd_avg_by = dict()

    for key in ['age_group', 'income_group', 'gender']:
        rcv_by[key], vwd_by[key], cpd_by[key] = get_offer_stat_by(customers,
                                                                  'net_expense',
                                                                  offer, key,
                                                                  aggr='mean')
        by_data = get_average_expense_by(customers, offer, key)
        rcv_avg_by[key], vwd_avg_by[key], cpd_avg_by[key] = by_data

    plt.figure(figsize=(16, 10))

    plt.subplot(231)
    plt.plot(rcv_by['age_group'], label='{}-received'.format(offer))
    plt.plot(vwd_by['age_group'], label='{}-viewed'.format(offer))
    if offer not in ['informational', 'I1', 'I2']:
        plt.plot(cpd_by['age_group'], label='{}-completed'.format(offer))
    plt.legend(loc='best')
    plt.title('Net Expense');
    plt.grid();

    plt.subplot(232)
    plt.plot(rcv_by['income_group'], label='{}-received'.format(offer))
    plt.plot(vwd_by['income_group'], label='{}-viewed'.format(offer))
    if offer not in ['informational', 'I1', 'I2']:
        plt.plot(cpd_by['income_group'], label='{}-completed'.format(offer))
    plt.legend(loc='best')
    plt.title('Net Expense');
    plt.grid();

    index = np.array([0, 1, 2])
    bar_width = 0.3
    plt.subplot(233)
    plt.bar(index, rcv_by['gender'].reindex(['M', 'F', 'O']),
            bar_width, label='{}-received'.format(offer))
    plt.bar(index + bar_width, vwd_by['gender'].reindex(['M', 'F', 'O']),
            bar_width, label='{}-viewed'.format(offer))
    if offer not in ['informational', 'I1', 'I2']:
        plt.bar(index + 2*bar_width, cpd_by['gender'].reindex(['M', 'F', 'O']),
                bar_width, label='{}-completed'.format(offer))
    plt.grid();
    plt.legend(loc='best');
    plt.title('Net Expense');
    plt.xticks(index + bar_width, ('M', 'F', 'O'));

    plt.subplot(234)
    plt.plot(rcv_avg_by['age_group'], label='{}-received'.format(offer))
    plt.plot(vwd_avg_by['age_group'], label='{}-viewed'.format(offer))
    if offer not in ['informational', 'I1', 'I2']:
        plt.plot(cpd_avg_by['age_group'], label='{}-completed'.format(offer))
    plt.legend(loc='best')
    plt.title('Average Transaction Value');
    plt.grid();

    plt.subplot(235)
    plt.plot(rcv_avg_by['income_group'], label='{}-received'.format(offer))
    plt.plot(vwd_avg_by['income_group'], label='{}-viewed'.format(offer))
    if offer not in ['informational', 'I1', 'I2']:
        plt.plot(cpd_avg_by['income_group'], label='{}-completed'.format(offer))
    plt.legend(loc='best')
    plt.title('Average Transaction Value');
    plt.grid();

    plt.subplot(236)
    plt.bar(index, rcv_avg_by['gender'].reindex(['M', 'F', 'O']), bar_width,
            label='{}-received'.format(offer))
    plt.bar(index + bar_width, vwd_avg_by['gender'].reindex(['M', 'F', 'O']),
            bar_width, label='{}-viewed'.format(offer))
    if offer not in ['informational', 'I1', 'I2']:
        plt.bar(index+2*bar_width, cpd_avg_by['gender'].reindex(['M', 'F', 'O']),
                bar_width, label='{}-completed'.format(offer))
    plt.grid();
    plt.legend(loc='best');
    plt.title('Average Transaction Value');
    plt.xticks(index + bar_width, ('M', 'F', 'O'));

def round_age(x):
    """
    Round age to the 5th of each 10th (15, 25,..., 105)
    Input:
    - x: age
    Output:
    - rounded age. Returns 0 if the value is less than 15 or more than 105
    """
    for y in range(15, 106, 10):
        if x >= y and x < y+10:
            return y
    return 0


def round_income(x):
    """
    Round income to the lower 10000th
    Intput:
    - income
    Output:
    - lower 10000th of the income. Return 0 if the income
    is less than 30,000 or more than 120,000
    """
    for y in range(30, 130, 10):
        if x >= y*1000 and x < (y+10)*1000:
            return y*1000
    return 0

def get_offer_stat(customers, stat, offer):
    """ Get any column for customers that received but not viewed an offer,
    viewed but not completed the offer, and those that viewed and completed
    the offer
    Input:
    - customers: dataframe with aggregated data of the offers
    - stat: column of interest
    - offer: offer of interest
    Output:
    - (received, viewed, completed): tuple with the corresponding column
    """
    valid = (customers.valid == 1)
    rcv_col = '{}_received'.format(offer)
    vwd_col = '{}_viewed'.format(offer)
    received = valid & (customers[rcv_col] > 0) & (customers[vwd_col] == 0)
    cpd = None
    if offer not in ['informational', 'I1', 'I2']:
        cpd_col = '{}_completed'.format(offer)
        viewed = valid & (customers[vwd_col] > 0) & (customers[cpd_col] == 0)
        completed = valid & (customers[vwd_col] > 0) & (customers[cpd_col] > 0)
        cpd = customers[completed][stat]
    else:
        viewed = valid & (customers[vwd_col] > 0)

    return customers[received][stat], customers[viewed][stat], cpd

def get_average_expense(customers, offer):
    """ Get the average expense for customers that received but not
    viewed an offer, viewed but not completed the offer, and those
    that viewed and completed the offer
    Input:
    - customers: dataframe with aggregated data of the offers
    - offer: offer of interest
    Output:
    - (received, viewed, completed): tuple with the average expense
    """
    rcv_total, vwd_total, cpd_total = get_offer_stat(customers,
                                                     'total_expense', offer)
    rcv_trans, vwd_trans, cpd_trans = get_offer_stat(customers,
                                                     'total_transactions',
                                                     offer)

    rcv_avg = rcv_total / rcv_trans
    rcv_avg.fillna(0, inplace=True)
    vwd_avg = vwd_total / vwd_trans
    vwd_avg.fillna(0, inplace=True)

    cpd_avg = None
    if offer not in ['informational', 'I1', 'I2']:
        cpd_avg = cpd_total / cpd_trans

    return rcv_avg, vwd_avg, cpd_avg

def get_offer_stat_by(customers, stat, offer, by_col, aggr='sum'):
    """ Get any column for customers that received but not viewed an offer,
    viewed but not completed the offer, and those that viewed and completed
    the offer, grouped by a column
    Input:
    - customers: dataframe with aggregated data of the offers
    - stat: column of interest
    - offer: offer of interest
    - by_col: column used to group the data
    - aggr: aggregation method sum or mean
    Output:
    - (received, viewed, completed): tuple with sum aggregation
    """
    valid = (customers.valid == 1)
    rcv_col = '{}_received'.format(offer)
    vwd_col = '{}_viewed'.format(offer)
    received = valid & (customers[rcv_col] > 0) & (customers[vwd_col] == 0)
    cpd = None
    if offer not in ['informational', 'I1', 'I2']:
        cpd_col = '{}_completed'.format(offer)
        viewed = valid & (customers[vwd_col] > 0) & (customers[cpd_col] == 0)
        completed = valid & (customers[cpd_col] > 0)
        if aggr == 'sum':
            cpd = customers[completed].groupby(by_col)[stat].sum()
        elif aggr == 'mean':
            cpd = customers[completed].groupby(by_col)[stat].mean()
    else:
        viewed = valid & (customers[vwd_col] > 0)
    if aggr == 'sum':
        rcv = customers[received].groupby(by_col)[stat].sum()
        vwd = customers[viewed].groupby(by_col)[stat].sum()
    elif aggr == 'mean':
        rcv = customers[received].groupby(by_col)[stat].mean()
        vwd = customers[viewed].groupby(by_col)[stat].mean()

    return rcv, vwd, cpd


def get_average_expense_by(customers, offer, by_col):
    """ Get the average expense for customers that received but not
    viewed an offer, viewed but not completed the offer, and those
    that viewed and completed the offer, group by a column
    Input:
    - customers: dataframe with aggregated data of the offers
    - offer: offer of interest
    - by_col: column used to group the data
    Output:
    - (received, viewed, completed): tuple with the average expense
    """
    rcv_total, vwd_total, cpd_total = get_offer_stat_by(customers,
                                                        'total_expense',
                                                        offer, by_col)
    rcv_trans, vwd_trans, cpd_trans = get_offer_stat_by(customers,
                                                        'total_transactions',
                                                        offer, by_col)

    rcv_avg = rcv_total / rcv_trans
    rcv_avg.fillna(0, inplace=True)
    vwd_avg = vwd_total / vwd_trans
    vwd_avg.fillna(0, inplace=True)

    cpd_avg = None
    if offer not in ['informational', 'I1', 'I2']:
        cpd_avg = cpd_total / cpd_trans

    return rcv_avg, vwd_avg, cpd_avg
