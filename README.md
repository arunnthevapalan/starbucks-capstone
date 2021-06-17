# Promotions Optimization for Starbucks

This repository consists of an analysis of Starbucks Offers, transactions and Customers and builds recommendions on the best offers to be sent to different customers. To read a more in-depth project walkthrough [checkout this article.](https://www.arunnthevapalan.com/starbucks-capstone/)

**The final project report can be [found here.](/project_report.pdf)**

**The project proposal can be [found here.](proposal.pdf)**


## Project Overview

Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks. The goal of sending advertisement and offers to customers is to increase the customer purchases. However, it is not advisable to send all offers to all customers at the same time. Here the goal is to take advantage of the transactions and demographics data to determine the offers that should be targeted to different groups of customers. 

## Pre-requistes

The project was developed using python 3.6.7 with the following packages.
- Pandas
- Numpy
- Matplotlib
- Json

## Files
- Starbucks_Capstone_notebook.ipynb : Jupyter Notebook with all the workings including data preparation, analysis and recommendations.
- data/ : data from Starbucks
- eda.py: functions for exploratory data analysis
- data_preprocessing.py : functions for data preprocessing
- recommendations.py : functions for recommendations
- ouput/ : images from Data Visualization

## Summary

The datasets were pre-processed and a master datatable was created with all the relavent information. An explorative data analysis was carried out and several trends and correlations were discovered. With that, this project further attempts to create 2 recommendation systems from scratch. They knowledge-based recommender system has been proposed, backed by in-depth analysis and evalutions. In reality both the models can be coupled together and used. That is, the simple systems should be use for customers that do not provide their personal information, while the one with filters can be used for customers that do. For a more detailed walthrough of the project, [please checkout this blogpost.](https://www.arunnthevapalan.com/starbucks-capstone/)


## Acknowledgements

[Starbucks](https://www.starbucks.com/), for providing the data based on simulations of real environment. 
