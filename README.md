# Welcome! In this project, I built a model to predict customer churn!

![hello](https://github.com/tiaplagata/dsc-phase-3-project/blob/main/Images/Phase%203%20Project.png?raw=true)

# Navigation

* [Project Overview](#Project-Overview)
* [EDA](#EDA)
* [Model Analysis](#Model-Analysis)
* [Future Work](#Future-Work)

# Important Links

* [Slideshow Presentation](https://github.com/tiaplagata/dsc-phase-3-project/blob/main/Non_Technical_Slideshow.pdf)
* [Non-Technical Video Presentation](https://youtu.be/8KW4okIJfRc)
* [Jupyter Notebook with Cleaning](https://github.com/tiaplagata/dsc-phase-3-project/blob/main/Notebooks/Cleaning.ipynb)
* [Jupyter Notebook with Exploratory Data Analysis (EDA)](https://github.com/tiaplagata/dsc-phase-3-project/blob/main/Notebooks/EDA.ipynb)
* [Jupyter Notebook Modeling and Validation](https://github.com/tiaplagata/dsc-phase-3-project/blob/main/Notebooks/Modeling.ipynb)


# Project Overview

SyriaTel Communications is a Telecommunications company that is looking to predict and prevent customer churn. Customer churn is when a customer leaves/discontinues their service with SyriaTel. Customer churn is a major problem for many service-based companies because it is so expensive. Not only does the company lose the customer's monthly/yearly payment, but they also incur a customer acquisition cost to replace that customer.

To help SyriaTel fix the problem of customer churn, I first conducted an Exploratory Data Analysis (EDA) and then built a machine learning classifier that will predict the customers that are going to churn. This way, SyriaTel can create a more robust strategy to circumvent their customers from churning.

In the EDA portion, I explored the following questions: 
* Is calling customer service a sign of customer unhappiness/potential churn?
* How much are people using their plan? What can this tell us about churn?
* Are customers in certain areas more likely to churn?

 
**Scope & Data Used**

This project used the churn in telecoms dataset, which can be found in the data folder, and on kaggle via [this link](https://www.kaggle.com/becksddf/churn-in-telecoms-dataset). This dataset included 21 columns and 3333 unique values. It was already pretty clean with no large outliers or null values.


# EDA

## Is calling customer service a sign of customer unhappiness/potential churn?

**Findings**

Our current churn rate for the training data set is about 14.5%. When we look at customer service calls, we can see that as the number of customer service calls increases, the *likelihood* of churning increases as well. Specifically, with at least 4 customer service calls, the likelihood of a customer churning increases from about 10% to 50%.

![customer_service](https://github.com/tiaplagata/dsc-phase-3-project/blob/main/Images/customer_service_bar.png?raw=true)

Customer service calls alone cannot guarantee that a customer will churn. In fact, the majority of customers who DID NOT churn made 1-2 customer service calls. However, it is important to note that the majority people who DID churn made 1-4 calls to customer service. Therefore, more than 3 calls to customer service should be a red flag that a customer is more likely to churn.

![customer_service_2](https://github.com/tiaplagata/dsc-phase-3-project/blob/main/Images/customer_service_boxen.png?raw=true)

**Recommendation:** 

Based on these findings, I would recommend revisiting our customer service protocol. It may be useful to offer a larger incentive/discount to customers making more than 3 calls to customer service. 


## How much are people using their plan? What can this tell us about churn?

**Findings**

It is clear that the customers who churned and those that did not churn had almost exactly the same usage across day, eve, night and international calls. The rates for international minutes are the same regardless of whether the customer has an international plan or not (27 cents per minute). It is also interesting to note that the percentage of customers who churned was higher for customers with international plans than for customers without international plans. Because of this similar charge for international calls, it is possible that the customers who had an international plan and churned did not feel that paying for the international plan was worth it.

![international_plan](https://github.com/tiaplagata/dsc-phase-3-project/blob/main/Images/international_plan_bar_subplots.png?raw=true)

**Recommendations:**

Based on these findings, I recommend changing the rates for international minutes. If a customer has an international plan, they should have cheaper rates for international calls than a customer without an international plan.


## Are customers in certain areas more likely to churn?

**Findings**

It is clear that there are certain states with much higher churn. When grouped by state, Texas has a much higher churn than any other state (27%). New Jersey, Maryland and California also have higher churn (over 23%). States with the least churn include Hawaii and Iowa (under .05%). 

![state_choropleth](https://github.com/tiaplagata/dsc-phase-3-project/blob/main/Images/churn_by_state_choropleth.png?raw=true)

There could be a few reasons for this difference in churn in different states. One reason could be the lack of competitors in places like Hawaii and Iowa, which are more remote. States such as California, New Jersey or Texas could have many other big players in the market, which causes our customers to have other options when they feel inclined to leave. Another reason could be the lack of good service in certain areas in states with high churn.

**Recommendations**

Based on these findings, I would recommend looking into competitors in Texas, California, New Jersey, and other states with high churn to see if they are offering introductory offers that might compel some of our customers to churn. I also recommend looking into the cell signal in these states with higher churn to see if there are any deadzones contributing to the higher rates. 

## EDA Conclusion

In conclusion, calls to customer service seems to be one of the biggest indicators of customer churn. We can also see higher churn in certain states, although the reasons why specific states are more likely to churn is unclear based on this data. We can also see that customers may not be happy with their international plans, which is why customers with an international plan are more likely to churn than customers without an international plan.


# Model Analysis

My final model was a Gradient Boosting classifier, which can predict customer churn with 83% recall.
    
## Metric Used

I used recall to score this model because with churn rate, false negatives will cost us more than false positives. For example, misidentifying someone as 'churned' and hitting them with a customer retention strategy to keep them engaged would be less costly than missing someone who churned, losing them as a customer AND having to pay a new customer acquisition cost to replace them. 


## Cost Benefit Analysis

Let's say the cost of a False Positive is having to give a customer a discount of 50% off one month of free service when they were not actually going to churn. For this analysis we will say that the **cost of a FP = 25 USD per customer (-25)**.

Alternatively, the cost of a False Negative is losing that customer (and their monthly payment of 50 USD) and having to go out and get a new customer (customer acqusition cost of 50 USD). Therefore, we will say that the **cost of a FN = 100 USD per customer (-100)**.

The benefit of a True Positive is keeping that customer on and having them continue paying their 50 USD monthly payment, minus the 50% discount. **Benefit of TP = 25**

The **benefit of a True Negative = 0** since they were not going to churn and we predicted that, so we did not offer any discounts.

![cb_analysis](https://github.com/tiaplagata/dsc-phase-3-project/blob/main/Images/cb_analysis.png?raw=true)

Based on this cost benefit analysis, our expected value from this strategy is 52 cents per customer per month. That may not seem like much, but for millions of customers it would add up. The good news here is that with this model predicting churn, we are not LOSING money! We can see the breakdown of each cost and benefit multiplied by the number of TP, TN, FP, FNs on the confusion matrix above. 


## Feature Importances

The final model's feature importances are graphed below. 

![feat_importances](https://github.com/tiaplagata/dsc-phase-3-project/blob/main/Images/feature_importances.png?raw=true)


## Model Fit & Score

The final model had the following training and validation recall scores:
* Validation Recall Score 0.83
* Training Recall Score 0.88

Since these recall scores are so close, we can assume the model is slightly overfit, but overall very good on recall. This model produced only 9 (2%) false negatives for the validation set. It produced only 1 (0.003%) false positive from the validation set, but if our customer retention strategy is to keep these customers engaged, it is not a bad thing to keep a customer engaged who is mispredicted as potentially exiting.


## Model Conclusion

In conclusion, based on this model's recall score and cost benefit analysis, using this model to predict the churn of SyriaTel's customers will result in a large cost savings, and even an opportunity to make money ($0.52 per customer per month). 


# Future Work

If I had time to explore further, I would investigate the following:

* Get more data regarding competitors in states with higher churn
* Get more data on cell signal across the US to look for patterns in states with higher churn
* Look into voicemail data to see if that may be a good indicator of churn
* Continue to tweak this model to keep improving recall
* Build a class for expected value and integrate it into the model's pipeline, then run a gridsearch using the expected value as the scoring method