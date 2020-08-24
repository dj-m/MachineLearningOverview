# Data Scientists and Machine Learning #

## Two Categories of Challenges ##

There are two categories of challenges that machine learning practitioners and data scientists face: bad algorithms and bad data. 

**Bad algorithms** can come from: 

- Poor feature engineering
- Poor feature selection or poor feature extraction
- Overfitting the training data — the model performs well on the training data alone but doesn't generalize well to unseen data
- Underfitting the training data — your model is too basic to learn the underlying structure of the data
- Failure to properly cross-validate — cross-validation gives you sensible hyperparameters, but you can easily misapply this technique — you'll learn more about this later 

**Bad data** can arise for many reasons. These include: 

- Nonrepresentative training data — the training data is not representative of the new cases to which you want to generalize
- Sampling bias exhibited by the data
- Poor-quality data — errors, outliers, and noise (due to poor-quality measurements). 

## Dealing with the Lack of Data in Machine Learning ##

The problem of data scarcity is very important since data are at the core of any AI project.

Supervised machine learning models are being successfully used to respond to a whole range of business challenges. However, these models are data-hungry and their performance relies heavily on the size of training data available. In many cases, it is difficult to create training datasets that are large enough.

**How much data do I need?**

Well, you need roughly 10 times as many examples as there are degrees of freedom in your model.

**Overfitting:** refers to a model that models the training data too well. It happens when a model learns the detail and noise in the training data to the extent that it negatively impacts the performance of the model on new data.

In general, different machine learning algorithms can be used to determine the missing values. This works by turning missing features to labels themselves and now using columns without missing values to predict columns with missing values.

Based on my experience, you will be confronted with a lack of data or missing data at some point if you decide to build an AI-powered solution, **but fortunately, there are ways to turn that minus into a plus.**

**Lack of data?**

The very nature of your project will influence significantly the amount of data you will need.

- **Number of categories to be predicted**
  - What is the expected output of your model? Basically the fewest number of categories the better.
- **Model Performance**
  - If you plan on getting a product in production, you need more. **A small dataset might be good enough for a proof of concept but in production, you’ll need way more data.**
  
In general, small datasets require models that have low complexity (or [high bias](https://en.wikipedia.org/wiki/Bias–variance_tradeoff)) to avoid [overfitting](https://en.wikipedia.org/wiki/Overfitting) the model to the data.

**Non-Technical Solutions**

It might sound obvious but before getting started with AI, please try to obtain as much data as possible by developing your external and internal tools with data collection in mind.

If you need external data for your project, it can be beneficial to form partnerships with other organizations in order to get relevant data. Forming partnerships will obviously cost you some time, but the proprietary data gained will build a natural barrier to any rivals.

- **Build a useful application, give it away, use the data**
  - Another approach that I used in my previous project was to give away access to a cloud application to customers. The data that makes it into the app can be used to build machine learning models.
  
**Small datasets**

Common approaches that can help with building predictive models from small data sets are:
  - Use a simpler classifier model, e.g. a short decision tree; less susceptible to over-fitting.
  - Use ensemble methods, in which voting between classifiers can compensate for individual over-learning.
  
In general, the simpler the machine learning algorithm the better it will learn from small data sets.
  - Small data requires models that have low complexity (or high bias) to avoid overfitting the model to the data.
  - The Naive Bayes algorithm is among the simplest classifiers and as a result learns remarkably well from relatively small data sets.
    - **Naive Bayes methods:** set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of conditional independence between every pair of features given the value of the class variable.

For very **small datasets**, Bayesian methods are generally the best in class, although the results can be sensitive to your choice of prior. The Naive Bayes classifier and ridge regression are the best predictive models.

- Examples:
  - Linear models such as linear/logistic regression. Not only can you adapt the number of parameters easily, but the models also assume linear interactions only.
  - Simple Bayesian models such as Naive Bayes where you also have few parameters and a direct way to adjust your prior.
  
**Transfer learning**

This is a framework that leverages existing relevant data or models while building a machine learning model.

Transfer learning uses knowledge from a learned task to improve the performance on a related task, typically reducing the amount of required training data.

Transfer learning techniques should be considered when you do not have enough target training data, and the source and target domains have some similarities but are not identical.

- What if you have no data at all?
  - This is where data generation can play a role. 
    - It is used when no data is available, or when you need to create more data than you could amass even through aggregation.
	- In this case, the small amount of data that does exist is modified to create variations on that data to train the model. 
	  - For example, many images of a car can be generated by cropping, cropping, downsizing, one single image of a car.

Another common application of transfer learning is to train models on cross-customer datasets to overcome the cold-start problem.
  - SaaS companies often have to deal with when onboarding new customers to their ML products. Indeed, until the new customer has collected enough data to achieve good model performance (which could take several months) it’s hard to provide value.
  
**Data Augmentation**

Data augmentation means increasing the number of data points.
  - In terms of traditional row/column format data, it means increasing the number of rows or objects.
  
Every data collection process is associated with a cost.
  - This cost can be in terms of dollars, human effort, computational resources and of course time consumed in the process.
  
There are many ways to augment data.
  - If you are generating artificial data using over-sampling methods such as SMOTE, then there is a fair chance you may introduce over-fitting.
  
**Synthetic Data**

Synthetic data means fake data that contains the same schema and statistical properties as its “real” counterpart.
  - Basically, it looks so real that it’s nearly impossible to tell that it’s not.
  
Synthetic data is more likely applied when we're dealing with private data (banking, healthcare, etc.), this makes the use of synthetic data a more secure approach to development.

Synthetic data is used mostly when there is not enough real data or there is not enough real data for specific patterns you know about. Usage mostly the same for training and testing datasets.

  - **Synthetic Minority Over-sampling Technique (SMOTE)** and Modified-SMOTE are two such techniques which generate synthetic data.
    - Simply put, SMOTE takes the minority class data points and creates new data points which lie between any two nearest data points joined by a straight line.
	- The algorithm calculates the distance between two data points in the feature space, multiplies the distance by a random number between 0 and 1 and places the new data point at this new distance from one of the data points used for distance calculation.

In order to generate synthetic data, you have to use a Training Set to define a model, which would require validation, and then by changing the parameters of interest, you can generate synthetic data, through simulation. The domain/data type is significant since it affects the complexity of the entire process.

Advantages

  - No risk of copyright issues.
  - Perfect for understanding a particular concept.

Disadvantages

  - Risk of introducing biases.
  - Issues with understanding real-world data problems.
  
## Best Practices for Feature Engineering ##

Feature engineering efforts mainly have two goals:

  - Preparing the proper input dataset, compatible with the machine learning algorithm requirements.
  - Improving the performance of machine learning models.
  
The features you use influence more than everything else the result. No algorithm alone, to my knowledge, can supplement the information gain given by correct **feature engineering.**
— Luca Massaron

According to a survey in Forbes, data scientists spend **80%** of their time on **data preparation:**

![DataScientistTimeShare](DataScientistTimeShare.jpg)

List of Techniques:

1. Imputation
  - Imputation is a more preferable option rather than dropping because it preserves the data size. 
  - Except for the case of having a default value for missing values, I think the best imputation way is to use the medians of the columns. As the averages of the columns are sensitive to the outlier values, while medians are more solid in this respect.
  - Replacing the missing values with the maximum occurred value in a column is a good option for handling categorical columns.
2. Handling Outliers
  - 
3. Binning
4. Log Transform
5. One-Hot Encoding 
6. Grouping Operations 
7. Feature Split 
8. Scaling
9. Extracting Date