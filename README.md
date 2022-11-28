# Customer Affinity for Loan

### Business Case:
The company analyzed corresponds to a Portuguese Banking Institution. The data in this Dataset is related to direct marketing campaigns (phone calls). The institution is interested in predicting whether a customer is likely to suscribe to a fixed term deposit.
<br>
<br>
### Objectives of the analysis:
From the Marketing Campaign we try to understand the behavior patterns of each user to be able to predict if a future client will access a fixed term deposit. With this information, the company will be able to correctly segment its advertising campaigns.
<br>
<br>
### Data description:
The Dataset is made up of more than 40,000 rows, where each one belongs to a telephone call made in previous marketing campaigns. There is a variable Target (y), which indicates whether or not a customer subscribed to a fixed term deposit.
<br>
<br>
### Attribute Information:
#### Bank client data:

Age (numeric)

Job : type of job (categorical: 'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown')

Marital : marital status (categorical: 'divorced', 'married', 'single', 'unknown' ; note: 'divorced' means divorced or widowed)

Education (categorical: 'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown')

Default: has credit in default? (categorical: 'no', 'yes', 'unknown')

Housing: has housing loan? (categorical: 'no', 'yes', 'unknown')

Loan: has personal loan? (categorical: 'no', 'yes', 'unknown')

#### Related with the last contact of the current campaign:

Contact: contact communication type (categorical:
'cellular','telephone') Month: last contact month of year (categorical: 'jan', 'feb', 'mar', …, 'nov', 'dec')

Dayofweek: last contact day of the week (categorical:
'mon','tue','wed','thu','fri')

Duration: last contact duration, in seconds (numeric). Important
note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

#### Other attributes:

Campaign: number of contacts performed during this campaign and for
this client (numeric, includes last contact)

Pdays: number of days that passed by after the client was last
contacted from a previous campaign (numeric; 999 means client was not previously contacted) Previous: number of contacts performed before this campaign and for this client (numeric)

Poutcome: outcome of the previous marketing campaign (categorical:
'failure','nonexistent','success')

#### Social and economic context attributes:

Emp.var.rate: employment variation rate - quarterly indicator
(numeric)

Cons.price.idx: consumer price index - monthly indicator (numeric)
Cons.conf.idx: consumer confidence index - monthly indicator
(numeric)

Euribor3m: euribor 3 month rate - daily indicator (numeric)

Nr.employed: number of employees - quarterly indicator (numeric)

#### Output variable (desired target):

y - has the client subscribed a term deposit? (binary: 'yes', 'no')
<br>
<br>
### Models:
It was defined to train 7 different models for the prediction of the binary target variable (y_yes: indicates whether a person will make a term deposit or not).
<br>
<br>
### Model Training:
For the training of the model we use Stratified K Fold method and Optuna library to find the best hyperparameters.
<br>
<br>
### Metrics:
We use Recall metric as it allows us to correctly deal with unbalanced datasets. It helps answer the question: What percentage of customers who are interested are we able to identify? (It is not relevant to identify those that are NOT, because they are the vast majority).
<br>
<br>
### Model Results:

Recall was the best metric to analyze due to the structure of the data. Please find below the results:

XGBoost: 5.98%
<br>
<br>
K-NN Classification: 10.8%
<br>
<br>
Random Forest: 4.49%
<br>
<br>
Logistic Regression: 22.34%
<br>
<br>
Support Vector Machines: 6.08%
<br>
<br>
Naive Bayes Classifier: 75.55%
<br>
<br>
Catboost Classifier: 3.23%
<br>
<br>
Stacking model: 6%

Recall is a metric that quantifies the number of correct positive predictions made out of all positive predictions that could have been made.

<br>
<br>

### Future Lines:

In the future, a detailed analysis could be carried out of the cases that were classified as negative, that is, they would not subscribe to a term deposit.
The idea is to cluster with clients who are not likely to adhere to a fixed term within the bank.
<br>
<br>

### Conclusions:

The results show that we are able to identify with more accuracy those who will not suscribe fixed term deposit, that those that will. <br>

Apparently, the stacking model it's not as effcient as Naive Bayes model for this particularly case. <br>

Finally, we believe that a more balanced dataset will provide this model more consistence, accuracy and efficience. 







