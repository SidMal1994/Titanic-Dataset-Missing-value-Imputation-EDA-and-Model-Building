# Titanic-Dataset-Missing-value-Imputation-EDA-and-Model-Building
Overview

The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).

Variable Notes

embarked Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton

pclass: A proxy for socio-economic status (SES) 1st = Upper 2nd = Middle 3rd = Lower

age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

sibsp: The dataset defines family relations in this way... Sibling = brother, sister, stepbrother, stepsister Spouse = husband, wife (mistresses and fiancés were ignored)

parch: The dataset defines family relations in this way... Parent = mother, father Child = daughter, son, stepdaughter, stepson Some children travelled only with a nanny, therefore parch=0 for them.

Survived(o/v)= 0:No,1:Yes.


Steps taken:
1. Imported and cleaned the data into the Jupyter notebook.
2. After some preliminary cleaning, drew some insights to fill the 'nan' value.
3. Removed some columns and created a model to predict and fill the age column's "nan" values.
4. Exported the data to Excel for further analysis in a separate notebook.
5. Imported the data into another notebook using standard libraries, performed EDA, and drew additional insights.
6. Separated the dependent and independent variables to preprocess the data.
7. Scaled down OneHot Encoder and Standard Scaler to apply algorithm and build model.
8. The data was in nd. Array format after scaling; therefore, using pandas, the array was transformed into a dataframe.
9. Divided the data again into train and test to build the model.
10. Finally, the model was built and the results were compared using logistic regression, random forest classifier, and xgboost classifier.
