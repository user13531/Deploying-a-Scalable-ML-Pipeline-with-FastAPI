# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Purpose of the model is to determine if a person makes over $50K per year based on a number of factors.  These factors include education, marital status, occupation, race, sex, and country of origin.

The encoder used was OneHotEncoder from sklearn.

## Intended Use

This analysis is used for the student to learn and demonstrate how to deploy a machine learning model. It is the second project for the Machine Learning DevOps D501 course from WGU and was developed by Udacity.

## Training Data

The data was from the census data provided with the course. The census data included more categories than were used for the analysis. The categories used for the analysis were as follows:

workclass
education
marital-status
occupation
relationship
race
sex
native-country

## Evaluation Data

## Metrics

The metric output was in the following format and saved in the slice_output.txt file.

Precision: 0.7500 | Recall: 0.7881 | F1: 0.7686
workclass: Self-emp-not-inc, Count: 498

## Ethical Considerations

This model was only designed to be used for the student to learn about the process. If this had been intended for serious study for drawing conclusion in a real-world setting, further examination of the data would be needed to determine if there was any bias in the data that could skewe the conclusions.

## Caveats and Recommendations
No caveats or recommendations.
