# Model code used from:
# https://www.youtube.com/watch?v=nzNp05AyBM8&t=2s

# Chatgpt used to show how to make a pivot table
import numpy as np
import pandas as pd
import argparse

# read in files
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Homework 1",
        epilog = "CSCI 4360/6360 Data Science II: Fall 2023",
        add_help = "How to use",
        prog = "python homework1.py [train-data] [train-label] [test-data]")
    parser.add_argument("paths", nargs = 3)
    args = vars(parser.parse_args())

    # Access command-line arguments as such:
    
    training_data_file = args["paths"][0]
    training_label_file = args["paths"][1]
    testing_data_file = args["paths"][2]

train_data= pd.read_csv(training_data_file, delim_whitespace=True, header = None)
train_labels= pd.read_csv(training_label_file, delim_whitespace=True, header = None)
test_data = pd.read_csv(testing_data_file, delim_whitespace=True, header = None)

# Formatting the input files
def format_data(df):
    df.columns = ['DocID', 'WordID', 'Count']
    pivot_df = df.pivot_table(index='DocID', columns='WordID', values='Count', aggfunc='sum', fill_value=0)
    max_word_id = df['WordID'].max()
    pivot_df = pivot_df.reindex(columns=range(1, max_word_id+1), fill_value=0)
    return pivot_df
    
training_data = format_data(train_data)
training_labels = train_labels
testing_data = format_data(test_data)

# Making sure the testing and training dataframes have the same column dimensions
columns_df1 = training_data.columns
columns_df2 = testing_data.columns

if len(columns_df1) > len(columns_df2):
    testing_data = testing_data.reindex(columns=columns_df1, fill_value=0)

elif len(columns_df1) < len(columns_df2):
    training_data = training_data.reindex(columns=columns_df2, fill_value=0)

else:
    pass

# Converting to numpy
training_data = training_data.values
training_labels = training_labels.values
testing_data = testing_data.values

# Code from link below is directly used here 
# Slight modifications but math is the same:
#https://www.youtube.com/watch?v=nzNp05AyBM8&t=2s

training_data = training_data.T
training_labels = training_labels.reshape(1, training_data.shape[1])
testing_data = testing_data.T

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def model(X, Y, step_size, iterations):
    
    m = training_data.shape[1]
    n = training_data.shape[0]
    
    W = np.zeros((n,1))
    B = 0
    
    for i in range(iterations):
        
        Z = np.dot(W.T, X) + B
        A = sigmoid(Z)
        
        cost = -(1/m)*np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))
        
        dW = (1/m)*np.dot(A-Y, X.T)
        W = W - step_size*dW.T

        dB = (1/m)*np.sum(A - Y)        
        B = B - step_size*dB

    return W, B


iters = 10000
STEP_SIZE = 0.0001

W, B = model(training_data, training_labels, STEP_SIZE, iters)

def predict(X, W, B):
    Z = np.dot(W.T, X) + B
    A = sigmoid(Z)

    A = A > 0.4

    A = np.array(A, dtype = 'int64')
    return (A)

predicted = predict(testing_data, W, B)

# Print out the numpy array
def printer(df):
    rows, cols = df.shape
    for i in range(rows):
        for j in range(cols):
            print(df[i, j])

printer(predicted)