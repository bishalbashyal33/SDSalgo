
import numpy as np
import keras.backend as K
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
#np.random.seed(1)
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

def select_random(model, f_list, x_test, y_test, delta, iterate):
    batch = delta
    arr = np.random.permutation(len(f_list))
    min_index0 = arr[0:30]
    acc_list = []
    for i in range(len(model)):
        acc_list.append([])
    for i in range(iterate):
        # random
        arr = np.random.permutation(len(f_list))
        start = int(np.random.uniform(0, len(f_list) - batch))
        min_index = arr[start:start + batch]
        min_index0 = np.append(min_index0, min_index)
        for j in range(len(model)):
            label = y_test[np.array(f_list)[min_index0]]
            orig_sample = x_test[np.array(f_list)[min_index0]]
            orig_sample = orig_sample.reshape(-1, 28, 28, 1)
            pred = np.argmax(model[j].predict(orig_sample), axis=1)
            acc = np.sum(pred == label) / orig_sample.shape[0]
            acc_list[j].append(acc)
            print("numuber of samples is {!s}, SDS acc is {!s}".format(
                orig_sample.shape[0], acc))
    return acc_list


def mode_count(mode_list, mode):
    count = 0
    for i in range(len(mode_list)):
        if mode_list[i] == mode:
            count += 1
    return count


def label_read(model, x_test, y_test):
    label_list = []
    for i in range(len(model)):
        print('model{} begins testing.'.format(i))
        label_list.append([])
        for j in range(len(y_test)):
            print('sample{} finishes testing.'.format(j))
            orig_sample = x_test[j]
            
            #here, -1 automatically infers the total number of samples, 28,28 is the image dimension of fashion mnist, and 1 is the number of channels representing that we are dealing with grayscale images.
            orig_sample = orig_sample.reshape(-1, 28, 28, 1)
            
            
            #here, model.predict gives out logits corresponding to 10 classes in the shape of (1,10) for each sample, and since the index of the highest logits represent the predicted class, np.argmax gives the class index to which it was classified. axis = 1 means the axis with size =10.
            pred = np.argmax(model[i].predict(orig_sample), axis=1)
            
            #for each model i, append the output predictions for all samples j at the ith list inside the list of list "label"
            label_list[i].append(pred[0])
    
    #the final list inside the label_list is the actual_labels (y_test) itself so that we can use it for validation purposes.
    label_list.append([])
    for i in range(len(y_test)):
        label = y_test[i]
        label_list[-1].append(label)
    print('y_test gets.')
    dataframe = pd.DataFrame(label_list)
    
    #the prediction corresponding to each samples :=> sample number is uptop horizontally, model numbers are going down from the leftest position vertically, and the corresponding classifcation values are filled, the last row represents the actual expected output
    dataframe.to_csv(r"test.csv")
    return label_list


    # Accuracy of the majority vote (rate).
    # The number of models agreeing with the majority vote for each sample (mode_counts).
    # Indices of samples where the majority vote matches the true label (con_list).
    # The majority vote (mode) of predictions for each sample (mode_list).

def label_compare(model, label_list, y_test):
    from scipy import stats
    mode_list = []
    mode_counts = []
    con_list = []
    for i in range(len(y_test)):
        mode_store = []
        for j in range(len(model)):
            mode_store.append(label_list[j][i])
            
            
        # Computes the mode of mode_store (list of 25 predictions).
        # Returns a tuple: (array of modes, array of counts).
        # For a single mode, stats.mode(...)[0] is an array of the mode(s), and [0][0] extracts the first mode as a scalar.
        
        mode = stats.mode(mode_store)[0][0]
        mode_list.append(mode)
        
        # counting how many models make prediction that match the majority voting for sample i
        m_count = mode_count(mode_store, mode)
        
        # append the total number of "majority matching predictions" in mode_counts for each sample i
        mode_counts.append(m_count)
        print('mode of sample{} gets.'.format(i))
     
    # Compares the majority vote (mode_list[i]) to the true label (y_test[i]) for each sample, tracking correct predictions, appends 1 for matching, 0 for not matching.
    compare_flag = []
    count = 0
    for i in range(len(y_test)):
        if mode_list[i] == y_test[i]:
            a = 1
            compare_flag.append(a)
            count = count + 1
            
            # con_list is appending the indices of those samples on which the majority voting matches the true outputs.
            con_list.append(i)
        else:
            a = 0
            compare_flag.append(a)
    
    
    # Converts mode_counts (raw counts of models agreeing with the mode) to fractions by dividing by the number of models (25).        
    mode_counts_ = []
    for i in range(len(mode_counts)):
        mode_counts_.append(mode_counts[i]/len(model))
    
    # this csv writing part seems repetetive to me as it is writing the same label_list again to compare.csv
    dataframe = pd.DataFrame(label_list)
    dataframe.to_csv(r"compare.csv")
    
    return count/len(mode_list), mode_counts, con_list, mode_list


def r_rate(model, label_list, mode_list):
    r_rate = []
    for i in range(len(model)):
        score = 0
        for j in range(len(label_list[i])):
            if label_list[i][j] == mode_list[j]:
                score += 1
                
        # Calculates the fraction of samples where model i matches mode_list and stores it in r_rate.
        scores = score/len(label_list[i])
        r_rate.append(scores)
    print(len(r_rate))
    print(len(label_list))
    print(r_rate)
    
    # sorts in ascending order
    return r_rate, np.argsort(r_rate)


# r_rank: Sorted model indices (descending performance).
# label_list: Predictions and true labels.
# y_test: True labels (not used directly in the function).
# mode_list: Majority vote labels.

def item_discrimination(r_rank, label_list, y_test, mode_list):
    
    # select top and bottom 27% models based on r_rank
    top_rank = r_rank[:int(len(r_rank) * 0.27)]
    last_rank = r_rank[int(len(r_rank) * 0.73):]
    
    
    # initializes an empty list to store discrimination scores for each of the 10 samples.
    item_dis = []
    for i in range(len(y_test)):
        
        # Computes the fraction of top models agreeing with mode_list[i].
        score1 = 0
        
        # Computes the fraction of bottom models agreeing with mode_list[i].
        score2 = 0
        for j in range(len(top_rank)):
            if label_list[top_rank[j]][i] == mode_list[i]:
                score1 += 1
        
        # Normalize Top Model Score
        scores1 = score1/len(top_rank)
        
        for k in range(len(last_rank)):
            if label_list[last_rank[k]][i] == mode_list[i]:
                score2 += 1
                
        # Normalize Bottom Model Score
        scores2 = score2/len(last_rank)
        
        # calculation of discrimination scores
        item_dis.append(scores1-scores2)
    
    # prints discrimination score for all samples    
    print(item_dis)
    
    # Returns the discrimination scores and their sorted indices.
    return item_dis, np.argsort(item_dis)

if __name__ == '__main__':
    import datetime
    start = datetime.datetime.now()
    # preprocess the data set
    from keras.datasets import fashion_mnist
    img_rows, img_cols = 28, 28
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # for running the experiment on a smaller sample dataset
    # num_samples = 1000
    # x_test = x_test[:num_samples]
    # y_test = y_test[:num_samples]
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_train = x_train.astype('float32')
    x_train /= 255
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    x_test = x_test.astype('float32')
    x_test /= 255
    from keras.models import load_model

    model = []
    model_name = '../models/model-'
    filename = []
    # this range is the total number of models present in the models folder, we are basically loading all models in the models folder here.
    for i in range(1, 26):
        model_name_ = model_name + str(i) + '.h5'
        filename.append(model_name_)

    #  once we parse all filenames and save that in a list, we make a list of loaded models from the h5 file.
    for i in range(len(filename)):
        model.append(load_model(filename[i]))
        print('model{} has been loaded.'.format(i+1))
    
    
    # Create the 'experiment' directory if it doesn't exist
    os.makedirs('experiment', exist_ok=True)

    # Create output directories for each model inside the 'experiment' folder
    for i in range(len(model)):
        os.makedirs(f'experiment/model{i+1}', exist_ok=True)

    
    
    # outputs a list of list of model predidctions for each sample, with the final row conssisting of the real classification corresponding to each sample.
    label_list = label_read(model, x_test, y_test)
    
    
    # total number of models that agree with the mode divide by the total number of models(25) (rate).
    # total number of models that agree with the mode (mode_counts).
    # Indices of correctly classified samples (con_list).
    # tuple of (array of modes, array of counts) (mode_list).
    
    rate, mode_counts, con_list, mode_list = label_compare(model,label_list,y_test)
    
    
    
    
    
    # Purpose of r_rate:
    # Evaluates each model by computing a score (r_rate) based on how often its predictions match the majority vote labels (mode_list).
    # Ranks the models by these scores, returning indices (r_rank) from lowest to highest score.
    # These scores and ranks are used later in item_discrimination to identify discriminative samples by comparing high- and low-performing models.
    
    # Inputs:
    # model: List of 25 Keras models.
    # label_list: List of 26 lists (25 model predictions + true labels), each containing 10 predicted/true labels for the 10 samples.
    # mode_list: List of 10 majority vote labels, one per sample, derived from label_compare.
    
    # Outputs (assigned to r_rate, r_rank):
    # r_rate: List of 25 floats, each representing the fraction of samples where a model’s predictions match the majority vote.
    # r_rank: NumPy array of indices (0 to 24) sorting models by their r_rate scores in ascending order (lowest to highest score).
    
    r_rate, r_rank = r_rate(model, label_list, mode_list)
    
    
    # sorts in descending order since the previous function gives results in ascending order based on model performance (r_rate).
    r_rank = r_rank[::-1]
    
    
    # gives out the discrimination scores for each sample, and the indices of samples based on sorted discrimination score from low to high.
    item_dis, item_dis_rank = item_discrimination(r_rank, label_list, y_test, mode_list)
    
    
    print(item_dis_rank)
    print(item_dis[item_dis_rank[0]])
    
    # sorts ranking of sample indicies from high to low with respect to discrimination scores.
    item_dis_rank = item_dis_rank[::-1]
    
    # selects 25 percent of the most discriminating samples.
    rank_test = item_dis_rank[:int(len(item_dis_rank) * 0.25)]


    for k in range(50):  #5 random experiments, is 50 in the original code
        print("the {} exp".format(k))
        acc_list = select_random(model,rank_test,x_test,y_test,5,30)
        for i in range(len(model)): 
            np.savetxt('model{}/random{}.csv'.format(i+1,k), acc_list[i])


    end = datetime.datetime.now()
    print((end - start).seconds)
