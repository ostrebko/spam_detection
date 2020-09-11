import pandas as pd
import math
import re
import os

# Функция для создания словарей ("спам" или "не спам") из текста (одна строка) тренировочной модели;  
# body, label - исходные данные
def calculate_word_frequencies(body, label):
    pattern = re.compile('[a-zа-я0-9]+')
    if type(body) == str:
        list_of_str = pattern.findall(body.lower())
    else:
        body = str(body)
        list_of_str = pattern.findall(body.lower())    
   
    if label == 'SPAM':
        for word in list_of_str:
            if word in A_spam.keys():
                A_spam[word] += 1
            else:
                A_spam[word] = 1
        return A_spam
    
    else:
        for word in list_of_str:
            if word in A_not_spam.keys():
                A_not_spam[word] += 1
            else:
                A_not_spam[word] = 1
        return A_not_spam


# Функия для тренировки модели - создания словарей ("спам" или "не спам") из набора тренировочных данных. 
# Вероятность встретить спам и не встретить спам - pA и pNotA
# Набор тренировочных данных берем из файла spam_or_not_spam.csv

train_data = []
# функция приведена для запуска на linux
def df_to_train_data(d_frame = pd.read_csv('/home/' + os.environ['USERNAME'] + '/task_2_9/spam_or_not_spam.csv')):
    for row in range(len(d_frame)):
        text = d_frame.loc[row, ('email')]
        if d_frame.loc[row, ('label')] == 1:
            label = 'SPAM'
        else:
            label = 'NOT_SPAM'
        train_data.append([text, label])
    return train_data

def train(training_data = df_to_train_data()):
    count = 0
    for data in training_data:
        calculate_word_frequencies(data[0], data[1])
        pA_dict[data[1]] += 1
    return pA_dict

A_spam = {}
A_not_spam = {}
pA_dict = {'SPAM':0, 'NOT_SPAM':0} # Прежде чем использовать этот словарь - датасет причесывается

# Функция расчета вероятности встретить слова среди спама. 
# label - дополнительный параметр, который задается вручную, если мы знаем относится слово к спаму или нет и
# хотим вычислить вероятность. 
def calculate_P_Bi_A(word, label = 'unknown'):
    if label == 'SPAM':
        if (word in A_spam.keys()):
            P_Bi_A_spam = math.log((A_spam[word] + 1) / sum(A_spam.values()))
        else:
            P_Bi_A_spam = math.log(1 / sum(A_spam.values()))
        return P_Bi_A_spam
    
    elif label == 'NOT_SPAM':
        if word in A_not_spam.keys():
            P_Bi_A_not_spam = math.log((A_not_spam[word] + 1) / sum(A_not_spam.values()))
        else:
            P_Bi_A_not_spam = math.log(1 / sum(A_not_spam.values()))
        return P_Bi_A_not_spam    
    
    else:
        if (word in A_spam.keys())&(word in A_not_spam.keys()):
            P_Bi_A_spam = math.log((A_spam[word] + 1) / sum(A_spam.values()))
            P_Bi_A_not_spam = math.log((A_not_spam[word] + 1) / sum(A_not_spam.values()))
        elif (word in A_spam.keys())&(word not in A_not_spam.keys()):
            P_Bi_A_spam = math.log((A_spam[word] + 1) / sum(A_spam.values()))
            P_Bi_A_not_spam = math.log(1 / sum(A_not_spam.values()))
        elif (word not in A_spam.keys())&(word in A_not_spam.keys()):
            P_Bi_A_spam = math.log(1 / sum(A_spam.values()))
            P_Bi_A_not_spam = math.log((A_not_spam[word] + 1) / sum(A_not_spam.values()))
        else:
            P_Bi_A_spam = math.log(1 / sum(A_spam.values()))
            P_Bi_A_not_spam = math.log(1 / sum(A_not_spam.values()))
        return [P_Bi_A_spam, P_Bi_A_not_spam]


# Функция расчета вероятности отнесения текста к спаму на основе расчета calculate_P_Bi_A(word, label). 
# label - дополнительный параметр, который задается вручную, если мы знаем относится текст к спаму или нет и
# хотим вычислить вероятность. 

def calculate_P_B_A(text, label = 'unknown'):
    if label == 'SPAM':
        P_B_A_spam = 0
        for word in text.split():
            P_B_A_spam = P_B_A_spam + calculate_P_Bi_A(word, label)
        return P_B_A_spam
    
    elif label == 'NOT_SPAM':
        P_B_A_not_spam = 0
        for word in text.split():
            P_B_A_not_spam = P_B_A_not_spam + calculate_P_Bi_A(word, label)
        return P_B_A_not_spam   
    
    else:
        P_B_A_spam = 0
        P_B_A_not_spam = 0
        for word in text.split():
            P_B_A_spam = P_B_A_spam + calculate_P_Bi_A(word, label)[0]
            P_B_A_not_spam = P_B_A_not_spam + calculate_P_Bi_A(word, label)[1]
        return [P_B_A_spam, P_B_A_not_spam]
    

# Функция классификатора текста на основе расчета функции calculate_P_B_A(text, label)
# label - дополнительный параметр, который задается вручную, если мы знаем относится текст к спаму или нет, 
# то просто выводим вероятность

def classify(email, label = 'unknown'):
    pattern = re.compile('[a-zа-я0-9]+')
    list_of_words = pattern.findall(email.lower())
    text = ' '.join(list_of_words)
    pA = math.log(pA_dict['SPAM'] / (pA_dict['SPAM'] + pA_dict['NOT_SPAM']))
    pNotA = math.log(1 - pA)
    
    if label == 'SPAM':
        P_spam = pA + calculate_P_B_A(text, label)
        return f'ln(P_spam) = {P_spam}, SPAM'
    
    elif label == 'NOT_SPAM':
        P_not_spam = pNotA + calculate_P_B_A(text, label)
        return f'ln(P_not_spam) = {P_not_spam}, NOT_SPAM'
    
    else:
        P_spam = pA + calculate_P_B_A(text, label)[0]
        P_not_spam = pNotA + calculate_P_B_A(text, label)[1]
        if P_spam > P_not_spam:
            return 'SPAM'    #f'ln(P_spam) = {P_spam}, SPAM'
        else:
            return 'NOT_SPAM'    #f'ln(P_not_spam) = {P_not_spam}, NOT_SPAM'