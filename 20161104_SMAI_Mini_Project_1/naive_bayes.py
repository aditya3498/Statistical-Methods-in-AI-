from PIL import Image

import os, glob, re, math, sys

import numpy as np

import matplotlib.pyplot as plt

list_img = []

arr1, arr2 = [], []

list_full, list_full_1 = [], []

losses = []

list_path, list_path_new = [], []

list_labels = []

list_dis_lab = []

with open(sys.argv[1]) as f:

	y = f.readlines()

	for i in y:

		i = i.strip('\n')

		i = re.sub(r'[ ]+', ' ', i)

		i = re.split(' ', i)

		arr1.append(i)

	i = 0

	while i < len(arr1):

		list_full += arr1[i]

		#list_labels += arr1[]

		i += 1

i = 0

while i < len(list_full):

	if i % 2 == 0:

		list_path.append(list_full[i])

	else:

		list_labels.append(list_full[i])

	i += 1

for i in list_labels:

	if i not in list_dis_lab:

		list_dis_lab.append(i)

allfiles = list_path

for file in allfiles:

	list_img.append(file)

matrix = np.asarray([np.array(Image.open(i).resize((64, 64)).convert('L')).flatten() for i in list_img], 'f')

final_matrix = matrix - matrix.mean(axis = 0)

M = np.cov(final_matrix.T)

e, EV = np.linalg.eigh(M)

idx = e.argsort()[::-1]

e = e[idx]

EV = EV[:,idx]

vectors_32 = EV.T[:32].T

Final_reduced = np.dot(matrix, vectors_32)

Final_reduced_norm = Final_reduced / Final_reduced.max(axis = 0)

with open(sys.argv[2]) as e:

	p = e.readlines()

	for i in p:

		i = i.strip('\n')

		i = re.sub(r'[ ]+', ' ', i)

		i = re.split(' ', i)

		arr2.append(i)

	i = 0

	while i < len(arr2):

		list_full_1 += arr2[i]

		i += 1

i = 0

while i < len(list_full_1):

	list_path_new.append(list_full_1[i])

	i += 1

temp = list_path_new

list_path_new = []

for i in temp:

	if i:
	
		list_path_new.append(i)

new_matrix = np.asarray([np.array(Image.open(i).resize((64, 64)).convert('L')).flatten() for i in list_path_new], 'f')

Final_reduced_1 = np.dot(new_matrix, vectors_32)

sep_data = {}

for i in range(len(list_labels)):

	index1 = list_dis_lab.index(list_labels[i])

	if index1 in sep_data:

		sep_data[index1].append(Final_reduced[i])

	else:

		sep_data[index1] = [Final_reduced[i]]

Final_reduced_norm_1 = Final_reduced_1 / Final_reduced_1.max(axis = 0)

def standard_dev(input1):

	avg = sum(input1) / float(len(input1))

	a = math.sqrt(sum([pow(i - avg, 2) for i in input1]) / float(len(input1) - 1))

	return a

def summary2(set1):

	summary = [(sum(attribute) / float(len(attribute)), standard_dev(attribute)) for attribute in zip(*set1)]

	return summary

def class_sum(dataset):

	summary1 = {}

	for val, inst in sep_data.items():

		summary1[val] = summary2(inst)

	return summary1

def calculate(x, mean, standard_deviation):

	a = (1 / (math.sqrt(2 * math.pi) * abs(standard_deviation))) * math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(abs(standard_deviation), 2))))

	return a

def predict(inputvec, summaries):

	probability = {}

	for class_val, summary1 in summaries.items():

		probability[class_val] = 1

		i = 0

		while i < len(summary1):

			k = inputvec[i]

			mean1, stdev1 = summary1[i]

			probability[class_val] *= calculate(k, mean1, stdev1)

			i += 1

	bestLab, bestProb = None, -1

	for val, prob in probability.items():

		if bestLab is None or prob > bestProb:

			bestProb = prob

			bestLab = val

	return bestLab

def getPredict(summaries, test):

	predict1 = []

	i = 0

	while i < len(test):

		result = predict(test[i], summaries)

		predict1.append(result)

		i += 1

	return predict1

summary_3 = class_sum(Final_reduced)

pred1 = getPredict(summary_3, Final_reduced_1)

for i in pred1:

	print(list_dis_lab[i])