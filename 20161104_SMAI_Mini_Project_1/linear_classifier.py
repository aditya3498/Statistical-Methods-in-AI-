from PIL import Image

import os, glob, re, sys

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

a = len(list_labels)

b = len(list_dis_lab)

x = np.zeros((a, b))

allfiles = list_path

for file in allfiles:

	list_img.append(file)

matrix = np.asarray([np.array(Image.open(i).convert('L').resize((64, 64))).flatten() for i in list_img], 'f')

final_matrix = matrix - matrix.mean(axis = 0)

M = np.cov(final_matrix.T)

e, EV = np.linalg.eigh(M)

idx = e.argsort()[::-1]

e = e[idx]

EV = EV[:,idx]

vectors_32 = EV.T[:32].T

Final_reduced = np.dot(matrix, vectors_32)

c = 0

for i in list_labels:

	index1 = list_dis_lab.index(i)

	x[c][index1] = 1

	c += 1

w = np.random.rand(Final_reduced.shape[1], len(list_dis_lab))

Final_reduced_norm = Final_reduced / Final_reduced.max(axis = 0)

n = 0.15

for i in range(100000):

	m = Final_reduced_norm.shape[1]

	phi = Final_reduced_norm.dot(w)

	smax = (np.exp(phi.T) / np.sum(np.exp(phi), axis = 1)).T

	loss = (-1 / m) * np.sum(x * np.log(smax))

	gradient = (-1 / m) * np.dot(Final_reduced_norm.T, (x - smax))

	losses.append(loss)

	w = w - (n * gradient)

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

new_matrix = np.asarray([np.array(Image.open(i).convert('L').resize((64, 64))).flatten() for i in list_path_new], 'f')

Final_reduced_1 = np.dot(new_matrix, vectors_32)

Final_reduced_norm_1 = Final_reduced_1 / Final_reduced_1.max(axis = 0)

value = np.dot(Final_reduced_norm_1, w)

smax1 = (np.exp(value.T) / np.sum(np.exp(value), axis = 1)).T

preds = np.argmax(smax1, axis = 1)

for i in preds:

	print(list_dis_lab[i])