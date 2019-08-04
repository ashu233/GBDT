# -*- coding: utf-8 -*-
from data import *
from CART_regression_tree import *
import numpy as np
import random

class GBDT:
	def __init__(self):
		self.maxIter = 3
		self.maxDepth = 10
		self.treesList = None

	def initialize(self, data, idSet, Type):
		if Type == 'multi_classification':
			#初始化f, p, residual, c, y
			self.treesList = []
			f = {}
			residual = {}
			y = {}
			prob = {}
			c = {}
			for k in range(0, data.K):
				f[k] = {}
				residual[k] = {}
				y[k] = {}
				prob[k] = {}
				c[k] = {}
				for Id in idSet:
					f[k][Id] = 0.
					y[k][Id] = 1. if data.getLabel(Id) == k else 0.
					c[k][Id] = 0.
			prob = self.caculate_prob_Multi(data, idSet, f, prob)
			residual = self.caculate_residual_Multi(data, idSet, prob, y, residual)
			return f, residual, y, prob, c
		elif Type == 'binary_classification':
			#初始化y, f, residual, c
			self.treesList = []
			f = {}
			residual = {}
			c = {}
			y = {}
			for Id in idSet:
				f[Id] = 0.
				c[Id] = 0.
				if data.getLabel(Id)%2 == 0:
					#偶数为正类，奇数为负类
					y[Id] = 1
				else:
					y[Id] = -1
			residual = self.caculate_residual_Binary(idSet, y, f, residual)
			return f, residual, c, y

		elif Type == 'regression':
			#初始化y, residual, c
			pass
		else:
			print('输出参数有误')
			return
	#以下区域计算binary的更新函数
	def caculate_residual_Binary(self, idSet, y, f, residual):
		for Id in idSet:
			residual[Id] = y[Id]/(1. + np.exp(y[Id]*f[Id]))
		return residual

	def caculate_c_Binary(self, leafList, residual, c):
		for node in leafList:
			for Id in node.idSet:
				c[Id] = node.predictValue
		return c

	def update_f_Binary(self, idSet, c, f):
		for Id in idSet:
			f[Id] += c[Id]
		return f

	def binary_classification(self, data, idSet):
		f, residual, c, y= self.initialize(data, idSet, 'binary_classification')
		t = 0
		featSet = data.featSet
		depth = self.maxDepth
		while True:
			if t >= self.maxIter:
				print('到达迭代次数上限终止')
				return
			print('******************************************************************现在开始建立第' + str(t) + '棵树')
			tree = CART()
			tree.root = tree.constructTree(data, idSet, featSet, depth, residual, Type = 'binary_classification')
			self.treesList.append(tree)
			c = self.caculate_c_Binary(tree.leafList, residual, c)
			f = self.update_f_Binary(idSet, c, f)
			residual = self.caculate_residual_Binary(idSet, y, f, residual)
			self.precision_binary(idSet, f, y)
			t += 1

	def precision_binary(self, idSet, f, y):
		count = 0.
		for Id in idSet:
			if f[Id]*y[Id] > 0:
				count += 1
		print('**********************************************************本次预测准确率为：' + str(round(count/len(idSet)*100, 2)) + '%')
	#以下区域计算multi_classification的更新和预测
	def caculate_prob_Multi(self, data, idSet, f, prob):
		for Id in idSet:
			sums = 0.
			maxK = 0
			for k in range(0, data.K):
				if f[k][Id] > f[maxK][Id]:
					maxK = k
			for k in range(0, data.K):
				sums += np.exp(f[k][Id] - f[maxK][Id])
			for k in range(0, data.K):
				prob[k][Id] = np.exp(f[k][Id] - f[maxK][Id])/sums
		return prob
	def caculate_residual_Multi(self, data, idSet, prob, y, residual):
		for Id in idSet:
			for k in range(0, data.K):
				residual[k][Id] = y[k][Id] - prob[k][Id]
		return residual

	def caculate_c_Multi(self, leafList, c, k):
		for node in leafList:
			for Id in node.idSet:
				c[k][Id] = node.predictValue
		return c
	def update_f_Multi(self, data, idSet, f, c):
		for k in range(0, data.K):
			for Id in idSet:
				f[k][Id] += c[k][Id]
		return f
	def multi_classification(self, data, idSet):
		f, residual, y, prob, c = self.initialize(data, idSet, Type = 'multi_classification')
		t = 0
		depth = self.maxDepth
		featSet = data.featSet
		while True:
			if t >= self.maxIter:
				print('到达迭代次数上限终止')
				return
			print('**************************************************************************************************************现在开始建立第' + str(t) + '棵树')
			trees = {}
			for k in range(0, data.K):
				print('*******************************************************************现在开始建立第' + str(k) + '棵小树苗')
				tree = CART()
				tree.root = tree.constructTree(data, idSet, featSet, depth, residual[k], Type = 'multi_classification')
				trees[k] = tree
				c = self.caculate_c_Multi(tree.leafList, c, k)
			self.treesList.append(trees)
			f = self.update_f_Multi(data, idSet, f, c)
			prob = self.caculate_prob_Multi(data, idSet, f, prob)
			residual = self.caculate_residual_Multi(data, idSet, prob, y, residual)
			self.precision_Multi(data, idSet, f)

			t += 1

	def precision_Multi(self, data, idSet, f):
		count = 0.
		for Id in idSet:
			predictLabel = 0.
			for k in range(0, data.K):
				if f[k][Id] > f[predictLabel][Id]:
					predictLabel = k
			if predictLabel == data.getLabel(Id):
				count += 1
			else:
				print('样本' + str(Id) + '真实的label为：' + str(data.getLabel(Id)) +  ' ' +  '错误估计为：' + str(predictLabel))
		print('******************************************************************************************************************本次预测准确率为：' + str(round(count/len(idSet)*100, 2)) + '%')

	def getPredict_Multi(self, data, Id):
		score = {}
		for trees in self.treesList:
			for k in range(0, data.K):
				score[k] = score.get(k, 0) + trees[k].getPredict(data, Id, trees[k].root)
		bestK = 0
		for k in range(0, data.K):
			if score[k] > score[bestK]:
				bestK = k
		return bestK

	def precisionTest(self, data, idSetTest):
		count = 1.
		for Id in idSetTest:
			if self.getPredict_Multi(data, Id) == data.getLabel(Id):
				count += 1
		print('***********************************************************************************************************************测试集上的准确率为：' + str(round(count/len(idSetTest)*100, 2)) + '%')

model = GBDT()
data = Data()
idSet = random.sample(range(0, 50000), 2000)
model.multi_classification(data, idSet)
idSetTest = random.sample(range(0, 50000), 200)
model.precisionTest(data, idSetTest)


		

