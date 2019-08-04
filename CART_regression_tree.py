# -*- coding: utf-8 -*-
from data import *
import numpy as np
import random

class leafNode:
	def __init__(self):
		self.predictValue = None
		self.idSet = None

class internalNode:
	def __init__(self):
		self.feat = None
		self.c = None
		self.lChild = None
		self.rChild = None

class CART:
	def __init__(self, maxDepth = 20, maxFeatNum = 10):
		self.maxDepth = maxDepth
		self.maxFeatNum = maxFeatNum
		self.leafList = []
		self.minScore = 0.0000001
		self.root = None

	def bestDivision(self, data, idSet, featSet, residual):
		#计算sumLeft*sumLeft/countLeft + sumRight*sumRight/countRight得分高的切分结果胜出
		#这个计算公式由平方损失推导出
		if not idSet:
			print('样本集合为空, bestDivision函数出错')
			return

		if len(featSet) > self.maxFeatNum:
			featSet = random.sample(featSet, self.maxFeatNum)
		bestScore = 0.
		bestFeat = None
		bestC = None
		lenAll = len(idSet)
		for feat in featSet:
			featValueSet = set(data.getByFeat(idSet, feat))
			for c in featValueSet:
				sumLeft = 0.
				sumRight = 0.
				countLeft = 0.
				for Id in idSet:
					if data.getInstance(Id, feat) <= c:
						sumLeft += residual[Id]
						countLeft += 1
					else:
						sumRight += residual[Id]
				countRight = lenAll - countLeft
				if countLeft == 0:
					score = sumRight*sumRight/countRight
				elif countRight == 0:
					score = sumLeft*sumLeft/countLeft
				else:
					score = sumRight*sumRight/countRight + sumLeft*sumLeft/countLeft
				if score > bestScore:
					bestScore = score
					bestFeat = feat
					bestC = c
		if not bestFeat:
			#筛选特征报警系统
			print('特征筛选出错，此时idset为：')
			print(idSet)
			print('得分为：' + str(score) + 'sumRight:' + str(sumRight) + 'sumLeft:' + str(sumLeft))
		return bestFeat, bestC, bestScore

	def constructTree(self, data, idSet, featSet, depth, residual, Type):
		if not idSet:
			return

		#print('当前的深度为：' + str(depth))
		if depth == 0:
			#到达叶节点
			node = leafNode()
			node.idSet = idSet
			node.predictValue = self.predict(data, idSet, residual, Type)
			self.leafList.append(node)
			return node

		bestFeat, bestC, bestScore = self.bestDivision(data, idSet, featSet, residual)
		if not bestFeat or bestScore <= self.minScore:
			#得分小或者没有选出特征，实际上是因为残差过于小，此时分类结果已经很完美无需改动
			node = leafNode()
			node.idSet = idSet
			node.predictValue = self.predict(data, idSet, residual, Type)
			self.leafList.append(node)
			return node

		#print('本轮选出的最佳特征为：' + str(bestFeat) + '最佳切分点为：' + str(bestC) + '得分为：' + str(bestScore))
		left = []
		right = []
		for Id in idSet:
			if data.getInstance(Id, bestFeat) <= bestC:
				left.append(Id)
			else:
				right.append(Id)
		node = internalNode()
		node.feat = bestFeat
		node.c = bestC
		node.lChild = self.constructTree(data, left, featSet, depth - 1, residual, Type)
		node.rChild = self.constructTree(data, right, featSet, depth - 1, residual, Type)
		return node

	def predict(self, data, idSet, residual, Type):
		if Type == 'binary_classification':
			result = 0.
			sums = 0.
			for Id in idSet:
				result += residual[Id]
				sums += abs(residual[Id])*(1 - abs(residual[Id])) if abs(residual[Id]) >= 0.95 else 0.95*0.05

			if not sums == 0.:
				predictValue = result/sums
				print('得到估计值：' + str(predictValue))
				return predictValue
			else:
				print('predict函数出错')

		elif Type == 'multi_classification':
			result = 0.
			sums = 0.
			for Id in idSet:
				result += residual[Id]
				sums += abs(residual[Id])*(1 - abs(residual[Id])) if abs(residual[Id]) >= 0.95 else 0.95*0.05

			if not sums == 0.:
				predictValue = (data.K - 1)/data.K*result/sums
				#print('得到估计值：' + str(predictValue))
				return predictValue
			else:
				print('predict函数出错')

	def getPredict(self, data, Id, node):
		if not node:
			print('样本号' + str(Id) + '预测失败')
			return 0

		if isinstance(node, leafNode):
			return node.predictValue

		if data.getInstance(Id, node.feat) <= node.c:
			return self.getPredict(data, Id, node.lChild)
		else:
			return self.getPredict(data, Id, node.rChild)

"""
data = Data()
idSet = random.sample(range(0, 12000), 5000)
featSet = data.featSet
depth = 20

residual = {}
for Id in idSet:
	if data.getLabel(Id) == 0.:
		residual[Id] = -1./2
	else:
		residual[Id] = 1./2
tree = CART()
tree.constructTree(data, idSet, featSet, depth, residual)
"""
