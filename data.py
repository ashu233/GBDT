# -*- coding:utf-8 -*-
import numpy as np
import scipy.io as sio
import os

class Data:
	def __init__(self):
		mat_path = os.path.join('mldata', 'mnist-original.mat')
		mnist = sio.loadmat(mat_path)
		self.data = mnist['data']
		self.labels = mnist['label'][0]
		self.dataNum = len(self.labels)
		self.featNum = self.data.shape[0]
		self.featSet = range(0, self.featNum)
		self.idSet = range(0, self.dataNum)
		self.labelSet = set(self.labels)
		self.K = len(self.labelSet)

	def getInstance(self, Id, feat):
		return self.data[feat][Id]

	def getByFeat(self, idSet, feat):
		return self.data[feat][idSet]

	def getLabel(self, Id):
		return self.labels[Id]

	def getLabels(self, idSet):
		return self.labels[idSet]

