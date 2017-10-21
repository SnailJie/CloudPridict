from sklearn import linear_model  
import numpy as np  
from numpy import genfromtxt   
  
datapath = "../dataTest.csv"  
deliverData = genfromtxt(datapath,delimiter=",")   
  
print "data:",deliverData  
