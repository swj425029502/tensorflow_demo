#coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# mnist=input_data.read_data_sets('./data/',one_hot=True)

nums = [2, 7, 11, 15]
target = 9


class Solution:
    def __init__(self,Li,target):
        self.__init__(Li,target)
        self.Li=list(Li)
        self.target=target
    def twoSum(Li,target):
        for i in range(len(Li)):
            result = target - Li[i]
            if result in Li:
                print(i, Li.index(result))
                break


if __name__ == '__main__':
    s = Solution
    Li = [2, 7, 11, 15]
    target = 9
    s.twoSum(Li, target)

