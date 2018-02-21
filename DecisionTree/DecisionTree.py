#!/usr/bin/env python3
import sys
import random
import numpy as np
from sklearn.model_selection import KFold
from collections import Counter

# Class for instances with operations
class Instances(object):
    def __init__(self):
        self.label = []
        self.attrs = []
        self.num_attrs = -1
        self.num_instances = 0
        self.attr_set = []
        
        #self.label_set = []
        

    def add_instance(self, _lbl, _attrs):
        self.label.append(_lbl)
        self.attrs.append(_attrs)
        if self.num_attrs == -1:
            self.num_attrs = len(_attrs)
        else:
            assert(self.num_attrs == len(_attrs))
        self.num_instances += 1
        assert(self.num_instances == len(self.label))

    
    def make_attr_set(self):
        self.attr_set = [set([self.attrs[i][j] for i in range(self.num_instances)]) for j in range(self.num_attrs)]
        
    #def make_label_set(self):
        #self.label_set = set([self.label[i] for i in range(self.num_instances)])


    def load_file(self, file_name):
        with open(file_name, 'r') as f:
            for line in f:
                data = line.strip().split(',')
                self.add_instance(data[0], data[1:])
        self.make_attr_set()
        #self.make_label_set()
        return self
    

    def split(self, att_idx):
        assert(0 <= att_idx < self.num_attrs)
        split_data = {x: Instances() for x in self.attr_set[att_idx]}
        for i in range(self.num_instances):
            key = self.attrs[i][att_idx] # ith data's att_index's value
            split_data[key].add_instance(self.label[i], self.attrs[i])
        for key in split_data:
            split_data[key].attr_set = self.attr_set
            #split_data[key].label_set= self.label_set
        return split_data
    

    def shuffle(self):
        indices = list(range(len(self.label)))
        random.shuffle(indices)  #shuffle a 3(from 1 to 3) value array --> get array [3,1,2]/or other order radomly
        res = Instances()
        for x in indices:
            res.add_instance(self.label[x], self.attrs[x])
        res.attr_set = self.attr_set
        #add@
        #res.label_set = self.label_set
        return res


    def get_subset(self, keys):
        res = Instances()
        for x in keys:
            res.add_instance(self.label[x], self.attrs[x])
        res.attr_set = self.attr_set
        #add@
        #res.label_set = self.label_set
        return res


def compute_entropy(data): #data is a Instances()
    total_entropy = 0.0
    ########## Please Fill Missing Lines Here ##########
    
    label_ratio = 0
    label_map = {} #label and its count 
    #create label set (with distinct labels), move to instance class function
    label_set = set([data.label[i] for i in range(data.num_instances)]) 
    #initialization
    label_map = {label_key: 0 for label_key in label_set}
    for instance_label in data.label:
        label_map[instance_label] = label_map[instance_label] + 1
    
    #calculate label_ratio
    for label_index in label_set:
        label_ratio = label_map[label_index] / data.num_instances
        if label_ratio != 0:
            total_entropy = total_entropy - label_ratio * np.log2(label_ratio)

    return total_entropy
    

def compute_info_gain(data, att_idx): # this data only have the proper data after split
    info_gain = 0.0
    info = 0.0
    #create label set (with distinct labels), move to instance class function
    #label_set = set([data.label[i] for i in range(data.num_instances)]) 
    
    attrValues_set = data.attr_set[att_idx] #is a set
    
    ########## Please Fill Missing Lines Here ##########
    attrValue_count_map = {} #attr_value of this attr and how many data has the value
    attrValue_count_map = {x: 0 for x in attrValues_set}  #initialization
    attrValue_label_map = {} #attr_value and its labels that have this attr_value
    attrValue_label_map = {y: [] for y in attrValues_set}  #initialization
    #get attrValue_count_map, and attrValue_label_map, traverse all the data
    for dataIndex in range(0, data.num_instances):
        #attr_value = data.attrs[dataIndex,att_idx]
        attr_value = (data.attrs[dataIndex])[att_idx]
        attrValue_count_map[attr_value] =  attrValue_count_map[attr_value] + 1
        attrValue_label_map[attr_value].append(data.label[dataIndex])

    #for attr_value in self.attrs[:,att_idx]:
    #    attrValue_count_map[attr_value] =  attrValue_count_map[attr_value] + 1
    #calculate attr_value_ratio
   
    for value_temp in attrValues_set:
        label_count_map = {} #label number in this value: key-label; value-number of this label for this attrValue
        attrValue_ratio = attrValue_count_map[value_temp] / data.num_instances
        #get label ratio in this attriValue
        #initialization
        #for label_temp in attrValue_label_map[value_temp]:  #attrValue_label_map[value_temp] is a array that contain the labels
            #label_count_map[label_temp] = 0
            #label_count_map = {label_temp: 0}
        label_count_map = {label_temp: 0 for label_temp in attrValue_label_map[value_temp]}
        #print ('attrValue_label_map[value_temp]')
        #print (attrValue_label_map[value_temp])
        #print ('label_count_map')
        #print (label_count_map)
        for label_temp1 in attrValue_label_map[value_temp]:
            #print('label_temp1')
            #print(label_temp1)
            label_count_map[label_temp1] = label_count_map[label_temp1] + 1
        
        temp_entropy = 0 #temp_entropy is I(2,3) in formula
        for label_key in label_count_map:
            ratio = label_count_map[label_key] / len(attrValue_label_map[value_temp])
            if ratio != 0:
                temp_entropy = temp_entropy - ratio * np.log2(ratio)
            
        info = info + attrValue_ratio * temp_entropy
      
    info_gain = compute_entropy(data) - info
    
    return info_gain


def comput_gain_ratio(data, att_idx):
    gain_ratio = 0.0
    splitInfo = 0.0
    ########## Please Fill Missing Lines Here ##########    
    attrValues_set = data.attr_set[att_idx] #is a set
    
    ########## Please Fill Missing Lines Here ##########
    attrValue_count_map = {} #attr_value of this attr and how many data has the value
    attrValue_count_map = {x: 0 for x in attrValues_set}  #initialization
    
    for dataIndex in range(0, data.num_instances):
        #attr_value = data.attrs[dataIndex,att_idx]
        attr_value = (data.attrs[dataIndex])[att_idx]
        attrValue_count_map[attr_value] =  attrValue_count_map[attr_value] + 1
    
    for value_temp in attrValues_set:
        attrValue_ratio = attrValue_count_map[value_temp] / data.num_instances
        if attrValue_ratio != 0:
            splitInfo = splitInfo - attrValue_ratio * np.log2(attrValue_ratio)
    
    if splitInfo != 0:  #if == 0, it's a leaf node
        gain_ratio = compute_info_gain(data, att_idx) / splitInfo
    
    return gain_ratio


# Class of the decision tree model based on the ID3 algorithm
class DecisionTree(object): #object means send in many argument: from __init__ we know, object contain _instances and _sel_func
    def __init__(self, _instances, _sel_func):
        self.instances = _instances
        self.sel_func = _sel_func
        self.gain_function = compute_info_gain if _sel_func == 0 else comput_gain_ratio
        self.m_attr_idx = None # The decision attribute if the node is a branch
        self.m_class = None # The decision class if the node is a leaf
        self.make_tree()

    def make_tree(self): #self is the split_data's instances
        if self.instances.num_instances == 0:
            # No any instance for this node
            self.m_class = '**MISSING**'
        else:
            gains = [self.gain_function(self.instances, i) for i in range(self.instances.num_attrs)]
            #print ('gains')
            #print (gains)
            self.m_attr_idx = np.argmax(gains) #Returns the indices of the maximum value. select new sttribute
            if np.abs(gains[self.m_attr_idx]) < 1e-9:
                # A leaf to decide the decided class
                self.m_attr_idx = None
                ########## Please Fill Missing Lines Here ##########
                #now, all the instances have the same label
                self.m_class = self.instances.label[0]
            else:
                # A branch
                split_data = self.instances.split(self.m_attr_idx) # put all data in map, classify by value of this attr
                self.m_successors = {x: DecisionTree(split_data[x], self.sel_func) for x in split_data}
                for x in self.m_successors:
                    self.m_successors[x].make_tree()

    def classify(self, attrs):
        assert((self.m_attr_idx != None) or (self.m_class != None))
        if self.m_attr_idx == None:
            return self.m_class
        else:
            return self.m_successors[attrs[self.m_attr_idx]].classify(attrs)
            
 

if __name__ == '__main__':
    if len(sys.argv) < 1 + 1:
        print('--usage python3 %s data [0/1, 0-Information Gain, 1-Gain Ratio, default: 0]' % sys.argv[0], file=sys.stderr)
        sys.exit(0)
    random.seed(27145)
    np.random.seed(27145)
    
    sel_func = int(sys.argv[2]) if len(sys.argv) > 1 + 1 else 0
    assert(0 <= sel_func <= 1) 

    data = Instances().load_file(sys.argv[1])
    data = data.shuffle()
    print (data)
    print ('data')
    
    # 5-Fold CV
    kf = KFold(n_splits=5)
    n_fold = 0
    accuracy = []
    for train_keys, test_keys in kf.split(range(data.num_instances)):
        train_data = data.get_subset(train_keys)
        test_data = data.get_subset(test_keys)
        n_fold += 1
        model = DecisionTree(train_data, sel_func)
        predictions = [model.classify(test_data.attrs[i]) for i in range(test_data.num_instances)]
        num_correct_predictions = sum([1 if predictions[i] == test_data.label[i] else 0 for i in range(test_data.num_instances)])
        nfold_acc = float(num_correct_predictions) / float(test_data.num_instances)
        accuracy.append(nfold_acc)
        print('Fold-{}: {}'.format(n_fold, nfold_acc))

    print('5-CV Accuracy = {}'.format(np.mean(accuracy)))
        
