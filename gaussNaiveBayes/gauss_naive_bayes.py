import  csv
import  re
from collections import defaultdict
import random
import statistics as stc
import math
from google.colab import files

'''
    - Input: data table (supervised learning)
    - Output: Gaussian Naive Bayes classifier
'''


class GaussNaiveBayes:

    def __init__(self):
        self.summary_dict = {}

    def load_data(self, data, header):

        dt = csv.reader(data, delimiter = ',')
        dataset = list(dt)
        if header:
            dataset = dataset[1:]
        for i in range(len(dataset)):
            # convert float string to float on each row, label still remains string
            dataset[i] = [float(value) if re.search('\d', value) else value for value in dataset[i]]

        return dataset

    def split_training_set(self, dataset, test_size):

        random.shuffle(dataset)
        split1 = int(test_size*len(dataset))
        train_set = dataset[:split1]
        test_set = dataset[split1:]

        return [train_set, test_set]

    def mapping_class(self, training_set):
        # initialize dict: key ---> list
        class_dict = defaultdict(list)
        # each row of training examples
        for i in range(len(training_set)):
            row = training_set[i]
            if not row:
                continue

            className = row[-1]
            class_dict[className].append(row[:-1])
        return dict(class_dict)

    def train(self, training_set):
        group_dict = self.mapping_class(training_set)
        for class_name, row_data_list in group_dict.items():
            class_desc = {
                'prior_probability': self.prior_probability(group_dict, class_name, training_set),
                'describe': [s for s in self.describe(row_data_list)]
            }
            self.summary_dict[class_name]= class_desc
        return self.summary_dict

    def prior_probability(self, group_dict, class_name, training_set):
        sample = float(len(training_set))
        class_count = len(group_dict[class_name])
        return class_count/sample

    def posterior_probability(self, evident_line):
        posterior_dict = {}
        joint_prob_dict = self.joint_probability(evident_line)
        # ignore denominator (margin prob)
        for class_name, join_prob in joint_prob_dict.items():
            posterior_dict[class_name] = join_prob # dont care about denominator
        return posterior_dict

    def joint_probability(self, evident_line):
        joint_dict = {}
        for class_name, desc_info in self.summary_dict.items():
            number_of_attribute = len(desc_info['describe']) # return 4 for iris dataset
            likelihood = 1
            for i in range(number_of_attribute):
                if not evident_line:
                    continue
                attribute_value = evident_line[i]
                mean = desc_info['describe'][i]['mean']
                std = desc_info['describe'][i]['std']
                predictor_prob = self.gauss_probability(attribute_value, mean, std)
                likelihood *= predictor_prob
            prior_probability  = desc_info['prior_probability']
            joint_dict[class_name] = prior_probability*likelihood # ignore denominator 
        return joint_dict

    def describe(self, row_data_list):
        '''
        :param row_data_list: list of multiple row values [ [],[],[],...]
        :return: describe (mean, std) of an attribute (column)
        '''
        for att_val_list in zip(*row_data_list): # unpack 1 level
            yield {
                'mean': stc.mean(att_val_list),
                'std':  stc.stdev(att_val_list)
            }

    def gauss_probability(self, x, mean, std):
        variance = std**2
        A = (x - mean) ** 2
        A = -A/(2*variance)
        A = math.e ** A
        B = ((2*math.pi)**.5)*std
        return A/B

    def predict_MLE(self, aRow):
        posterior_prob_dict = self.posterior_probability(aRow)
        predicted_class = max(posterior_prob_dict, key=posterior_prob_dict.get)
        return predicted_class

    def gnb_classify(self, test_set):

        predicted_list = []
        for line in test_set:
            prediction = self.predict_MLE(line)
            predicted_list.append(prediction)
        return predicted_list


    def accuracy(self, test_set, predicted_set):
        correct = 0
        label = [item[-1] for item in test_set]
        for x,y in zip(label, predicted_set):
            if x == y:
                correct += 1
        return 1.0*correct/ len(test_set)

def main():

    gnb = GaussNaiveBayes()
    uploaded = files.upload()
    data = open('iris.csv')
    dataset = gnb.load_data(data, False)
    train_set, test_set = gnb.split_training_set(dataset, 0.6)
    gnb.mapping_class(train_set)
    gnb.train(train_set)
    print(train_set[:5], sep = '\n')
    p_list = gnb.gnb_classify(test_set)
    acc = 100*gnb.accuracy(test_set, p_list)
    print(acc)

if __name__ == '__main__':
    main()