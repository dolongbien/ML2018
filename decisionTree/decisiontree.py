import math
import copy
from collections import Counter


class Node:
    def __init__(self, root):
        self.root = root
        self.children = []

    def addChild(self, children):
        self.children.append(children)


class Example:
    def __init__(self):
        self.attributeDict = {}
        self.label = None

    def addAtribute(self, attribute, value):
        self.attributeDict[attribute] = value

    def setLabel(self, label):
        self.label = label


class DecisionTree:

    def __init__(self):
        self.examples = []
        self.labels = []
        self.attributeList = []
        self.valueListDict = {}
        self.tree = None

    def load_data(self, file):
        header_line = file.readline().split()
        for name in header_line[:-1:]:  # ignore the label name
            self.attributeList.append(name)
        for line in file:
            example = Example()
            valueList = line.strip().split()
            if valueList[-1].lower() == 'yes':
                example.setLabel(True)
            else:
                example.setLabel(False)
            # build label list of tree
            self.labels.append(example.label)

            for i, attribute in enumerate(self.attributeList):
                example.addAtribute(attribute=attribute, value=valueList[i])
                if attribute not in self.valueListDict:
                    self.valueListDict[attribute] = [valueList[i]]
                else:
                    if valueList[i] not in self.valueListDict[attribute]:
                        self.valueListDict[attribute].append(valueList[i])
                        # value list always a SET (distinct value)
            self.examples.append(example)
        print("Number of example:", len(self.examples))

    def train(self):
        self.tree = DTL(current_examples=self.examples,parentExamples=self.examples,attributes=self.attributeList,valueListDict=self.valueListDict)

    def accuracyTest(self):
        total = 0
        positive = 0
        for i in range(len(self.examples)):
            total += 1
            exampleSet = copy.deepcopy(self.examples)
            crossValidateExample = exampleSet[i]
            del exampleSet[i]
            decisionTree = DTL(exampleSet, exampleSet, self.attributeList, self.valueListDict)
            if testResult(decisionTree, crossValidateExample):
                positive += 1
        return positive / total * 100


def testResult(node, cv):
    for child in node.children:
        if type(child[1]) == bool:
            if child[1] == cv.label:
                return True
            else:
                return False
        else:
            if child[0] == cv.attributeDict[node.root]:
                return testResult(child[1], cv)

'''
function DTL(ex,attr,default);;; returns decision tree
   ;;; set of examples, of attributes, default for goal predicate
  if ex empty then return default
  elseif all ex have same classification then return it
  elseif attributes empty (but both positive and negative example) then return MajVal(ex)
  (When	all	attributes exhausted, assign majority label to the leaf node)
  // Examples have same description but different classification due to incorrect data (noise), not enough information, or nondeterministic domain
  else ChooseAttribute(attr,ex) -> best1
        new decision tree with root test best1 -> tree1
       for each value vi of best1 do
         {elements of ex with best1=vi} -> exi
         DTL(exi,attr,MajVal(ex)) -> subtree1
         add branch to tree with label vi
     subtree subtree1
       end
  return tree
'''


def DTL(current_examples, parentExamples, attributes, valueListDict):

    # BASE CASE of recursion
    if len(current_examples) == 0:
        return majority_label(parentExamples)
    elif len({*[example.label for example in current_examples]}) == 1:
        return current_examples[0].label
    elif len(attributes) == 0:
        return majority_label(current_examples)
    else:
        # RECURSIVE CASE
        '''1.  Calculate IG for all attribute, choose the best attribute with max IG'''
        best_attribute = attributes[0]
        best_score = IG(best_attribute, current_examples, valueListDict[best_attribute])
        for attribute in attributes:
            ig = IG(attribute=attribute, examples=current_examples, values_of_this_att=valueListDict[attribute])
            if ig > best_score:
                best_score = ig
                best_attribute = attribute
        '''2. Create a new node of best attribute as current root in current tree'''
        '''3. For each value v_i of best attribute, 
                    - 3.1 New examples set --> extract sub-examples has attribute A =  v_i
                    - 3.2 New attribute list --> remove chosen attribute from attribute list
                    - 3.3 DTL on new data --> recursively DTL(new example_set, new attribute_list)'''
        ''' Step 3--------------------------------'''
        tree = Node(best_attribute)
        for value_i in valueListDict[best_attribute]:
            sub_examples =  []
            ''' 3.1 New example set -----------'''
            for example in current_examples:
                # collect examples having value_i (i.e Sunny) on attribute (ie Outlook)
                if example.attributeDict[best_attribute] == value_i:
                    sub_examples.append(example)
            ''' 3.2 New attribute list ---------'''
            new_attribute_list = copy.deepcopy(attributes)
            new_attribute_list.remove(best_attribute)
            ''' 3.3 DTL on new data ---------'''
            child_tree = DTL(current_examples=sub_examples, attributes=new_attribute_list, valueListDict=valueListDict,
                             parentExamples=parentExamples)
            tree.addChild((value_i, child_tree))
        return tree

'''
for every attribute/feature:
       1.H(S) Calculate the entropy of current state S (example set)
       2.H(S,A_i) Calculate Remainder (entropy with respect to the attribute A_i)
       3.IG(S,A_i) = H(S) - H(S,A_i) : IG for current attribute
'''


def IG(attribute, examples, values_of_this_att):
    '''
        1. Each possible value in current attribute, filtrate examples whose Attribute=value (ie Outlook=sunny,...)
        2. Dictionary : value --> set of corresponding examples
            (ex Attr = Outlook, Dict:{sunny=[ex1,ex2,ex9,ex12], rain=[ex3,ex5], ...}
        3. Calculate IG for current attribute on filtrated examples
        '''
    ig = 0
    value_exampleDict = {} # each value in attribute ---> set of its examples

    '''-----------------Step 1,2'''
    for value in values_of_this_att: #valuelist is distinct values
        sub_examples = []
        for example in examples:
            # an example {Outlook=[sunny],Temp=[hot],Hum=[high],wind=[week],play=[no]}
            if example.attributeDict[attribute] == value:
                sub_examples.append(example)
        value_exampleDict[value] = sub_examples
    '''-----------------Step 3'''
    remainder = remainder_calculate(value_exampleDict, examples)
    entropy = entropy_calculate(examples)
    ig = entropy - remainder
    return ig


def remainder_calculate(exampleDict, examples):
    '''
    :param exampleDict: value ----> set of examples
    :param examples: current state of example set (S)
    :return: H(S,A = value)
    '''
    remainder = 0
    for value in exampleDict:
        probability = len(exampleDict[value])/len(examples)
        entropy = entropy_calculate(exampleDict[value])
        remainder += probability*entropy
    return remainder


def entropy_calculate(examples):
    '''
    :param examples: set of examples, p_i = (n_i/(pos+neg))
    :return: entropy value of current example set I(+,-) = -sum(p*log2(p))
    '''
    positive,negative = count_label(examples)
    entropyBit = 0
    if positive*negative == 0:
        return 0
    probability = [positive/len(examples), negative/len(examples)]
    for p in probability:
        entropyBit += p*math.log2(p)
    return (-1)*entropyBit


def count_label(examples):
    '''
    :param examples: example set
    :return: number of positive & negative example
    '''
    positive = 0
    negative = 0
    for example in examples:
        if example.label:
            positive += 1
        else:
            negative += 1
    return positive, negative


def majority_label(examples): # Majority==binary classifier; Plurality when it comes to Multi-classifier

    label_list = []
    for example in examples:
        label_list.append(example.label)
    data = Counter(label_list)
    return data.most_common(1)[0][0]


def print_tree(node, level):
    for child in node.children:
        if level > 0:
            print("|", end="")
        for i in range(level):
            print("\t", end="")

        if type(child[1]) == bool:
            print(node.root, "=", child[0] + ":", str(child[1]))
        else:
            print(node.root, "=", child[0])
            print_tree(child[1], level + 1)


def main():
    dt = DecisionTree()
    file = open('tennis.txt')
    dt.load_data(file)
    dt.train()
    print("------------ DECISION TREE --------------\n")
    print_tree(dt.tree, 0)
    print("Accuracy:", dt.accuracyTest(), "%")


if __name__ == '__main__':
    main()