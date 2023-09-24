import numpy as np
import matplotlib.pyplot as plot

class DecisionTree(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.data = None


def determineBestSplit(d):
    best_split = [0, 0]
    for x in d:
        curr_split = calculateGainRatio(d, x)
        if curr_split[0] == -1:
            return curr_split
        if curr_split[1] > best_split[1]:
            best_split = curr_split
    return best_split


def calculateGainRatio(d, x):
    l = [0, 0]
    lt = [0, 0]
    r = [0, 0]
    rt = [0, 0]
    for y in d:
        for i in range(2):
            if x[i] <= y[i]:
                r[i] += 1
                if int(y[2]) == 1:
                    rt[i] += 1
            else:
                l[i] += 1
                if int(y[2]) == 1:
                    lt[i] += 1
    total1s = [lt[0] + rt[0], lt[1] + rt[1]]
    total = l[0] + r[0]
    split_gain_ratios = [0, 0]
    for i in range(2):
        if l[i] == 0 or r[i] == 0:
            continue
        if total1s[i] == 0 or total1s[i] == total:
            return [-1,0,x]
        totals_arr = np.log2([total1s[i] / total, (total - total1s[i]) / total])
        total_gain_ratio = -1 * ((total1s[i] * totals_arr[0]) / total + ((total - total1s[i]) * totals_arr[1]) / total)
        entropy_logs = np.log2([l[i]/total, r[i]/total])
        entropy = -1 * ((l[i]*entropy_logs[0])/total + (r[i]*entropy_logs[1])/total)
        if entropy == 0:
            return [-1, 0, x]
        logs = np.log2([lt[i] / l[i], (l[i] - lt[i]) / l[i], rt[i] / r[i], (r[i] - rt[i]) / r[i]])
        if lt[i] == 0 or lt[i] == l[i]:
            split_ratiol = 0
        else:
            split_ratiol = -1 * ((lt[i] * logs[0]) / l[i] + (((l[i] - lt[i]) * logs[1]) / l[i]))
        if rt[i] == 0 or rt[i] == r[i]:
            split_ratior = 0
        else:
            split_ratior = -1 * ((rt[i] * logs[2]) / r[i] + (((r[i] - rt[i]) * logs[3]) / r[i]))
        split_gain_ratios[i] = ((total_gain_ratio - ((l[i] * split_ratiol) / total + (r[i] * split_ratior) / total))/entropy)
    if split_gain_ratios[0] > split_gain_ratios[1]:
        return [0, split_gain_ratios[0], x]
    else:
        return [1, split_gain_ratios[1], x]


def generateTree(root, d):
    if not d:  # empty node condition
        root.data = 1
        return
    best_split = determineBestSplit(d)
    if best_split[1] == 0 or best_split[0] == -1:
        root.data = highestPlurality(d)  # other two leaf conditions
        return
    root.data = best_split
    root.left = DecisionTree()
    root.right = DecisionTree()
    subsets = d_subset(best_split, d)
    generateTree(root.left, subsets[0])
    generateTree(root.right, subsets[1])


def d_subset(split, d):
    ind = split[0]
    threshold = split[2][split[0]]
    left_set = []
    right_set = []
    for x in d:
        if x[ind] < threshold:
            left_set.append(x)
        else:
            right_set.append(x)
    return [left_set, right_set]


def highestPlurality(d):
    count = 0
    num1s = 0
    for x in d:
        count += 1
        num1s += x[2]
    if count - num1s >= count // 2 - 1:
        return 0
    else:
        return 1


def classify(root, x):
    curr_node = root
    while True:
        if curr_node.left is None and curr_node.right is None:
            return curr_node.data
        if x[curr_node.data[0]] <= curr_node.data[2][curr_node.data[0]]:
            curr_node = curr_node.left
        else:
            curr_node = curr_node.right


def buildData(f1):
    d = []
    for line in f1:
        list1 = [float(number) for number in line.split(' ')]
        d.append(list1)
    return d

def visualizeBoundary(d, root):
    colors = ['red','blue']
    for x in d:
        plot.scatter(x[0], x[1], color = colors[int(x[2])])
    l = getLeaves(0, 0, root)
    print("num leaves = " + str(len(l)))
    for p in l:
        plot.scatter(p[0],p[1],color = 'green')
    plot.show()

def getLeaves(x, y, root):
    if root.left is None and root.right is None:
        return [[x,y]]
    if root.data[0] == 0:
        x = root.data[2][0]
    else:
        y = root.data[2][1]
    l1 = getLeaves(x, y, root.left)
    l2 = getLeaves(x, y, root.right)
    return l1 + l2


def main():
    # Use a breakpoint in the code line below to debug your script.
    f = open('Dbig.txt', 'r')
    d = buildData(f)
    root = DecisionTree()
    generateTree(root,d)
    visualizeBoundary(d, root)
    highestPlurality(d)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# INSTANCE: [feature_1,feature_2,label]
# SPLIT: [feature_index,gain_ratio,instance]
