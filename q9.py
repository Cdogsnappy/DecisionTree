from sklearn import tree
import tree as t
import random
import matplotlib.pyplot as plot

def q_9():
    n_err = []
    n_tree_size = []
    n = [32,128,512,2048,8192]
    f = open('Dbig.txt', 'r')
    d = t.buildData(f)
    random.shuffle(d)
    g,l = build_skdata(d)
    test_data = g[8192:]
    test_labels = l[8192:]
    root = tree.DecisionTreeClassifier()
    for val in n:
        root = root.fit(g[:val],l[:val])
        n_err.append(len(test_data) - (root.score(test_data,test_labels)*len(test_data)))
        n_tree_size.append(root.tree_.node_count)
    plot.plot(n,n_err)
    plot.show()
    for i in range(len(n)):
        print("n = " + str(n[i]))
        print("n_err = " + str(n_err[i]))
        print("tree size = " + str(n_tree_size[i]))
        print("")
    t.q_7(d)


def build_skdata(d):
    g = []
    l = []
    for x in d:
        g.append([x[0],x[1]])
        l.append(x[2])
    return g,l

def test(test_set,tree):
    tree.predict(test_set)

def main():
    # Use a breakpoint in the code line below to debug your script.
    q_9()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()