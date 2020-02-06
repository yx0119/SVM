import numpy as np
import pandas as pd
import svm

# 保证所有数据能够显示，而不是用省略号表示，np.inf表示一个足够大的数
np.set_printoptions(threshold = np.inf)

# 若想不以科学计数显示:
np.set_printoptions(suppress = True)

def F1_test(predictor, test_X_set, test_y_set = None, init_set = None):

    if test_y_set == None:
        with open("test_out.csv", "w") as f:
            for i in range(0, len(test_X_set)):
                #print(i)
                #print(test_X_set[i])
                ans = predictor.predict(test_X_set[i])
                f.write(str(init_set[i]) + "," + str(ans) + '\n')
                #print(ans)
        return


    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(0, len(test_X_set)):
        ans = predictor.predict(test_X_set[i])
        #print("ans:" + str(ans))
        if ans > 0:
            if test_y_set[i] > 0:
                TP += 1
            else:
                FP += 1
        else:
            if test_y_set[i] > 0:
                FN += 1
            else:
                TN += 1
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = (2 * precision * recall) / (precision + recall)
    f_log.write("TP:" + str(TP) + "\n")
    f_log.write("FP:" + str(FP) + "\n")
    f_log.write("FN:" + str(FN) + "\n")
    f_log.write("TN:" + str(TN) + "\n")
    f_log.write("F1:" + str(f1) + "\n")
    print("F1:" + str(f1) + "\n")


def run(X, y, train_rate, kernel, C, max_iter, gama):
    f_log.write("this is para{train_set_rate:" + str(train_rate) + " kernel:" + str(kernel) + " C:"+str(C) +
                " max_iter:"+str(max_iter) + " gama:" + str(gama) + "}" + "\n")
    n = len(X)
    xx = int(n*train_rate)
    train_X_set = X[:xx]
    train_y_set = y[:xx]
    test_X_set = X[xx:]
    test_y_set = y[xx:]
    svm_Model = svm.SVM(kernel, C)
    svm_Model.set_gama(gama)
    alpha, b = svm_Model.train(train_X_set, train_y_set, max_iter)
    f_log.write("alpha:" + str(alpha) + "\n")
    f_log.write("b:" + str(b) + "\n")
    f_log.write("train_rate:" + str(train_rate) + "\n")
    if kernel == 'linear':
        w = np.dot(train_X_set.T, np.multiply(alpha, train_y_set))
        np.save('./linear_w.npy', w)
        np.save('./linear_b.npy', b)
    else:
        np.save('./rbf_alpha.npy', alpha)
        np.save('./rbf_b.npy', b)
    predictor = svm.Predictor(kernel, alpha, b, train_y_set, train_X_set, gama)

    #F1_test(predictor, test_X_set, test_y_set)


def get_data(file):
    Nominal = ['x2', 'x5', 'x6', 'x7', 'x8', 'x9']
    Ordinal = ['x4']
    Ratio = ['x1', 'x3', 'x10', 'x11', 'x12']
    df = pd.read_csv(file)
    df.drop(columns=['index'])
    y = df['label']
    train = df[Ordinal + Nominal + Ratio]
    train_norm = (train - train.min()) / (train.max() - train.min())
    np_train = np.array(train_norm)
    np_y = np.array(y)
    return np_train, np_y

if __name__ == '__main__':
    file = "./svm_training_set.csv"
    log = "./train_log.txt"
    f_log = open(log, "a+")
    np_train, np_y = get_data(file=file)
    train_rate = 0.7
    kernel = 'rbf'
    kernel = 'linear'
    f_log.write("**************************" + "\n")
    run(np_train, np_y, 1, kernel, 2, 15, 0.5)

    #alpha = np.load('./alpha.npy')
    #b = -2.6099299394676434
    '''
    w1 = np.load('./linear_w.npy')
    b1 = np.load('./linear_b.npy')
    predictor1 = svm.Predictor(kernel, None, b1, np_y, np_train, 0.5, w1)
    F1_test(predictor1, np_train, np_y)

    
    w2 = np.load('./models/linear_w_10.npy')
    b2 = np.load('./models/linear_b_10.npy')
    predictor1 = svm.Predictor(kernel, None, b2, np_y, np_train, 0.5, w2)
    F1_test(predictor1, np_train, np_y)
    '''
