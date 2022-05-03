from sklearn import metrics


def testResult(result_file="test_result.txt"):
    """

    :param result_file: 文件中每一行都是真实值 预测值的形式
    如：
    1 0
    1 1
    0 0
    """
    truelabels = []
    predicts = []
    with open(result_file, "r") as f:
        for i in f.readlines():
            i = i.split(" ")
            j = i[1].split("\n")
            truelabels.append(int(i[0]))
            predicts.append(int(j[0]))
    print(metrics.precision_score(truelabels, predicts))
    print(metrics.recall_score(truelabels, predicts))
    print(metrics.f1_score(truelabels, predicts))
