import argparse

from sklearn import metrics


def testResult(result_file):
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
    print("precision_score:", metrics.precision_score(truelabels, predicts))
    print("recall_score:", metrics.recall_score(truelabels, predicts))
    print("f1_score:", metrics.f1_score(truelabels, predicts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file", type=str, default="test_result.txt")
    hp = parser.parse_args()
    testResult(hp.result_file)
