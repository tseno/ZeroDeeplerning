# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


# 学習済みデータの読み込み(W1,W2,W3,b1,b2,b3が入っている)
def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        # pickleは、pythonのオブジェクトをそのままファイルに入出力する仕組み。
        network = pickle.load(f)
    return network


def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # 入力値と第一層の重みの行列計算＋バイアス
    a1 = np.dot(x, w1) + b1
    # シグモイド関数
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    # ソフトマックス関数
    y = softmax(a3)

    return y


# MNISTデータセットの読み込み
x, t = get_data()
# 学習済みデータの読み込み
network = init_network()

batch_size = 100  # バッチの数
accuracy_cnt = 0  # 正解数

# バッチサイズごとに処理を行う
for i in range(0, len(x), batch_size):
    x_batch = x[i:i + batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)  # 最も確率の高い要素のインデックスを取得
    accuracy_cnt += np.sum(p == t[i:i + batch_size])

# 正解率の表示
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
