# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 11:11:29 2020

@author: myy388
"""
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data[:,[2,3]]
#花の種類
y = iris.target
#クラスラベルが整数値で格納　setosa = 0, versicolor = 1, virginica = 2

#datasetを、30%のテストデータと70%のトレーニングデータに分割(ランダム)
#train_test_splitのシャッフルにランダムシードを使用
#stratify = y y(正解ラベル)と同じ比率のデータセットを作る
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3, random_state = 1, stratify = y)

sc = StandardScaler()
#トレーニングデータの特徴量を標準化
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

ppn = Perceptron(max_iter = 40, eta0 = 0.1, random_state = 3)
ppn.fit(X_train_std,y_train)

y_pred = ppn.predict(X_test_std)

#変換指定子は整数が%d、浮動小数点が%f、文字列が%s
print('misclassified samples:%d'%(y_test != y_pred).sum())
print('Accuracy:%.2f'%accuracy_score(y_test,y_pred))

def plot_decision_regions(X, y,Classifier,test_idx=None,resolution = 0.02):
    #マーカーとカラーマップの準備
    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    #ListedColormapで、colorsの2列目までの色取得
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    #決定領域のプロット-1,+1はそれぞれ余裕を持つため
    #x[:,0]は、全行1列
    x1_min, x1_max = X[:,0].min() - 1,X[:,0].max() + 1
    x2_min, x2_max = X[:,1].min() - 1,X[:,1].max() + 1
    #グリッドポイントの設定 np.arange(start,stop,step)
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),
                          np.arange(x2_min,x2_max,resolution))
    #各特徴量を1次元配列に変換して予測を実行 .T=転置
    #np.array([xx1.ravel(),xx2.ravel()]).Tは、各格子点がn行2列の形で格納されている
    Z = Classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    #予測結果をグリッドポイントのデータサイズに変換
    #reshape(list) listの形に変換
    z = Z.reshape(xx1.shape)
    #グリッドポイントの等高線のプロット contour=等高線 f 塗りつぶし
    #(xx1,xx2)座標の各点に対し、zで高さを与える
    plt.contourf(xx1,xx2, z, alpha = 0.3,cmap = cmap)
    #軸の範囲の設定
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())
    
    #クラスごとにサンプルをプロット　enumerate インデックスも同時に取得
    #X[y == cl,0]　Xの対応するyがclのときの1行目全て
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y == cl, 0],
                    y = X[y == cl, 1],
                    alpha = 0.8,
                    c = colors[idx],
                    marker = markers[idx],
                    label = cl,
                    edgecolor='black')
    #テストサンプル全てを目立たせる
    if test_idx:
        X_test,y_test = X[test_idx,:],y[test_idx]
        plt.scatter(X_test[:,0],X_test[:,1],
                    c='',
                    edgecolor='black',
                    alpha=1,
                    linewidth=1,
                    marker='o',
                    s=100,
                    label='test set')
#トレーニングデータとテストデータの特徴量を行方向に結合
X_combined_std = np.vstack((X_train_std,X_test_std))
y_combined = np.hstack((y_train,y_test))

#決定境界のプロット
plot_decision_regions(X_combined_std,y_combined,Classifier=ppn,test_idx=range(105,150))

#軸ラベルの設定
plt.xlabel('petal length[cm]')   
plt.ylabel('petal width[cm]')
plt.legend(loc = 'upper left')
plt.tight_layout()
plt.show()    
