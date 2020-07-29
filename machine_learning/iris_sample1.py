# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import perceptron
from matplotlib.colors import ListedColormap


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
print(df.tail())

#1-100行目の目的変数の抽出,iloc= 行列指定
y = df.iloc[0:100,4].values
#print(y)

#iris-setosaを-1,versicolorを1に変換
#mp.where(if節,T,F)
y = np.where(y == 'Iris-setosa',-1,1)
#print(y)

#1,3列目のみ抽出 1列目：がくの長さ、3列目：花びらの長さ 100行2列の行列
X = df.iloc[0:100,[0,2]].values

#setosaプロット　赤○ x軸がくの長さ、y軸花びらの長さ マーカーは、オーとエックス
plt.scatter(X[:50,0],X[:50,1], color = 'red',marker = 'o',label = 'setosa')

#verscolorプロット　青×　
plt.scatter(X[50:100,0],X[50:100,1], color = 'blue',marker = 'x',label = 'verscolor')

plt.xlabel('sepal length[cm]')
plt.ylabel('pepal length[cm]')

#凡例の表示
plt.legend(loc = 'upper left')

plt.show()


#perceptron.pyのperceptronクラス呼び出し
ppn = perceptron.perceptron(eta =  0.1, n_iter = 10)
ppn.fit(X,y)
plt.plot(range(1,(len(ppn.errors_) + 1)),ppn.errors_,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of update')
plt.show()

def plot_decision_regions(X, y,Classifier,resolution = 0.02):
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
        plt.scatter(x = X[y == cl,0],
                    y = X[y == cl,1],
                    alpha = 0.8,
                    c = colors[idx],
                    marker = markers[idx],
                    label = cl,
                    edgecolor='black')

plot_decision_regions(X,y,Classifier=ppn)
#軸ラベルの設定
plt.xlabel('sepal length[cm]')   
plt.ylabel('petal length[cm]')
plt.legend(loc = 'upper left')
plt.show()    