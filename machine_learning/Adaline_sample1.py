# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 13:33:19 2020

@author: myy388
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Adaline
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

#1,3列目のみ抽出 1列目：がくの長さ、3列目：花びらの長さ
X = df.iloc[0:100,[0,2]].values

#subplot 1枚に複数のグラフ作成　subplots(行,列,グラフ番号)
fig, ax = plt.subplots(nrows = 1,ncols = 2,figsize=(10,4))

ada1 = Adaline.AdalineGD(n_iter = 10,eta = 0.01)
ada1.fit(X,y)
ax[0].plot(range(1,len(ada1.cost_)+1),np.log10(ada1.cost_),marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-errors)')
ax[0].set_title('Adaline learning rate 0.01')

ada2 = Adaline.AdalineGD(n_iter = 10,eta = 0.0001)
ada2.fit(X,y)
ax[1].plot(range(1,len(ada2.cost_)+1),np.log10(ada2.cost_),marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('log(Sum-squared-errors)')
ax[1].set_title('Adaline learning rate 0.0001')

plt.show()

#plot用
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

#データコピー
X_std = np.copy(X)
#各列の標準化
X_std[:,0] = (X[:,0] - X[:,0].mean())/X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean())/X[:,1].std()

#勾配降下法によるadalineの学習(標準化後、学習率eta = 0.01)
ada3 = Adaline.AdalineGD(n_iter = 15,eta = 0.01)
#モデルの適合
ada3.fit(X_std,y)


#境界領域のプロット
plot_decision_regions(X_std,y,Classifier = ada3)
plt.title('Adline Gradient Descent')
#軸のラベル設定
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
#凡例の設置
plt.legend(loc = 'upper left')
#図の表示 tight_layout()　複数グラフのときのサイズ調整
plt.tight_layout()
plt.show()

#エポック数とコストの関係を表すグラフ
plt.plot(range(1,len(ada3.cost_)+1),ada3.cost_,marker='o')
#軸のラベル設定
plt.xlabel('Epochs')
plt.set_ylabel('Sum-squared-errors')
plt.tight_layout()
plt.show()
