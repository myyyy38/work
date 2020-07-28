# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 15:40:55 2020

@author: myy388
"""

import matplotlib.pyplot as plt
import numpy as np

#シグモイド関数の定義
def sigmoid(z):
    return 1.0/(1 + np.exp(-z))

#0.1間隔で-7-7の数値を生成
z = np.arange(-7,7,0.1)

phi_z = sigmoid(z)
plt.plot(z,phi_z)

#垂直線の追加
plt.axvline(0.0,color='k')

plt.ylim(-0.1,1.1)
plt.xlabel('z')
plt.ylabel('$\phi (z)$')

#y軸目盛追加
plt.yticks([0.0,0.5,1.0])
#Axesクラスのオブジェクト取得
#gca get current axes 現在アクティブな軸の取得
ax = plt.gca()
#グリッド線の追加
ax.yaxis.grid(True)

plt.tight_layout()
plt.show()

#y = 1のコストを計算する関数
def cost_1(z):
    return - np.log(sigmoid(z))

#y = 0のコストを計算する関数
def cost_0(z):
    return - np.log(1 - sigmoid(z))

z = np.arange(-10,10,0.1)
phi_z = sigmoid(z)

#y = 1のコストを計算する関数
c1 = [cost_1(x) for x in z]
#プロット
plt.plot(phi_z,c1,label='J(w) if y = 0')

#y = 0のコストを計算する関数
c0 = [cost_0(x) for x in z]
#プロット
plt.plot(phi_z,c0,linestyle='--',label='J(w) if y = 1')

plt.ylim(0.0,5.1)
plt.xlim(0,1.0)
plt.xlabel('$\phi (z)$')
plt.ylabel('J(w)')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


