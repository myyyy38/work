# -*- coding: utf-8 -*-
import numpy as np



class perceptron(object):
    """
    eta: float型 学習率η
    n_iter: int型　トレーニングデータの試行回数
    random_state: int型　乱数のシード
    
    w_: 1次元配列　適合後の重み
    erroes_ 各エポックの誤分類の数
    
    """
    
    def __init__(self, eta = 0.01,n_iter = 50,random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self,X,y):
        """
        Parameters
        ----------
        X : 配列っぽいデータ, shape = [n_samples, n_features]
            n_samples: サンプルの個数, n_features: 特徴量の個数
            
        y : 配列っぽいデータ, shape = [n_samples]
            目的変数
        Returns
        -------

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                #重みの更新　η* (真のクラスラベルy - 予測値y^)* x
                update = self.eta*(target - self.predict(xi))
                self.w_[1:] += update *xi
                #重みw_0(バイアス)の更新
                self.w_[0] += update
                #重みの更新(update)が0でない場合、誤分類としてカウント
                errors += int(update != 0)
            #エポックごとの誤差の格納
            self.errors_.append(errors)
        return self
    
    def net_input(self,X):
        #総入力の計算
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self,X):
        #1ステップ後のクラスラベルを返す
        return np.where(self.net_input(X) >= 0.0,1,-1)
    