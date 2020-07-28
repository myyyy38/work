# -*- coding: utf-8 -*-
import numpy as np

class AdalineGD(object):
    """Adaptive Liner Neuron分類器
    eta: float型 学習率η
    n_iter: int型　トレーニングデータの試行回数
    random_state: int型　乱数のシード
    
    w_: 1次元配列　適合後の重み
    cost_:リスト型 各エポックの平方誤差和のコスト関数
    
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
        -------

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            """
            activationメソッドは、ただの恒等関数のため、これは意味がない。
            output = self.net_input(X)と直接指定することもできたが、そうしていないのは、activationメソッドが概念的なものであるため
            ロジスティック回帰の場合、分類機を実装するためにシグモイド関数に変更することもできる
            """
            output = self.activation(net_input)
            #誤差y^i - Φ(z)^i errorsは、すべてのサンプルのデータが入った配列
            #a.shape 行列の行数、列数取得、１次元配列の場合は(n,)で要素数
            errors = (y - output)
            #重みw_1 ~ w_mの更新
            #X.T.dot(errors) 要素数2の1次元配列
            self.w_[1:] += self.eta * X.T.dot(errors)
            #w_0の更新
            self.w_[0] += self.eta * errors.sum()
            #コスト関数の計算
            cost = (errors**2).sum()/2.0
            #コストの格納
            self.cost_.append(cost)
        return self
    
    def net_input(self,X):
        #総入力の計算
        #n個のサンプルに対して、総入力全て計算(サンプルの個数分の列の内積)
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self,X):
        #線形活性化関数の出力を計算
        return X
    
    def predict(self,X):
        #1ステップ後のクラスラベルを返す
        return np.where(self.net_input(X) >= 0.0,1,-1)


