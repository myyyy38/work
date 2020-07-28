# -*- coding: utf-8 -*-
import numpy as np

#確率的勾配降下法
class AdalineSGD(object):
    """Adaptive Liner Neuron分類器
    eta: float型 学習率η
    n_iter: int型　トレーニングデータの試行回数
    random_state: int型　乱数のシード
    
    w_: 1次元配列　適合後の重み
    cost_:リスト型 各エポックの平方誤差和のコスト関数
    
    """
    
    def __init__(self, eta = 0.01,n_iter = 10,shuffle = True,random_state = None):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        #重みの初期化フラグはFalseに設定
        self._initialized = False
        #各エポックでのデータのシャッフルをするか否か
        self.shuffle = shuffle
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
        #重みベクトルの生成
        self._initialize_weights(X.shape[1])
        #コストを格納するリスト
        self.cost_ = []
        
        #トレーニング回数分データを反復
        for i in range(self.n_iter):
            #指定された場合はデータをシャッフル
            if self.shuffle == True:
                X,y = self._shuffle(X,y)
            #各サンプルのコストを格納するリストの生成
            cost = []
            #各サンプルに対する計算 ここでXからサンプル抽出
            for xi,target in zip(X,y):
                #特徴量xiと目的関数yを用いた重みの更新とコストの計算
                cost.append(self._update_weights(xi,target))
                #サンプルの平均コストの計算
            avg_cost = sum(cost)/len(y)
            #平均コストの格納
            self.cost_.append(avg_cost)
        return self
        
    def partial_fit(self,X,y):
        """重みを初期化することなくトレーニングデータに適用させる"""
        #初期化されていない場合は初期化を実行
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        #目的変数yの要素数が２以上の場合は
        #各サンプルの特徴量xiと目的変数targetで重みを更新
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X,y):
                self._upadte_weights(xi,target)
        #目的変数の要素数が１の場合は、
        #サンプル全体の特徴量Xと目的変数yで重みを更新
        else:
            self._update_weights(X,y)
        return self
    
    def _shuffle(self,X,y):
        """"トレーニングデータのシャッフル"""
        #r 0-99までの乱数が入ったリスト
        r = self.rgen.permutation(len(y))
        #X[r] Xをrのインデックスの順にシャッフルしたリストを返す
        return X[r],y[r]
    
    def _initialize_weights(self,m):
        """重みを小さな乱数に初期化"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc = 0.0,scale = 0.01,size = 1 + m)
        self.w_initialized = True
        
    def _update_weights(self,xi,target):
        """Adalineの学習規則を用いて重みを更新 各サンプルごと"""
        #活性化関数の出力
        output = self.activation(self.net_input(xi))
        #誤差の計算
        error = (target - output)
        #重みw_1 ~ w_mの更新
        self.w_[1:] += self.eta * xi.dot(error)
        #重みw_0の更新
        self.w_[0] += self.eta * error
        #コストの計算
        cost = (error**2)/2.0
        return cost
        
    def net_input(self,X):
        #各サンプルに対しての総入力計算　Xは各サンプル(1行2列)
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self,X):
        #線形活性化関数の出力を計算
        return X
    
    def predict(self,X):
        #1ステップ後のクラスラベルを返す
        return np.where(self.net_input(X) >= 0.0,1,-1)


