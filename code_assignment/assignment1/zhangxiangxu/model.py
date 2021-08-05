import numpy as np

class Model:
    def __init__(self, train_data, test_data):
        """
        创建实例，需要训练和测试数据
        :param train_data: 包含X(d*n),Y(c*n)
        :param test_data: 包含test_X(d*n_test),test_Y(c*n_test)
        """
        self.X = train_data[0]
        self.Y = train_data[1]

        self.test_X = test_data[0]
        self.test_Y = test_data[1]

        self.parameter = {} #  盛放W，b的字典
        self.cache = {} #  盛放Z，Y_batch_hat的字典
        self.gradient = {} #  盛放dW, db的字典

    def dict_empty(self):
        """
        清空所有存储的参数、缓存、梯度
        :return: None
        """
        self.parameter = {} #  盛放W，b的字典
        self.cache = {} #  盛放Z，Y_batch_hat的字典
        self.gradient = {} #  盛放dW, db的字典

    def parameter_normalize(self, seed=1):
        """
        初始化权重W（d * c），b（c * 1)
        参数：
        self:实例

        返回：
        W（d * c），b（c * 1)
        """
        c = self.Y.shape[0]
        d = self.X.shape[0]
        np.random.seed(seed)
        W = np.random.randn(d, c)
        b = np.zeros((c, 1))

        self.parameter['W'] = W
        self.parameter['b'] = b

        return W, b

    def create_mini_batches(self, batch_size, seed=1):
        """
        参数：
        self:实例
        batch_size：大小，标量
        seed：随机种子，标量

        返回：
        mini_batches：包含mini_batch的列表，内部是（mini_batch_X, mini_batch_Y）
        """
        np.random.seed(seed)
        n = self.X.shape[1]
        mini_batches = []
        num_of_batches = int(n / batch_size)
        for k in range(0, num_of_batches):
            mini_batch_X = self.X[:, k * batch_size: (k + 1) * batch_size]
            mini_batch_Y = self.Y[:, k * batch_size: (k + 1) * batch_size]

            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        if n % num_of_batches != 0:
            mini_batch_X = self.X[:, num_of_batches * batch_size:]
            mini_batch_Y = self.Y[:, num_of_batches * batch_size:]

            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches

    def forward(self, X_batch):
        """
        执行前向传播，Z = W.T*X_batch + b
        :param X_batch: 输入，array，dimension * batch_size

        :return: cache：包含Z，Y_batch_hat的字典
        """

        W = self.parameter['W']
        b = self.parameter['b']

        Z = np.dot(W.T, X_batch) + b
        Y_batch_hat = softmax(Z)

        self.cache['Z'] = Z
        self.cache['Y_batch_hat'] = Y_batch_hat
        assert Z.shape == (W.shape[1], X_batch.shape[1])
        return self.cache

    def compute_loss(self, Y_batch, Y_batch_hat, lamda):
        """
        计算损失
        :param Y_batch_hat: array，预测值，c * batch_size
        :param Y_batch: array，标签，one_hot矩阵，c * batch_size
        :param lamda: 正则化系数
        :return: L，标量
        """
        W = self.parameter['W']

        n = Y_batch.shape[1]
        l = - np.sum(Y_batch * np.log(Y_batch_hat), axis=0)
        loss = np.sum(l) / n + 1 / 2 / n * lamda * np.square(np.linalg.norm(W))

        return loss

    # 计算梯度
    def backward(self, X_batch, Y_batch):
        """
        计算梯度
        X_batch: array, float, d * n
        Y_batch: array, float, c * n

        return：一个包含梯度dW,db的字典，维度与W，b相同
        """
        Y_batch_hat = self.cache['Y_batch_hat']
        n = X_batch.shape[1]

        # 计算梯度
        dW = - np.dot(X_batch, (Y_batch - Y_batch_hat).T) / n
        db = - np.sum((Y_batch - Y_batch_hat), axis=1) / n

        self.gradient['dW'] = dW
        self.gradient['db'] = db

        return self.gradient

    def update_gradient(self, learning_rate, lamda, belta1, belta2, epsilon, t, n, opt='SGD'):
        """
        反向传播
        参数：
        self.parameter：包含参数W，b, m_W, v_W, m_b, v_b的字典
            W: 权重矩阵，float，d * c
            b：偏置，float，c * 1
        self.gradient：包含梯度的字典
        learning_rate：学习率
        lamda：正则化系数
        #t: 当前进行下降的次数

        返回：
        self.parameter：包含参数W，b的字典
        """
        dW = self.gradient['dW']
        db = self.gradient['db']
        W = self.parameter['W']
        b = self.parameter['b']
        m_W = self.parameter['m_W']
        v_W = self.parameter['v_W']
        m_b = self.parameter['m_b']
        v_b = self.parameter['v_b']
        c = W.shape[1]
        d = W.shape[0]
        n = W.shape[1]
        dW = dW.reshape((d, c))
        db = db.reshape((c, 1))

        if opt == 'Adam':
            # 更新
            m_W = belta1 * m_W + (1 - belta1) * dW
            v_W = belta2 * v_W + (1 - belta2) * np.square(dW)
            m_W_hat = m_W / (1 - np.power(belta1, t))
            v_W_hat = v_W / (1 - np.power(belta2, t))
            W = W - np.multiply(learning_rate / (np.sqrt(v_W_hat) + epsilon), m_W_hat)

            m_b = belta1 * m_b + (1 - belta1) * db
            v_b = belta2 * v_b + (1 - belta2) * np.square(db)
            m_b_hat = m_b / (1 - np.power(belta1, t))
            v_b_hat = v_b / (1 - np.power(belta2, t))
            b = b - np.multiply(learning_rate / (np.sqrt(v_b_hat) + epsilon), m_b_hat)
        elif opt == 'SGD':
            W = W - learning_rate * (dW + 1 / n * lamda * W)
            b = b - learning_rate * (db + 1 / n * lamda * b)
        assert W.shape == dW.shape
        #     print(b.shape, db.shape)
        assert b.shape == db.shape

        self.parameter['W'] = W
        self.parameter['b'] = b
        self.parameter['m_W'] = m_W
        self.parameter['v_W'] = v_W
        self.parameter['m_b'] = m_b
        self.parameter['v_b'] = v_b

        return self.parameter

    def train(self, num_epochs=100, batch_size=64, learning_rate=0.001, belta1=0.9, belta2=0.999,
              epsilon=1e-8, lamda=0.1, opt='SGD', if_print=True):
        """
        进行训练
        参数：
        X：训练数据
        Y：标签
        epoch：训练次数
        batch_size：batch大小
        learning_rate：学习率
        lamda：正则化系数
        """
        # 字典归零
        self.dict_empty()

        losses = []
        test_losses = []
        accuray_trains = []
        accuray_tests = []
        loss = 0
        # 初始化参数
        W, b = self.parameter_normalize()
        m_W = 0
        v_W = 0
        m_b = 0
        v_b = 0

        self.parameter['W'] = W
        self.parameter['b'] = b
        self.parameter['m_W'] = m_W
        self.parameter['v_W'] = v_W
        self.parameter['m_b'] = m_b
        self.parameter['v_b'] = v_b

        # 创建mini_batch
        mini_batches = self.create_mini_batches(batch_size, seed=1)
        t = 0
        for i in range(num_epochs):
            seed = 0
            np.random.seed(seed)
            permutation = np.random.permutation(len(mini_batches))
            mini_batches = np.array(mini_batches, dtype=object)[permutation]

            for mini_batch in mini_batches:
                t += 1
                mini_batch_X, mini_batch_Y = mini_batch
                self.cache = self.forward(mini_batch_X)  # 包含Z，Y_batch_hat
                Y_batch_hat = self.cache['Y_batch_hat']
                loss = self.compute_loss(mini_batch_Y, Y_batch_hat, lamda=lamda)
                self.gradient = self.backward(mini_batch_X, mini_batch_Y,)  # 包含dW，db
                self.parameter = self.update_gradient(
                    learning_rate=learning_rate,
                    belta1=belta1,
                    belta2=belta2,
                    epsilon=epsilon,
                    lamda=lamda,
                    n=mini_batch_X.shape[1],
                    t=t,
                    opt=opt
                )

            seed += 1

            if i % 10 == 0:
                losses.append(loss)
                Y_hat = self.forward(self.X)['Y_hat']
                preds = np.equal(Y_hat, np.max(Y_hat, axis=0))
                true = self.Y
                correct_prediction_train = np.equal(preds, true)
                accuray_train = np.mean(correct_prediction_train) * 100
                accuray_trains.append(accuray_train)

                test_Y_hat = self.forward(test_X)['Y_batch_hat']
                # 此处是迁就forward函数用“batch”，实际上是全集，可以理解成一个大的batch
                preds = np.equal(test_Y_hat, np.max(test_Y_hat, axis=0))
                test_true = self.test_Y
                correct_prediction_test = np.equal(preds, test_true)
                accuray_test = np.mean(correct_prediction_test) * 100
                accuray_tests.append(accuray_test)
                test_loss = self.compute_loss(test_true, test_Y_hat, lamda=lamda)
                test_losses.append(test_loss)
                if if_print:
                    print(f"已运行{i}次，当前损失为：" + str(loss), end=', ')
                    print(f"当前训练集准确率为：{accuray_train:.2f}%")
                    print(f"当前测试集损失为：{test_loss}", end=', ')
                    print(f"当前测试集准确率为：{accuray_test:.2f}%")

        return self.parameter, losses, accuray_trains, accuray_tests