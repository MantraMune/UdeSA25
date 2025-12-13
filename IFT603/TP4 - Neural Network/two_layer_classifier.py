import numpy as np


class TwoLayerClassifier(object):
    def __init__(
        self, x_train, y_train, x_val, y_val, num_features, num_hidden_neurons,
        num_classes, activation='relu'
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

        self.num_features = num_features
        self.num_classes = num_classes

        self.net = TwoLayerNet(
            num_features, num_hidden_neurons, num_classes, activation)

        self.momentum_cache_v_prev = {}

    def train(
        self, num_epochs=1, lr=1e-3, l2_reg=1e-4, lr_decay=1.0, momentum=0.0
    ):
        """
        Train the model with a cross-entropy loss
        Naive implementation (with loop)

        Inputs:
        - num_epochs: the number of training epochs
        - lr: learning rate
        - l2_reg: the l2 regularization strength
        - lr_decay: learning rate decay.  Typically a value between 0 and 1
        - momentum : variable mu for momentum update

        Returns a tuple for:
        - training accuracy for each epoch
        - training loss for each epoch
        - validation accuracy for each epoch
        - validation loss for each epoch
        """
        loss_train_curve = []
        loss_val_curve = []
        accu_train_curve = []
        accu_val_curve = []

        self.net.reinit()
        self.net.l2_reg = l2_reg

        self.momentum_cache_v_prev = {id(x): np.zeros_like(
            x) for x in self.net.parameters}  # variable v of momentum

        sample_idx = 0
        num_iter = num_epochs * len(self.x_train)
        for i in range(num_iter):
            # Take a sample
            x_sample = self.x_train[sample_idx]
            y_sample = self.y_train[sample_idx]

            # Forward + Backward
            loss_train = self.net.forward_backward(x_sample, y_sample)

            # Take gradient step
            for w, dw in zip(self.net.parameters, self.net.gradients):
                self.momentum_update(w, dw, lr, momentum)

            # Advance in data
            sample_idx += 1
            if sample_idx >= len(self.x_train):  # End of epoch

                accu_train, loss_train = \
                    self.global_accuracy_and_cross_entropy_loss(
                        self.x_train, self.y_train)
                accu_val, loss_val, = \
                    self.global_accuracy_and_cross_entropy_loss(
                        self.x_val, self.y_val)

                loss_train_curve.append(loss_train)
                loss_val_curve.append(loss_val)
                accu_train_curve.append(accu_train)
                accu_val_curve.append(accu_val)

                sample_idx = 0

                lr *= lr_decay

        return (
            loss_train_curve, loss_val_curve, accu_train_curve, accu_val_curve)

    def predict(self, x):
        """
        Returns the most likely class label associated to x
        Naive implementation (with loop).

        This function should not update weights.

        Inputs:
        - X: A numpy array of shape (N, D) containing one or many samples.
        Returns a tuple of:
        - A numpy array of shape (N) containing one or many class label
        """
        if len(x.shape) == 1:  # Predict on one sample

            scores = self.net.forward(x)  # Calcul des scores
            y_pred = np.argmax(scores)  # Classe prédite
            return y_pred
        
        elif len(x.shape) == 2:  # Predict on multiple samples

            # Créer un tableau pour stocker les prédictions        
            y_pred = np.zeros(x.shape[0], dtype=int)

            for i in range(x.shape[0]):
                scores = self.net.forward(x[i])  # Calcul des scores pour chaque échantillon
                y_pred[i] = np.argmax(scores)  # Classe prédite pour chaque échantillon

            return y_pred

    def global_accuracy_and_cross_entropy_loss(self, x, y, l2_r=-1.0):
        """
        Compute average accuracy and cross_entropy for a series of N data
        points. Naive implementation (with loop).

        This function should not update weights.

        Inputs:
        - x: A numpy array of shape (N, D) containing several samples.
        - y: A numpy array of shape (N) labels as an integer
        - reg: (float) L2 regularization strength
        Returns a tuple of:
        - average accuracy as single float
        - average loss as single float
        """
        if l2_r > 0:
            self.net.l2_reg = l2_r

        N = x.shape[0]
        loss = 0
        accu = 0

        for i in range(N):
            # Forward pass to get scores
            scores = self.net.forward(x[i])

            # SoftMax
            scores -= np.max(scores)
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores)

            # Accuracy
            y_pred = np.argmax(probs)
            if y_pred == y[i]:
                accu += 1
            
            # Perte par entropie croisée
            loss += -np.log(probs[y[i]])

        accu /= N # Pour recentrer l'accuracy entre 0 et 1
        loss /= N # Idem pour la loss

        # Régularisation L2
        if l2_r > 0:
            loss += 0.5 * l2_r * (
                np.sum(self.net.layer1.W**2) +
                np.sum(self.net.layer2.W**2)
            )
        
        return accu, loss

    def momentum_update(self, w, dw, lr, mu):
        """
        Compute momentum as in : http://cs231n.github.io/neural-networks-3/#sgd

        In our case, variable v is stored in self.momentum_cache_v_prev.

        The resulting w is put in parameter "w" which is passed by reference

        Returns nothing
        """

        v_prev = self.momentum_cache_v_prev[id(w)]
        v = mu * v_prev - lr * dw # Calcul de la vitesse
        w += v  # Mise à jour des poids
        self.momentum_cache_v_prev[id(w)] = v


class TwoLayerNet(object):
    """
    This class encodes a network with two layers or parameters : one between
    the input layer and the hidden layer and one between the hidden layer and
    the output layer.
    """

    def __init__(
        self, in_size, hidden_size, num_classes, activation='relu', l2_r=0.0
    ):
        self.in_size = in_size
        self.num_classes = num_classes
        self.l2_reg = l2_r
        self.layer1 = DenseLayer(in_size, hidden_size, activation=activation)
        self.layer2 = DenseLayer(hidden_size, num_classes)

    def reinit(self):
        self.layer1.reinit()
        self.layer2.reinit()

    def forward(self, x):
        x1 = self.layer1.forward(x)
        x2 = self.layer2.forward(x1)
        return x2

    def backward_(self, dloss_dscores):
        dx = self.layer2.backward(dloss_dscores, self.l2_reg)
        self.layer1.backward(dx, self.l2_reg)

    def forward_backward(self, x, y):
        self.layer1.zero_grad()
        self.layer2.zero_grad()

        scores = self.forward(x)
        loss, dscores = self.cross_entropy_loss(scores, y)
        self.backward_(dscores)
        return loss

    @property
    def parameters(self):
        return [self.layer1.W, self.layer2.W]

    @property
    def gradients(self):
        return [self.layer1.dW, self.layer2.dW]

    def cross_entropy_loss(self, scores, y):
        """
        Cross-entropy loss function and gradient with respect to the score
        C.f. Eq.(4.104 to 4.109) of Bishop book.

        Inputs:
        - scores: output of the network before the softmax.  A numpy array of
        shape (C) (C is the number of classes, self.num_classes).
        - y: training label as an integer
        Returns a tuple of:
        - loss as single float
        - gradient with respect to score; an array of same shape of scores
        """

        loss = 0
        dloss_dscores = np.zeros(np.size(scores))

        # Calcul des scores
        scores -= np.max(scores)  # Pour la stabilité numérique

        # Softmax
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores)

        # Perte par entropie croisée
        loss = -np.log(probs[y])

        # Gradient de la perte par rapport aux scores
        dloss_dscores = probs.copy()
        dloss_dscores[y] -= 1.0  # (p_k - 1_{k=y})

        return loss, dloss_dscores


class DenseLayer(object):
    """
    This class encodes a layer (could be the hidden layer or the output layer)
    """

    def __init__(self, in_size, out_size, activation=None):
        self.activation = activation  # Note, 'relu', or 'sigmoid'
        self.W = None
        self.dW = None
        self.in_size = in_size  # number of input neurons
        self.out_size = out_size  # number of output neurons
        self.reinit()

        self.last_x = None
        self.last_activ = None

    def reinit(self):
        self.W = np.random.randn(
            self.in_size + 1, self.out_size) * np.sqrt(
                0.1 / (self.in_size + self.out_size))
        self.dW = np.zeros_like(self.W)

    def zero_grad(self):
        self.dW.fill(0.0)

    def forward(self, x):
        """
        Computer forward pass for 1 layer :

                f=h(dot(W,x))

        where h is the layer's activation function.

        Inputs:
        - x: A numpy array of shape (D)
        Returns a tuple of:
        - f: a floating point value
        """
        # Augmenter x pour le biais
        x_aug = augment(x)

        # Produit linéaire
        z = x_aug @ self.W

        # Activation
        if self.activation is None:
            f = z
        elif self.activation == 'sigmoid':
            f = 1 / (1.0 + np.exp(-z))
        elif self.activation == 'relu':
            f = np.maximum(0.0, z)
        else:
            raise ValueError('Unknown activation "%s"' % self.activation)
        
        self.last_x = x_aug
        self.last_activ = f

        return f

    def backward(self, dnext_dout, l2_reg):
        if self.activation == 'sigmoid':
            dout_dsigmoid = self.last_activ * (1.0 - self.last_activ)
            dnext_dsigmoid = dnext_dout * dout_dsigmoid  # type:np.ndarray
            dnext_dW = self.last_x[:, np.newaxis] * \
                dnext_dsigmoid[np.newaxis, :]
            dnext_dX = dnext_dsigmoid.dot(self.W.T)
        elif self.activation == 'relu':
            dout_drelu = self.last_activ != 0.0
            dnext_drelu = dnext_dout * dout_drelu  # type:np.ndarray
            dnext_dW = self.last_x[:, np.newaxis] * dnext_drelu[np.newaxis, :]
            dnext_dX = dnext_drelu.dot(self.W.T)
        else:
            dnext_dW = self.last_x[:, np.newaxis].dot(
                dnext_dout[np.newaxis, :])
            dnext_dX = dnext_dout.dot(self.W.T)
        # discard the gradient wrt the 1.0 of homogeneous coord
        dnext_dX = dnext_dX[:-1]

        self.dW += dnext_dW
        self.dW += l2_reg * self.W  # add regul. gradient
        return dnext_dX


def augment(x):
    if len(x.shape) == 1:
        return np.concatenate([x, [1.0]])
    else:
        return np.concatenate([x, np.ones((len(x), 1))], axis=1)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
