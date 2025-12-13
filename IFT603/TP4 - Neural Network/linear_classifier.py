import numpy as np


class LinearClassifier(object):
    def __init__(
        self, x_train, y_train, x_val, y_val, num_classes, bias=False
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        # when bias is True then the feature vectors have an additional 1
        self.bias = bias

        num_features = x_train.shape[1]
        if bias:
            num_features += 1

        self.num_features = num_features
        self.num_classes = num_classes
        self.W = self.generate_init_weights(0.01)

    def generate_init_weights(self, init_scale):
        return np.random.randn(
            self.num_features, self.num_classes) * init_scale

    def train(
        self, num_epochs=1, lr=1e-3, l2_reg=1e-4, lr_decay=1.0, init_scale=0.01
    ):
        """
        Train the model with a cross-entropy loss
        Naive implementation (with loop)

        Inputs:
        - num_epochs: the number of training epochs
        - lr: learning rate
        - l2_reg: the l2 regularization strength
        - lr_decay: learning rate decay.  Typically a value between 0 and 1
        - init_scale : scale at which the parameters self.W will be randomly
                       initialized

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

        self.W = self.generate_init_weights(init_scale)  # type: np.ndarray

        sample_idx = 0
        num_iter = num_epochs * len(self.x_train)
        for i in range(num_iter):
            # Take a sample
            x_sample = self.x_train[sample_idx]
            y_sample = self.y_train[sample_idx]
            if self.bias:
                x_sample = augment(x_sample)

            # Compute loss and gradient of loss
            loss_train, dW = self.cross_entropy_loss(
                x_sample, y_sample, l2_reg)

            # Take gradient step
            self.W -= lr * dW

            # Advance in data
            sample_idx += 1
            if sample_idx >= len(self.x_train):  # End of epoch

                accu_train, loss_train = \
                    self.global_accuracy_and_cross_entropy_loss(
                        self.x_train, self.y_train, l2_reg)
                accu_val, loss_val, = \
                    self.global_accuracy_and_cross_entropy_loss(
                        self.x_val, self.y_val, l2_reg)

                loss_train_curve.append(loss_train)
                loss_val_curve.append(loss_val)
                accu_train_curve.append(accu_train)
                accu_val_curve.append(accu_val)

                sample_idx = 0
                lr *= lr_decay

        return (
            loss_train_curve, loss_val_curve, accu_train_curve, accu_val_curve)

    def predict(self, X):
        """
        return the class label with the highest class score i.e.

            argmax_c W.X

         X: A numpy array of shape (D,) containing one or many samples.

         Returns a class label for each sample (a number between 0 and
         self.num_classes-1)
        """
        # Considérer le biais avant de faire la prédiction
        if self.bias:
            X = augment(X)
        
        # Définir un tableau pour stocker les classes
        class_label = np.zeros(X.shape[0], dtype=int)
        
        # Calculer le score de chaque classe
        scores = X @ self.W  

        class_label= np.argmax(scores, axis=1)

        return class_label

    def global_accuracy_and_cross_entropy_loss(self, X, y, reg=0.0):
        """
        Compute average accuracy and cross_entropy for a series of N data
        points. Naive implementation (with loops).

        Accuracy is simply prediction == label.

        Inputs:
        - X: A numpy array of shape (N, D) containing many samples.
        - y: A numpy array of shape (N) labels as an integer
        - reg: (float) regularization strength

        Returns a tuple of:
        - average accuracy as single float
        - average loss as single float
        """
        N = X.shape[0]
        accu = 0
        loss = 0

        # Considérer que le biais est absorbé dans la matrice des poids
        if self.bias:
            X = augment(X)

        for i in range(N):
            # Scores pour chaque classe
            scores = X[i] @ self.W
            scores -= np.max(scores)  # Pour la stabilité numérique

            # Softmax
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores)

            # Prédiction
            y_pred = np.argmax(probs)

            # Accuracy
            if y_pred == y[i]:
                accu += 1

            # Cross-entropy loss
            loss += -np.log(probs[y[i]])
        
        accu /= N # Pour recentrer l'accuracy entre 0 et 1
        loss /= N # Idem pour la loss
        
        return accu, loss

    def cross_entropy_loss(self, x, y, reg=0.0):
        """
        Cross-entropy loss function for one sample pair (X,y) (with softmax)
        C.f. Eq.(4.104 to 4.109) of Bishop book.

        Inputs have dimension D, there are C classes.
        Inputs:
        - W: A numpy array of shape (D, C) containing weights.
        - x: A numpy array of shape (D,) containing one sample.
        - y: training label as an integer
        - reg: (float) regularization strength

        Returns a tuple of:
        - loss as single float
        - gradient with respect to weights W; an array of same shape as W

        Don't forget the bias ! (self.bias)
        """
        # Initialize the loss and gradient to zero.
        loss = 0.0
        dW = np.zeros_like(self.W)

        # Scores pour chaque classe
        scores = x @ self.W
        scores -= np.max(scores)  # Pour la stabilité numérique 

        # Softmax
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores)

        # Perte par entropie croisée
        loss = -np.log(probs[y])
        loss += 0.5 * reg * np.sum(self.W * self.W)  # Regularization

        # Gradient
        ds = probs.copy()
        ds[y] -= 1  # (p_k - 1_{k=y})  
        dW = np.outer(x, ds) + reg * self.W  #∂L / ∂W + régularisation

        return loss, dW


def augment(x):
    if len(x.shape) == 1:
        return np.concatenate([x, [1.0]])
    else:
        return np.concatenate([x, np.ones((len(x), 1))], axis=1)
