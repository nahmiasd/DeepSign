from keras.layers import Dense, Input, Dropout
from keras.models import Model
from keras.optimizers import SGD
from keras import regularizers
import numpy as np
import keras.callbacks
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint

tensorboard = keras.callbacks.TensorBoard(log_dir='./logs',
                 histogram_freq=0,
                 write_graph=True,
                 write_images=False)

class DeepSign():

    def __init__(self, input_vector_size=20000, dropout_rate=0.5, activation_function="relu", lr=0.001, epochs=5000,
                 batch_size=20, noise_factor=0.2):
        self.inputVectorSize = input_vector_size
        self.dropoutRate = dropout_rate
        self.activationFunc = activation_function
        self.autoencoders = []
        self.encoders = []
        self.predictModel = None
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.noise_factor = noise_factor
        self.mode='signature'


    def buildModels(self):
        """
        Generates all autoencoders as specified in the DeepSign paper.
        Then, adds up all the hidden layers from all autoencoders to a new encoding model.
        :return:
        """
        # First encoder
        self.stackAutoEncoder(input_size=self.inputVectorSize, hidden_size=5000)
        # Second encoder
        self.stackAutoEncoder(input_size=5000, hidden_size=2500)
        # Third encoder
        self.stackAutoEncoder(input_size=2500, hidden_size=1000)
        # 4th encoder
        self.stackAutoEncoder(input_size=1000, hidden_size=500)
        # 5th encoder
        self.stackAutoEncoder(input_size=500, hidden_size=250)
        # 6th encoder
        self.stackAutoEncoder(input_size=250, hidden_size=100)
        # 7th encoder
        self.stackAutoEncoder(input_size=100, hidden_size=30)

        # Predicting model, will be used to generate signatures
        input_predict = Input(shape=(self.inputVectorSize,))
        encoded_predict = self.encoders[0].layers[1](input_predict)
        for i in range(1, len(self.encoders)):
            encoded_predict = self.encoders[i].layers[1](encoded_predict)
        self.predictModel = Model(input_predict, encoded_predict)

    def stackAutoEncoder(self, input_size, hidden_size):
        '''
        Builds a single autoencoder and append it to the autoencoders list.
        :param input_size: Input layer size of the autoencoder
        :param hidden_size: Hidden layer size of the autoencoder
        :return:
        '''
        input_layer = Input(shape=(input_size,))
        encoded = Dense(hidden_size, activation=self.activationFunc,kernel_regularizer=regularizers.l2())(
            input_layer)
        dropout = Dropout(self.dropoutRate)(encoded)
        decoded = Dense(input_size)(dropout)
        autoencoder =Model(input_layer, decoded)
        autoencoder.compile(SGD(lr=self.lr, decay=0.999), loss='mse')
        self.autoencoders.append(autoencoder)
        self.encoders.append(Model(input_layer, encoded))

    def fit(self, X):
        '''
        Fits the model (signatures mode)
        :param X: Data
        :return:
        '''
        self.buildModels()

        # Add noise and clip
        X_to_fit = self.add_noise(X)


        # train each autoencoder iteratively
        for i in range(len(self.autoencoders)):
            autoencoder = self.autoencoders[i]
            encoder = self.encoders[i]
            history_callback=autoencoder.fit(X_to_fit, X, epochs=self.epochs, batch_size=self.batch_size,callbacks=[tensorboard])
            self.save_hist(history_callback,i)
            if i != len(self.autoencoders):
                X_to_fit = encoder.predict(X_to_fit)
                X=X_to_fit

        # save the predicting model
        self.predictModel.save('predict_model.h5')

    def add_noise(self, X):
        '''
        Add noise to the data.
        :param X: Data
        :return:
        '''
        X_to_fit = X + self.noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape)
        X_to_fit = np.clip(X_to_fit, 0., 1.)
        return X_to_fit

    def predict_signature(self, X):
        '''
        Generate signatures for data.
        :param X: Data
        :return: Encoded signature of each sample in X.
        '''
        return self.predictModel.predict(X)

    def add_softmax_layer(self):
        '''
        Adds a softmax layer on top of the encoding model. Used for prediction mode.
        :return:
        '''
        input_ann=Input(shape=(self.inputVectorSize,))
        ann=self.autoencoders[0].layers[1](input_ann)
        for i in range(1,len(self.autoencoders)):
            ann=self.autoencoders[i].layers[1](ann)
        clf_layer=Dense(4,activation='softmax')(ann)
        self.predictModel=Model(input_ann,clf_layer)
        self.predictModel.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.001),metrics=['accuracy'])
        self.mode='classify'

    def fit_ann_clf(self,X,y):
        '''
        Fits the encoding model in prediction mode.
        :param X: Signatures
        :param y: Ground truth
        :return:
        '''
        if self.mode=='signature':
            self.add_softmax_layer()
        x_to_fit=self.add_noise(X)
        self.predictModel.fit(x_to_fit,y,epochs=self.epochs,batch_size=self.batch_size)
        self.predictModel.save('predict_clf.h5')

    def predict_ann_clf(self,X):
        '''
        Generates predictions in prediction mode.
        :param X: Data to predict.
        :return: Classes of each sample x in X.
        '''
        return self.predictModel.predict(X)

    def save_hist(self, history_callback, i):
        '''
        Saves loss history
        :param history_callback:
        :param i: Number of autoencoder.
        :return:
        '''
        loss_history=np.array(history_callback.history['loss'])
        np.savetxt("history\\"+"loss_history_model"+str(i)+".txt", loss_history, delimiter=",")
