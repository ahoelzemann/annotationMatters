import os
import pickle
import wandb
import misc.Globals as project_globals

import numpy as np
import sklearn as sk
import tensorflow
from tensorflow.keras import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.regularizers import *
import deep_transferlearning_visualization.WandBViz as WandB_Logger
import misc.Globals as gl
# Clear allocated memory on GPU device

from deep_transferlearning_models.DeepConvLSTMParams import DeepConvLSTMParams



# author: Alexander Hoelzemann - alexander.hoelzemann@uni-siegen.de
# author: Lukas Wegmeth - lukas.wegmeth@uni-siegen.de


###############################################################
##            Class for customized callbacks                 ##
##          Can be called each epoch, batch etc.             ##
###############################################################

class CustomCallbacks(tensorflow.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        splits = self.model.name.split("_")
        fold = splits[2] + "_" + splits[3] + "_" + splits[4]
        WandB_Logger.save_best_model(logs, epoch, fold, self.model)
        epoch += 1
        WandB_Logger.log_metrics(logs, epoch, fold)

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass


class CustomMetrics:
    # copied from unnir's post https://github.com/keras-team/keras/issues/5400

    @staticmethod
    def recall_m(y_true, y_prediction):
        true_positives = tensorflow.keras.backend.sum(
            tensorflow.keras.backend.round(tensorflow.keras.backend.clip(y_true * y_prediction, 0, 1)))
        possible_positives = tensorflow.keras.backend.sum(
            tensorflow.keras.backend.round(tensorflow.keras.backend.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + tensorflow.keras.backend.epsilon())
        return recall

    @staticmethod
    def precision_m(y_true, y_prediction):
        true_positives = tensorflow.keras.backend.sum(
            tensorflow.keras.backend.round(tensorflow.keras.backend.clip(y_true * y_prediction, 0, 1)))
        predicted_positives = tensorflow.keras.backend.sum(
            tensorflow.keras.backend.round(tensorflow.keras.backend.clip(y_prediction, 0, 1)))
        precision = true_positives / (predicted_positives + tensorflow.keras.backend.epsilon())
        return precision

    @staticmethod
    def f1_m(y_true, y_prediction):
        y_prediction = _postprocess(y_prediction, 50, 5)
        precision = CustomMetrics.precision_m(y_true, y_prediction)
        recall = CustomMetrics.recall_m(y_true, y_prediction)
        result = 2 * ((precision * recall) / (precision + recall + tensorflow.keras.backend.epsilon()))

        return result


class DeepConvLSTM:

    def __init__(self, x_train, y_train, x_test, y_test, network_params: DeepConvLSTMParams, class_weights=None, ):
        """
        :param x_train: the training data
        :param y_train: the training labels
        :param x_test: the test data
        :param y_test: the test labels
        :param network_params: a DeepConvLSTMParams object containing all setup parameters for the network.
        """

        self.network_params = network_params
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_val = x_test
        self.y_val = y_test
        self.batch_depth = self.x_train.shape[0]
        self.batch_length = self.x_train.shape[1]
        self.n_channels = self.x_train.shape[2]
        self.n_classes = self.y_train.shape[1]
        self.val_split = network_params.val_split
        self.model_name = network_params.model_name

        self.num_cnn_layers = network_params.num_cnn_layers
        self.c_filters = network_params.c_filters
        self.c_kernel_size = network_params.c_kernel_size
        self.c_padding = network_params.c_padding
        self.c_activation = network_params.c_activation
        self.c_kernel_init = network_params.c_kernel_init
        self.c_kernel_regular = network_params.c_kernel_regular
        self.c_kernel_regular_rate = network_params.c_kernel_regular_rate
        self.c_bias_init = network_params.c_bias_init
        self.c_bias_regular = network_params.c_bias_regular
        self.c_bias_regular_rate = network_params.c_bias_regular_rate
        self.pool_type = network_params.pool_type
        self.pool_kernel_size = network_params.pool_kernel_size
        self.num_rnn_layers = network_params.num_rnn_layers
        self.lstm_units = network_params.lstm_units
        self.dropout_rate = network_params.dropout_rate
        self.optimizer = network_params.optimizer
        self.learning_rate = network_params.learning_rate
        self.verbose = network_params.verbose
        self.epochs = network_params.epochs
        self.batch_size = network_params.batch_size
        self.model_folder_base = network_params.model_folder_base
        self.checkpoint_frequency = network_params.checkpoint_frequency
        self.early_stopping_patience = network_params.early_stopping_patience
        self.fold = network_params.fold
        self.balanced_dataset = network_params.balanced_dataset

        self.wandb_config = {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size
        }
        self.neural_network = None
        self.model_evaluation_results = None
        self.prediction_evaluation_results = None
        self.purpose = network_params.purpose
        if class_weights is None:
            self.class_weights = np.full(fill_value=1.0, shape=(self.n_classes,))
        else:
            self.class_weights = class_weights

    def init_network(self):

        inputs = Input(shape=(self.batch_length, self.n_channels))
        print(f'Shape of Input: {inputs.shape}')
        x = Reshape(name='reshape_to_3d', target_shape=(self.batch_length, self.n_channels, 1))(inputs)
        print(f'Shape of Reshape3d: {x.shape}')
        for cnn_layer in range(self.num_cnn_layers):
            kernel_regular = None
            if self.c_kernel_regular[cnn_layer] == 'l1':
                kernel_regular = l1(l1=self.c_kernel_regular_rate[cnn_layer][0])
            elif self.c_kernel_regular[cnn_layer] == 'l2':
                kernel_regular = l2(l2=self.c_kernel_regular_rate[cnn_layer][0])
            elif self.c_kernel_regular[cnn_layer] == 'l1_l2':
                kernel_regular = l1_l2(l1=self.c_kernel_regular_rate[cnn_layer][0],
                                       l2=self.c_kernel_regular_rate[cnn_layer][1])
            bias_regular = None
            if self.c_bias_regular[cnn_layer] == 'l1':
                bias_regular = l1(l1=self.c_bias_regular_rate[cnn_layer][0])
            elif self.c_bias_regular[cnn_layer] == 'l2':
                bias_regular = l2(l2=self.c_bias_regular_rate[cnn_layer][0])
            elif self.c_bias_regular[cnn_layer] == 'l1_l2':
                bias_regular = l1_l2(l1=self.c_bias_regular_rate[cnn_layer][0],
                                     l2=self.c_bias_regular_rate[cnn_layer][1])
            x = Convolution2D(name=f'conv_{cnn_layer}', filters=self.c_filters[cnn_layer],
                              kernel_size=self.c_kernel_size[cnn_layer], padding=self.c_padding[cnn_layer],
                              activation=self.c_activation[cnn_layer], kernel_initializer=self.c_kernel_init[cnn_layer],
                              kernel_regularizer=kernel_regular, bias_initializer=self.c_bias_init[cnn_layer],
                              bias_regularizer=bias_regular)(x)
            print(f'Shape of Conv_{cnn_layer}: {x.shape}')
            x = Activation('relu')(x)
            x = BatchNormalization()(x)
        kernel_regular = None
        if self.c_kernel_regular[-1] == 'l1':
            kernel_regular = l1(l1=self.c_kernel_regular_rate[-1][0])
        elif self.c_kernel_regular[-1] == 'l2':
            kernel_regular = l2(l2=self.c_kernel_regular_rate[-1][0])
        elif self.c_kernel_regular[-1] == 'l1_l2':
            kernel_regular = l1_l2(l1=self.c_kernel_regular_rate[-1][0],
                                   l2=self.c_kernel_regular_rate[-1][1])
        bias_regular = None
        if self.c_bias_regular[-1] == 'l1':
            bias_regular = l1(l1=self.c_bias_regular_rate[-1][0])
        elif self.c_bias_regular[-1] == 'l2':
            bias_regular = l2(l2=self.c_bias_regular_rate[-1][0])
        elif self.c_bias_regular[-1] == 'l1_l2':
            bias_regular = l1_l2(l1=self.c_bias_regular_rate[-1][0],
                                 l2=self.c_bias_regular_rate[-1][1])
        x = Convolution2D(name=f'conv_reduce_channel', filters=1, kernel_size=self.c_kernel_size[-1],
                          padding=self.c_padding[-1], activation=self.c_activation[-1],
                          kernel_initializer=self.c_kernel_init[-1], kernel_regularizer=kernel_regular,
                          bias_initializer=self.c_bias_init[-1], bias_regularizer=bias_regular)(x)
        print(f'Shape of Conv_reduce_channel: {x.shape}')
        if self.pool_type == 'max':
            x = MaxPooling2D(name=f'max_pool', pool_size=self.pool_kernel_size)(x)
        elif self.pool_type == 'avg':
            x = AveragePooling2D(name=f'avg_pool', pool_size=self.pool_kernel_size)(x)
        elif self.pool_type == 'global_max':
            x = GlobalMaxPooling2D(name=f'global_max_pool', pool_size=self.pool_kernel_size)(x)
        elif self.pool_type == 'global_avg':
            x = GlobalAveragePooling2D(name=f'global_avg_pool', pool_size=self.pool_kernel_size)(x)
        print(f'Shape of pool: {x.shape}')
        x = Reshape(name="reshape_to_2d", target_shape=(x.shape[1], self.n_channels))(x)
        print(f'Shape of Reshape2d: {x.shape}')
        for rnn_layer in range(self.num_rnn_layers):
            ret_seq = True if rnn_layer < self.num_rnn_layers - 1 else False
            x = LSTM(name=f'lstm_{rnn_layer}', units=self.lstm_units[rnn_layer], return_sequences=ret_seq)(x)
            print(f'Shape of LSTM_{rnn_layer}: {x.shape}')
        x = Dropout(self.dropout_rate)(x)
        print(f'Shape of Dropout: {x.shape}')
        outputs = Dense(self.n_classes, activation='softmax')(x)
        print(f'Shape of Output: {outputs.shape}')
        self.neural_network = Model(inputs, outputs, name=self.model_name)
        optimizer = None
        if self.optimizer == 'sgd':
            optimizer = SGD(learning_rate=self.learning_rate)
        elif self.optimizer == 'adam':
            optimizer = Adam(learning_rate=self.learning_rate)
        elif self.optimizer == 'adagrad':
            optimizer = Adagrad(learning_rate=self.learning_rate)
        self.neural_network.compile(loss='categorical_crossentropy', optimizer=optimizer,
                                    metrics=['accuracy', CustomMetrics.f1_m, CustomMetrics.precision_m,
                                             CustomMetrics.recall_m])

        print(f'New network initialized')
        return

    def train_network(self):
        import numpy as np

        if not os.path.exists(f'{self.model_folder_base}/'):
            os.makedirs(f'{self.model_folder_base}/')

        checkpoint_base_folder = f'{self.model_folder_base}/checkpoints/'
        checkpoint_format = 'cp-{epoch:02d}.ckpt'
        steps_per_epoch = round(self.x_train.shape[0] / self.batch_size)
        cp_callback = ModelCheckpoint(filepath=f'{checkpoint_base_folder}{checkpoint_format}', save_weights_only=True,
                                      verbose=1, save_freq=steps_per_epoch * self.checkpoint_frequency)
        if self.early_stopping_patience != 0:
            es_callback = EarlyStopping(patience=self.early_stopping_patience, restore_best_weights=True)
        if project_globals.WANDB_LOGGING:
            wandb.init(project="projectname", entity="user-name")
        else:
            wandb.init(mode="disabled")
        if self.purpose not in wandb.run.name:
            if self.balanced_dataset:
                wandb.run.name =  self.purpose + "_" +  wandb.run.name.split("-")[-1] + "_balanced"
            else:
                wandb.run.name = self.purpose + "_" + wandb.run.name.split("-")[-1]
        wandb.config = self.wandb_config
        self.init_network()
        if self.early_stopping_patience != 0:
            history = self.neural_network.fit(self.x_train, self.y_train, epochs=self.epochs,
                                              batch_size=self.batch_size,
                                              verbose=self.verbose,
                                              validation_data=(self.x_val, self.y_val),
                                              class_weight=self.class_weights,
                                              callbacks=[cp_callback, es_callback, CustomCallbacks()])
        else:
            history = self.neural_network.fit(self.x_train, self.y_train, epochs=self.epochs,
                                              batch_size=self.batch_size, class_weight=self.class_weights,
                                              verbose=self.verbose,
                                              validation_data=(self.x_val, self.y_val),
                                              callbacks=[cp_callback, CustomCallbacks()])

        for metric_name, metric_values in history.history.items():
            if not "loss" in metric_name:
                if self.fold != "":
                    identifier = self.fold + "_" + metric_name
                else:
                    identifier = metric_name
                wandb.run.summary[identifier] = np.max(metric_values)
        with open(os.path.join(wandb.run.dir, self.fold + "_network_params.pkl"), 'wb') as fp:
            pickle.dump(self.network_params, fp)
            print(f'Network params saved.')
        gl.GLOBAL_HISTORY.append(np.max(history.history['f1_m']))
        with open(os.path.join(wandb.run.dir, self.fold + "_history.txt"), 'w') as fp:
            print(history.history, file=fp)
            print(f'History saved.')
        print(f'Model trained and saved.')
        return

    def evaluate_model(self):
        if not self.neural_network:
            print(f'Network contains no model to evaluate')
            return 0
        loss, accuracy, f1_score = self.neural_network.evaluate(self.x_test, self.y_test, verbose=self.verbose)
        self.model_evaluation_results = {'loss': loss, 'accuracy': accuracy, 'f1_score': f1_score}
        print(f'Model evaluated.')
        return

    def evaluate_prediction_scores(self, class_names, is_last):

        if not self.neural_network:
            print(f'Network contains no model to evaluate')
            return 0

        predictions = self.neural_network.predict(x=self.x_test, verbose=1)
        prediction_classes = np.argmax(predictions, axis=1)
        test_classes = np.argmax(self.y_test, axis=1)
        prediction_classes = self.postprocess(prediction_classes, self.batch_length, 5)

        score_weighted = sk.metrics.f1_score(test_classes, prediction_classes, average='weighted')

        score_classes = sk.metrics.f1_score(test_classes, prediction_classes, average=None)
        conf = sk.metrics.confusion_matrix(test_classes, prediction_classes)
        self.prediction_evaluation_results = {'f1_score_weighted_avg': score_weighted,
                                              'f1_score_per_class': score_classes, 'conf': conf}
        try:
            cn = list(class_names.values())
        except:
            cn = class_names
        print(str(self.prediction_evaluation_results))
        wandb.log({"conf_mat_" + self.fold: wandb.plot.confusion_matrix(probs=None, y_true=test_classes,
                                                                        preds=prediction_classes,
                                                                        class_names=cn)})
        with open(os.path.join(wandb.run.dir, self.fold + "_final_metrics.pkl"), 'wb') as fp:
            pickle.dump(
                {"class_names": class_names, "prediction_classes": prediction_classes, "predictions": predictions}, fp)
            print(f'confusion matrix saved.')
        with open(os.path.join(wandb.run.dir, self.fold + "_f1_scores.txt"), 'w') as fp:
            fp.write(str(self.prediction_evaluation_results))

        print(f'Model prediction scores evaluated.')
        if is_last:
            gl.GLOBAL_HISTORY.append(np.mean(gl.GLOBAL_HISTORY))
            with open(os.path.join(wandb.run.dir, "bestf1s.txt"), 'w') as fp:
                print(gl.GLOBAL_HISTORY, file=fp)
                print(f'History saved.')

        return conf, gl.GLOBAL_HISTORY, os.path.join(wandb.run.dir, "bestf1s.txt")


    @staticmethod
    def predict(model, data):
        probabilities = []
        predictions = []
        for window in data:
            window = np.expand_dims(window, axis=0)
            probability = model.predict(x=window, verbose=0)
            prediction = np.argmax(probability)
            probabilities.append(probability)
            predictions.append(prediction)

        return predictions, probabilities

    @staticmethod
    def postprocess(predictions, window_size, time_period_in_min):

        num_windows = int(time_period_in_min * 60 / (window_size / 25))  # 25 is sampling_rate
        start = 0
        for i in range(num_windows, len(predictions), num_windows):

            try:
                predictions[start:i] = np.argmax(np.unique(predictions[start:i], return_counts=True)[1])
                start = start + num_windows
            except:
                pass
        return predictions

def _postprocess(predictions, window_size, time_period_in_min):

    num_windows = int(time_period_in_min * 60 / (window_size / 25))
    start = 0
    for i in range(num_windows, len(predictions), num_windows):
        try:
            predictions[start:i] = np.argmax(np.unique(predictions[start:i], return_counts=True)[1])
            start = start + num_windows
        except:
            pass
    return predictions
