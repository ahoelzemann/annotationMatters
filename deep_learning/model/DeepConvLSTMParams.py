class DeepConvLSTMParams:

    def __init__(self,
                 dataset_name=None,
                 model_name=None,
                 val_split=None,
                 num_cnn_layers=None,
                 c_filters=None,
                 c_kernel_size=None,
                 c_padding=None,
                 c_activation=None,
                 c_kernel_init=None,
                 c_kernel_regular=None,
                 c_kernel_regular_rate=None,
                 c_bias_init=None,
                 c_bias_regular=None,
                 c_bias_regular_rate=None,
                 pool_type=None,
                 pool_kernel_size=None,
                 num_rnn_layers=None,
                 lstm_units=None,
                 dropout_rate=None,
                 optimizer=None,
                 learning_rate=None,
                 verbose=None,
                 epochs=None,
                 batch_size=None,
                 model_folder_base=None,
                 checkpoint_frequency=None,
                 early_stopping_patience=None,
                 fold=None,
                 purpose=None):
        """
        :param model_name: string, name of the model for output graphics
        :param val_split: float, percentage of training data used for validation, range: 0.01-0.99

        :param num_cnn_layers: integer, amount of conv layers in main convolution core
        :param c_filters: list of integers, filter sizes in conv, length=num_cnn_layers
        :param c_kernel_size: list of tuples of integers, kernel height and width in conv, length=num_cnn_layers+1
        :param c_padding: list of strings, padding for data in conv, length=num_cnn_layers+1,
            options: 'valid', 'same'
        :param c_activation: list of strings, activation functions in conv, length=num_cnn_layers+1,
            options: 'relu', 'sigmoid', 'tanh'
        :param c_kernel_init: list of strings, weight initialization for conv kernel, length=num_cnn_layers+1,
            options: glorot_normal, glorot_uniform, lecun_normal, lecun_uniform, random_normal, random_uniform
        :param c_kernel_regular: list of strings, weight regularization for conv kernel, length=num_cnn_layers+1,
            options: l1, l2, l1_l2
        :param c_kernel_regular_rate: list of list of integers, weight regularization rate for conv kernel,
            length=num_cnn_layers+1
        :param c_bias_init: list of strings, weight initialization for conv bias, length=num_cnn_layers+1,
            options: glorot_normal, glorot_uniform, lecun_normal, lecun_uniform, random_normal, random_uniform
        :param c_bias_regular: list of strings, weight regularization for conv bias, length=num_cnn_layers+1,
            options: l1, l2, l1_l2
        :param c_bias_regular_rate: list of list of integers, weight regularization rate for conv bias,
            length=num_cnn_layers+1
        :param pool_type: string, type of 2d pooling layer, options: 'max', 'avg', 'global_max', 'global_avg'
        :param pool_kernel_size: tuple of integers, kernel height and width in pooling
        :param num_rnn_layers: integer, amount of lstm in main recurrent core
        :param lstm_units: list of integers, size of lstm layer units, length=num_rnn_layers
        :param dropout_rate: float, rate of dropout for droput layer: 0.01 - 0.99
        :param optimizer: string, desired optimizer, options: 'sgd', 'adam', 'adagrad'
        :param learning_rate: float, learning rate of optimizer
        :param verbose: integer, verbosity of network progress output
        :param epochs: integer, number of training epochs
        :param batch_size: integer, batch size of step
        :param model_folder_base: string, folder to save the model to
        :param checkpoint_frequency: integer, number of epochs before a checkpoint is created
        :param early_stopping_patience: integer, early stopping patience in epochs
        """
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.val_split = val_split
        self.num_cnn_layers = num_cnn_layers
        self.c_filters = c_filters
        self.c_kernel_size = c_kernel_size
        self.c_padding = c_padding
        self.c_activation = c_activation
        self.c_kernel_init = c_kernel_init
        self.c_kernel_regular = c_kernel_regular
        self.c_kernel_regular_rate = c_kernel_regular_rate
        self.c_bias_init = c_bias_init
        self.c_bias_regular = c_bias_regular
        self.c_bias_regular_rate = c_bias_regular_rate
        self.pool_type = pool_type
        self.pool_kernel_size = pool_kernel_size
        self.num_rnn_layers = num_rnn_layers
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_folder_base = model_folder_base
        self.checkpoint_frequency = checkpoint_frequency
        self.early_stopping_patience = early_stopping_patience
        self.fold = fold
        self.purpose = purpose

    @staticmethod
    def example_params():
        params = DeepConvLSTMParams(
            dataset_name='',
            model_name='DeepConvLSTM',
            val_split=0.2,
            num_cnn_layers=3,
            c_filters=[64, 128, 256],
            c_kernel_size=[(5, 5), (4, 4), (3, 3), (2, 2)],
            c_padding=['same', 'same', 'same', 'same'],
            c_activation=['relu', 'sigmoid', 'tanh', 'relu'],
            c_kernel_init=['glorot_uniform', 'lecun_uniform', 'random_uniform', 'random_normal'],
            c_kernel_regular=['l2', 'l1', 'l1_l2', 'l2'],
            c_kernel_regular_rate=[[0.01], [0.001], [0.01, 0.001], [0.0001]],
            c_bias_init=['glorot_uniform', 'lecun_uniform', 'random_uniform', 'random_normal'],
            c_bias_regular=['l2', 'l1', 'l1_l2', 'l2'],
            c_bias_regular_rate=[[0.01], [0.001], [0.01, 0.001], [0.0001]],
            pool_type='max',
            pool_kernel_size=(2, 1),
            num_rnn_layers=2,
            lstm_units=[1024, 512],
            dropout_rate=0.3,
            optimizer='adagrad',
            learning_rate=0.001,
            verbose=2,
            epochs=2,
            batch_size=64,
            model_folder_base='./',
            checkpoint_frequency=25,
            early_stopping_patience=300,
            fold='',
            purpose='Baseline_loso'
        )
        return params

    @staticmethod
    def baseline_params():
        params = DeepConvLSTMParams(
            dataset_name='',
            model_name='DeepConvLSTM',
            val_split=0.2,
            num_cnn_layers=3,
            c_filters=[128, 256, 512],
            c_kernel_size=[(20, 1), (10, 1), (5, 1), (2, 1)],
            c_padding=['same', 'same', 'same', 'same'],
            c_activation=['relu', 'relu', 'relu', 'relu'],
            c_kernel_init=['lecun_uniform', 'lecun_uniform', 'lecun_uniform', 'lecun_uniform'],
            c_kernel_regular=['l2', 'l2', 'l2', 'l2'],
            c_kernel_regular_rate=[[0.01], [0.001], [0.01, 0.001], [0.0001]],
            c_bias_init=['lecun_uniform', 'lecun_uniform', 'lecun_uniform', 'lecun_uniform'],
            c_bias_regular=['l2', 'l2', 'l2', 'l2'],
            c_bias_regular_rate=[[0.01], [0.001], [0.01, 0.001], [0.0001]],
            pool_type='max',
            pool_kernel_size=(2, 1),
            num_rnn_layers=1,
            lstm_units=[1024, 1024],
            dropout_rate=0.3,
            optimizer='adagrad',
            learning_rate=0.001,
            verbose=2,
            epochs=1000,
            batch_size=32,
            model_folder_base='./',
            checkpoint_frequency=25,
            early_stopping_patience=0,
            fold='',
            purpose='baseline'
        )
        return params

    @staticmethod
    def loso_params():
        params = DeepConvLSTMParams(
            dataset_name='',
            model_name='DeepConvLSTM',
            val_split=0.2,
            num_cnn_layers=4,
            c_filters=[64, 128, 256, 512],
            c_kernel_size=[(20, 1), (20, 1), (20, 1), (20, 1)],
            c_padding=['same', 'same', 'same', 'same'],
            c_activation=['relu', 'relu', 'relu', 'relu'],
            c_kernel_init=['lecun_uniform', 'lecun_uniform', 'lecun_uniform', 'lecun_uniform'],
            c_kernel_regular=['l2', 'l2', 'l2'],
            c_kernel_regular_rate=[[0.01], [0.001], [0.01, 0.001], [0.0001]],
            c_bias_init=['glorot_uniform', 'lecun_uniform', 'random_uniform', 'random_normal'],
            c_bias_regular=['l2', 'l2', 'l2', 'l2'],
            c_bias_regular_rate=[[0.01], [0.001], [0.01, 0.001], [0.0001]],
            pool_type='max',
            pool_kernel_size=(2, 1),
            num_rnn_layers=1,
            lstm_units=[512],
            dropout_rate=0.3,
            optimizer='adam',
            learning_rate=0.001,
            verbose=2,
            epochs=10,
            batch_size=64,
            model_folder_base='./',
            checkpoint_frequency=25,
            early_stopping_patience=0,
            fold='',
            purpose='lodo'
        )

        return params
