class SensorDataset:
    """
    a generic dataset class for multimodal time-series data captured with wearable sensors
    """

    def __init__(self, window_size, overlapping_samples, name, status, sensitivity_mg, global_sensor_limits, recording_freq, random_seed, data=None):

        if data is not None:
            self.data = data
        else:
            self.data = None
        self.loso_folds = None
        self.window_size: int = window_size
        self.overlapping_samples: int = overlapping_samples
        self.name: str = name
        self.status: list = status
        self.class_names = []
        self.samples_per_class = None
        self.samples_per_subject = None
        self.sensitivity_mg = sensitivity_mg
        self.global_sensor_limits = global_sensor_limits
        self.recording_freq = recording_freq
        self.random_seed = random_seed

        # and so on...

    def get_shape(self):
        if self.data is None:
            return "dataset is not loaded"
        else:
            return self.data.shape

    def __getitem__(self, index):
        pass

    def get_class_names(self):
        # needs to be implemented in subclass
        pass

    def load(self):
        pass



    def load_from_original_files(self):
        # needs to be implemented in subclass
        pass

    def filter_columns(self, keep_timestamps, columns):
        # needs to be implemented in subclass
        pass

    def filter_activities(self, activities):
        # needs to be implemented in subclass
        pass

    def filter_subjects(self, subjects):
        # needs to be implemented in subclass
        pass

    def preprocess(self, freq_new, norm_strategy):
        # needs to be implemented in subclass
        pass

    def print_class_distribution(self):

        import matplotlib.pyplot as plt
        import seaborn as sns

        if not any("preprocessed" in string for string in self.status):
            self.preprocess()

        sns.set_style('white')
        ax = sns.barplot(x='classes', y='n_samples',
                         data=self.samples_per_class)
        for p in ax.patches:
            ax.annotate(format(p.get_height()),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 7), textcoords='offset points')
        plt.title(self.name + ': Total number of samples per class')
        plt.xlabel('Classes')
        plt.ylabel('Samples per class')

        plt.yticks(fontsize=12)
        sns.despine(bottom=True)
        ax.tick_params(bottom=False, left=True)
        for _, s in ax.spines.items():
            s.set_color('black')
        plt.show()

    def print_subject_distribution(self):
        import matplotlib.pyplot as plt
        import seaborn as sns

        if not any("preprocessed" in string for string in self.status):
            self.preprocess()

        sns.set_style('white')
        ax = sns.barplot(x='subject', y='n_samples',
                         data=self.samples_per_subject)
        for p in ax.patches:
            ax.annotate(format(p.get_height()),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 7), textcoords='offset points')
        plt.title(self.name + ': Total number of samples per subject')
        plt.xlabel('Subjects')
        plt.ylabel('Samples per subject')

        plt.yticks(fontsize=12)
        sns.despine(bottom=True)
        ax.tick_params(bottom=False, left=True)
        for _, s in ax.spines.items():
            s.set_color('black')
        plt.show()


    def getNextFold(self, next):
        tmp_test = self.data[next['test']]
        x_train, y_train ,x_test, y_test = [], [], [], []
        for window in tmp_test:
            y_test.append(window['activityID'])
            x_test.append(window.loc[:, window.columns != 'activityID'])

        for subject in next['train']:
            tmp_train = self.data[subject]
            tmp_x, tmp_y = [], []
            for window in tmp_train:
                tmp_y.append(window['activityID'])
                tmp_x.append(window.loc[:, window.columns != 'activityID'])
            x_train = x_train + tmp_x
            y_train = y_train + tmp_y
            # Y = current_subject_data

        # y_train = map(train_set, )
        return x_train, y_train ,x_test, y_test

    @classmethod
    def load_preprocessed_dataset(cls, dataset_name, concat=False, balanced=True):
        import misc.Globals as GLOBAL_VARIABLES
        import pickle
        dataset_name = dataset_name.lower()
        dataset_path = ""
        if not concat:
            if dataset_name == "pamap" or dataset_name == "pamap2":
                dataset_path = "/local_files/pamap2/pamap2.pkl"
            elif dataset_name == "realworld" or dataset_name == "realworld2016":
                dataset_path = "/local_files/realworld2016/realworld2016.pkl"
            elif dataset_name == "utd" or dataset_name == "utd-mhad" or dataset_name == "utd_mhad":
                dataset_path = "/local_files/UTD-MHAD/utd-mhad.pkl"
            elif dataset_name == "AnnoationMatters" or dataset_name == "AnnoationMatters" or dataset_name == "AnnoationMatters" or len(dataset_name) == 5:

                dataset_path = "/local_files/AnnoationMatters/preprocessed/AnnoationMatters_"+dataset_name+"_labeled.pkl"
            with open(GLOBAL_VARIABLES.PROJECT_ROOT + dataset_path, 'rb') as f:
                tmp = pickle.load(f)
        else:
            if dataset_name == "pamap" or dataset_name == "pamap2":
                dataset_path = "/local_files/pamap2/pamap2_concatenated.pkl"
            elif dataset_name == "realworld" or dataset_name == "realworld2016":
                dataset_path = "/local_files/realworld2016/realworld2016_concatenated.pkl"
            elif dataset_name == "utd" or dataset_name == "utd-mhad" or dataset_name == "utd_mhad":
                dataset_path = "/local_files/UTD-MHAD/utd_mhad_concatenated.pkl"
            with open(GLOBAL_VARIABLES.PROJECT_ROOT + dataset_path, 'rb') as f:
                tmp = pickle.load(f)

        return tmp


