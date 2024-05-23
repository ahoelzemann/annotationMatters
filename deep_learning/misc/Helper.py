import numpy as np
import pandas as pd


def check_if_dataset_has_sensor(x, ls):
    import numpy as np
    indices = []

    matching = [s for s in ls if x in s]
    if len(matching) == 0:
        return [], []
    else:
        for match in matching:
            indices.append(np.where(ls.values == match)[0][0])
        return matching, indices


def pca(data):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    plt.style.use('seaborn')
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(data.drop(['activityID'], axis=1))
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('Principle Component Analysis', fontsize=20)
    ax.scatter(principalComponents[:, 0], principalComponents[:, 1])
    ax.grid()
    plt.show()

    return True


def get_all_labels_with_coordinates(data, labels_column, index_is_time=True):
    import numpy as np

    standard_time_diff = (data.index[1] - data.index[0]).total_seconds() * 1000
    unique_labels = np.unique(data[labels_column])
    starts_dict = {}
    ends_dict = {}
    for label in unique_labels:
        locs_indices = data.index[data[labels_column] == label].tolist()

        starts = []
        ends = []
        starts.append(locs_indices[0])
        for i in range(len(locs_indices) - 1):
            current = locs_indices[i]
            next = locs_indices[i + 1]
            if not index_is_time:
                if next - current > 1:
                    starts.append(next)
                    ends.append(current)
            else:
                diff_in_milliseconds = (next - current).total_seconds() * 1000
                if diff_in_milliseconds > standard_time_diff:
                    starts.append(next)
                    ends.append(current)
        ends.append(locs_indices[-1])
        starts_dict[label] = starts
        ends_dict[label] = ends

    return starts_dict, ends_dict


def combine_data_and_predictions(sensor_data, predictions, window_indices, probabilities, class_names):
    labels = np.full(fill_value='not_labeled', shape=(sensor_data.shape[0], 1), dtype=object)
    local_probababilities = np.full(fill_value=0.0, shape=(sensor_data.shape[0], 1))
    class_names = list(class_names.values())
    for i in range(0, len(predictions)):
        prediction = predictions[i]
        window_index = window_indices[i]

        label = class_names[prediction]
        labels[window_index[0]:window_index[1]] = label
        local_probababilities[window_index[0]:window_index[1]] = probabilities[i][0][prediction]
    tmp = pd.DataFrame(np.hstack([labels, local_probababilities]), columns=['label', 'probability'], index=sensor_data.index)
    sensor_data = pd.concat([sensor_data, tmp], axis=1)
    return sensor_data


def get_immediate_subdirectories(a_dir):
    import os
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def compute_cohens_k(data, gt_column, week, ):
    from sklearn.metrics import cohen_kappa_score
    compare_to_columns = ['button', 'app'] if week == "week1" else ['button', 'app', 'diary']
    result = []
    for column in compare_to_columns:
        gt = data[[gt_column]]
        cc = data[[column]]
        result.append([gt_column + "_" + column, round(cohen_kappa_score(gt, cc), 3)])

    return result

def label_data(subject, week):
    import pandas as pd
    import numpy as np
    import bisect
    import datetime
    possible_labels = [possible_labels = ['laying', 'sitting', 'walking', 'running', 'cycling', 'bus_driving', 'car_driving', 'cleaning', 'vacuum_cleaning', 'laundry', 
                           'cooking', 'eating', 'shopping', 'showering', 'yoga', 'sport', 'playing_games', 'desk_work', 'guitar_playing', 'gardening', 'table_tennis', 
                           'badminton', 'horse_riding', 'cleaning', 'reading', 'weightlifting', 'manual_work', 'dish_washing']]
    try:
        labels = pd.read_csv(
            "../dataset/labels/" + week + "_" + subject + ".csv")
    except:
        print(subject + ": no labels found.")
        return
    preprocessed_folder = '../dataset/preprocessed/'
    days = ['day1', 'day2', 'day3', 'day4', 'day5', 'day6', 'day7']
    for day in days:
        data = pd.read_csv(preprocessed_folder + subject + '_' + week + '_' + day + ".csv")
        data = data.set_index(data.columns[0])
        data.index = pd.DatetimeIndex(data.index)
        labels_array = np.full(fill_value=0, shape=(data.shape[0], 4))

        for label in labels.iterrows():
            week_label, day_label = label[1]['week_day'].split("_")
            layer = label[1]['layer']
            if week_label == week and day_label == day:
                if label[1]['activity'] != '-':
                    try:
                        activitiy = possible_labels.index(label[1]['activity'])
                    except:
                        activitiy = possible_labels.index('other')
                    try:
                        start_index = bisect.bisect(data.index.time,
                                                    datetime.datetime.strptime(label[1]['start'],
                                                                               "%H:%M:%S.%f").time()) - 1
                        stop_index = bisect.bisect(data.index.time,
                                                   datetime.datetime.strptime(label[1]['stop'],
                                                                              "%H:%M:%S.%f").time()) - 1
                    except:
                        start_index = bisect.bisect(data.index.time,
                                                    datetime.datetime.strptime(label[1]['start'],
                                                                               "%H:%M:%S").time()) - 1
                        stop_index = bisect.bisect(data.index.time,
                                                   datetime.datetime.strptime(label[1]['stop'],
                                                                              "%H:%M:%S").time()) - 1
                    label_indices = np.arange(start=start_index, stop=stop_index)

                    if layer == 'b':
                        labels_array[:, 0][label_indices] = activitiy
                    elif layer == 'd':
                        labels_array[:, 1][label_indices] = activitiy
                    elif layer == 'a':
                        labels_array[:, 2][label_indices] = activitiy
                    elif layer == 'g':
                        labels_array[:, 3][label_indices] = activitiy
        data = data.reset_index()
        data = pd.concat([data, pd.DataFrame(labels_array, columns=['button', 'diary', 'app', 'gui'])],
                         axis=1).set_index(data.columns[0])
        data.to_csv(preprocessed_folder + subject + '_' + week + '_' + day + "_labeled.csv")
        print(subject + '_' + week + '_' + day + "_labeled.csv saved")
