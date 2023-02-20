import copy
import random

import numpy as np
import pandas as pd
import misc.Helper as helper


# author: Alexander Hoelzemann - alexander.hoelzemann@uni-siegen.de


def make_equidistant(data: pd.DataFrame, new_freq: int):
    freq_new_ms = round(((1 / new_freq) * 1000))

    data = data.set_index("timestamp")
    is_index_unique = data.index.is_unique
    if not is_index_unique:
        import datetime
        data = data.set_index(pd.date_range(start=datetime.datetime.now(), periods=data.shape[0], freq=str(int(freq_new_ms))+'ms'))
    columns = data.columns
    result = []
    activityIDs = data['activityID']
    data = data.drop(['activityID'], axis=1)
    activityIDs = activityIDs.resample(str(int(freq_new_ms)) + "ms", axis=0).nearest()
    new_indices = activityIDs.index
    result.append(activityIDs.to_numpy())

    for column in data:
        c = data[column]
        resampledData = c.resample(str(int(freq_new_ms)) + "ms", axis=0).mean()
        result.append(resampledData)
    result = np.array(result).T
    result = pd.DataFrame(result, columns=columns, index=new_indices, dtype=np.float64)

    return result

def sliding_window(x, window_size, overlapping_samples=0, scheme="last", random_seed=None,
                   labels_column='activityID'):

    from collections import Counter
    data, target = [], []
    start = 0
    end = window_size
    window_indices = []
    weeks_and_days = []
    labels_columns = ['button', 'diary', 'app', 'gui']
    try:
        y = list(x[labels_column])
        x = x.drop(labels_columns, axis=1)
    except:
        y = []
    try:
        week_day = x[['week', 'day']]
        x = x.drop('week', axis=1)
        x = x.drop('day', axis=1)
        x = x.drop('subject', axis=1)
        x = x.drop(x.columns[-1], axis=1)
    except:
        week_day = []
    len_week_day = len(week_day)
    while True:
        data.append(x[start:end])
        window_indices.append([start, end])
        if len_week_day > 0:
            weeks_and_days.append(week_day[start:end].values[-1].tolist())
        if len(y) > 0:
            target.append(int(y[start:end][-1] if scheme == "last" else np.argmax(np.bincount(y[start:end]))))
        if end >= x.shape[0]:
            break
        start = start + window_size - overlapping_samples
        end = end + window_size - overlapping_samples

    if len(data[-1]) < window_size:
        data.pop()
        weeks_and_days.pop()
        window_indices.pop()
        if len(target) > 0:
            target.pop()
    if random_seed != None:
        from sklearn.utils import shuffle
        if len(y) > 0:
            data, target = shuffle(data, target, random_state=random_seed)
    if len_week_day > 0:
        return [data, target, weeks_and_days, window_indices]
    else:
        return [data, target, window_indices]


def filter_activities(data, list_of_activities):
    list_of_activities = np.array(list_of_activities, dtype=data['activityID'].dtype)
    data = data.loc[data['activityID'].isin(list_of_activities)]
    return data


def resample_raw_data(data: pd.DataFrame, new_freq):
    new_freq_ms = int(((1 / new_freq) * 1000))
    data = data.resample(str(round(new_freq_ms)) + "ms").mean()
    data = data.interpolate()

    return data
