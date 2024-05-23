import numpy as np
import pandas as pd

import preprocessing.Preprocessing as preprocessor
from dataloader.DataLoader import SensorDataset


class Dataset(SensorDataset):
    def __init__(self, window_size=50, overlapping_samples=20, name='Bangle.js Data', status=None, random_seed=None,
                 data=None):
        if status is None:
            status = []
        super().__init__(window_size, overlapping_samples, name, status, sensitivity_mg=8000, global_sensor_limits=8,
                         recording_freq=25,
                         random_seed=random_seed, data=data)

    def decompress_new_data(self, subject=None):

        import glob
        import os.path
        decompressed_folder = 'folder'
        raw_folder = 'folder'
        self.windows = {}
        subfolders_weeks = ['week1', 'week2']
        subfolders_days = ['day1', 'day2', 'day3', 'day4', 'day5', 'day6', 'day7']

        for week in subfolders_weeks:
            for day in subfolders_days:
                daily_files = glob.glob(raw_folder + subject + "/" + week + "/" + day + "/" + "*.bin")
                daily_files.sort()
                if len(daily_files) > 0:
                    if not os.path.exists(decompressed_folder + subject + '_' + week + '_' + day + ".csv"):
                        tmp = unpack(daily_files, self.recording_freq)
                        tmp.to_csv(
                            decompressed_folder + subject + '_' + week + '_' + day + ".csv")
                        print(subject + '_' + week + '_' + day + " saved")

    def get_gui_label_timestamps(self, subject=None):
        import os

        possible_labels = ['void', 'sitting', 'walking', 'running', 'car_driving', 'cycling', 'badminton', 'showering',
                             'horse_riding', 'table_tennis', 'eating', 'gardening', 'playing_games', 'dish_washing',
                              'cooking', 'cleaning', 'vacuum_cleaning', 'laundry', 'weightlifting', 'other']
        decompressed_folder = 'folder'
        raw_folder = 'folder'
        subfolders_weeks = ['week2']
        subfolders_days = ['day1', 'day2', 'day3', 'day4', 'day5', 'day6', 'day7']
        result = []
        for week in subfolders_weeks:
            for day in subfolders_days:
                try:
                    self.data = pd.read_csv(decompressed_folder + subject + '_' + week + '_' + day + ".csv")
                    self.data = self.data.set_index(self.data.columns[0])
                    gui_labels = pd.read_csv(
                        raw_folder + subject + "/" + week + "/" + day + "/" + "IMU_Wrist_Activity.csv")
                except:
                    continue
                labels_array = np.full(fill_value=0, shape=(self.data.shape[0], 1))
                for label in gui_labels.iterrows():
                    start = label[1]['start']
                    end = self.data.shape[0]-1 if label[1]['end'] >= self.data.shape[0] else label[1]['end']
                    try:
                        description = label[1]['description'].replace(",", "").replace("'", "").replace("(",
                                                                                                        "").replace(")",
                                                                                                                    "")
                        labels_array[start:end] = possible_labels.index(description)
                        result.append([week + "_" + day, self.data.index[start].split(" ")[1],
                                       self.data.index[end].split(" ")[1], description])
                    except:
                        labels_array[start:end] = possible_labels.index('other')
                        result.append([week + "_" + day, self.data.index[start].split(" ")[1],
                                       self.data.index[end].split(" ")[1], 'other'])

                self.data = pd.concat([self.data, pd.DataFrame(labels_array, columns=['gui'])], axis=1, ignore_index=True)
        result = pd.DataFrame(result)
        result.to_csv(
            "../dataset/labels/" + subject + "_gui.csv")




def unpack(paths, resampling_freq):
    substring_in_list = [string for string in paths if "d20statusmsgs.bin" in string]
    try:
        paths.remove(substring_in_list[0])
    except:
        pass
    result = list(map(readBinFile, paths))
    i = 0
    subjectData = []
    for i in range(0, len(paths)):
        if len(result[i]) > 0:
            subjectData.append(decompress(result[i]))

        i += 1

    subjectData, true_freqs = resample(subjectData, resampling_freq)
    subjectData = pd.concat(subjectData, axis=0)
    subjectData = preprocessor.resample_raw_data(subjectData, resampling_freq)
    return subjectData


def resample(subject_files, new_freq):
    true_freqs = []

    for fc in range(len(subject_files)):
        current_df = subject_files[fc]
        if fc != len(subject_files) - 1:
            next_df = subject_files[fc + 1]

            current_df = current_df.append(next_df.iloc[0], ignore_index=False)
            t = pd.date_range(start=current_df.index[0],
                              end=current_df.index[-1],
                              periods=current_df.shape[0])
            current_df = current_df.set_index(t)[:-1]

        else:
            true_freq = 25 if true_freqs[-1] == 24 or true_freqs[-1] == 25 else true_freqs[-1] + 1
            true_freq_new_ms = int(((1 / true_freq) * 1000))

            new_range = pd.date_range(current_df.index[0], current_df.index[-1],
                                      freq=str(round(true_freq_new_ms)) + "ms")

            current_df = current_df.set_index(pd.DatetimeIndex(new_range[:current_df.shape[0]]))

        subject_files[fc] = current_df
        true_freqs.append(round(np.mean(checkfordrift(current_df))))

    return subject_files, true_freqs


def readBinFile(path):
    bufferedReader = open(path, "rb")
    return bufferedReader.read()


def int64_to_str(a, signed):
    import math

    negative = signed and a[7] >= 128
    H = 0x100000000
    D = 1000000000
    h = a[4] + a[5] * 0x100 + a[6] * 0x10000 + a[7] * 0x1000000
    l = a[0] + a[1] * 0x100 + a[2] * 0x10000 + a[3] * 0x1000000
    if negative:
        h = H - 1 - h
        l = H - l

    hd = math.floor(h * H / D + l / D)
    ld = (((h % D) * (H % D)) % D + l) % D
    ldStr = str(ld)
    ldLength = len(ldStr)
    sign = ''
    if negative:
        sign = '-'
    if hd != 0:
        result = sign + str(hd) + ('0' * (9 - ldLength))
    else:
        result = sign + ldStr

    return result


def decompress(bin_file):
    from datetime import datetime
    import numpy as np
    import pandas as pd

    # value = np.uint8(data)
    hd = bin_file[0: 32]
    accxA = []
    accyA = []
    acczA = []

    ts = np.frombuffer(bin_file[0:8], dtype=np.int64)[0]
    millis = int64_to_str(hd, True)
    GS = 8
    HZ = 12.5
    if hd[8] == 16:
        GS = 8
    elif hd[8] == 8:
        GS = 4
    elif hd[8] == 0:
        GS = 2
    if hd[9] == 0:
        HZ = 12.5
    elif hd[9] == 1:
        HZ = 25
    elif hd[9] == 2:
        HZ = 50
    elif hd[9] == 3:
        HZ = 100
    if HZ == 100:
        HZ = 90  # HACK!!
    delta = False
    deltaval = -1
    packt = 0
    sample = np.zeros(6, dtype='int64')
    # infoStr = "header: " + str(hd) + "\n" do not preceed with #!––
    lbls = []
    itr = 0
    for ii in range(32, len(bin_file) - 3, 3):  # iterate over data
        if (ii - 32) % 7200 == 0:
            pass
            # infoStr += "\n==== new page ====\n"  # mark start of new page
        if not delta:
            if (int(bin_file[ii]) == 255) and (int(bin_file[ii + 1]) == 255) and (packt == 0):  # delta starts
                if int(bin_file[ii + 2]) == 255:
                    pass
                    # infoStr += "\n*" + str((ii + 2)) + "\n"  # error -> this should only happen at the end of a page
                else:
                    # infoStr += "\nd" + value[ii + 2] + ":"
                    delta = True
                    deltaval = int(bin_file[ii + 2])
            else:
                if packt == 0:
                    sample[0] = int(bin_file[ii])
                    sample[1] = int(bin_file[ii + 1])
                    sample[2] = int(bin_file[ii + 2])
                    packt = 1
                else:
                    sample[3] = int(bin_file[ii])
                    sample[4] = int(bin_file[ii + 1])
                    sample[5] = int(bin_file[ii + 2])
                    packt = 0
                    mts = datetime.fromtimestamp(ts / 1000 + itr * (1000 / HZ) / 1000)
                    lbls.append(mts)
                    tmp = np.int16(sample[0] | (sample[1] << 8))
                    accxA.append(round((tmp / 4096), 5))
                    tmp = np.int16(sample[2] | (sample[3] << 8))
                    accyA.append(round((tmp / 4096), 5))
                    tmp = np.int16(sample[4] | (sample[5] << 8))
                    acczA.append(round((tmp / 4096), 5))
                    itr += 1


        else:
            sample[0] = int(bin_file[ii])
            sample[2] = int(bin_file[ii + 1])
            sample[4] = int(bin_file[ii + 2])  # fill LSBs after delta
            mts = datetime.fromtimestamp(ts / 1000 + itr * (1000 / HZ) / 1000)
            lbls.append(mts)
            tmp = np.int16(sample[0] | (sample[1] << 8))
            accxA.append(round((tmp / 4096), 5))
            tmp = np.int16(sample[2] | (sample[3] << 8))
            accyA.append(round((tmp / 4096), 5))
            tmp = np.int16(sample[4] | (sample[5] << 8))
            acczA.append(round((tmp / 4096), 5))
            itr += 1
            deltaval -= 1
            if (deltaval < 0):
                delta = False

    activity_data = np.array([accxA, accyA, acczA], dtype=np.float64).T
    dataframe = pd.DataFrame(data=activity_data, columns=["x_axis", "y_axis", "z_axis"], index=lbls)
    dataframe = dataframe[~dataframe.index.duplicated(keep='first')]

    return dataframe


def checkfordrift(df):
    timestamps = df.index
    counter = 1
    true_freqs = []
    last = None
    for entry in timestamps:
        second = entry.second
        if last is not None:
            if second != last:
                true_freqs.append(counter)
                counter = 1
            else:
                counter = counter + 1
        last = second

    return true_freqs
