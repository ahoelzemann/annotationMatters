
from dataloader.Datasets.Dataset_BangleJS import Dataset as Dataset_BangleJS
from datetime import datetime
import misc.Helper as helper


now = datetime.now().strftime("%H:%M:%S")
print("Application started: ", now)

ids = ['2b88', '36fd', '74e4', '90a4', '834b', '4531', 'a506', 'd8f2', 'eed7', 'f30d', 'fc25']
dataset = {}
for current_id in ids:
    dataset[current_id] = Dataset_BangleJS(data={})
    dataset[current_id].dataset_folder = "../preprocessed/"
    dataset[current_id].decompress_new_data(subject=current_id)
    dataset[current_id].get_gui_label_timestamps(subject=current_id)
    helper.label_data(subject=current_id, week='week1')
    helper.label_data(subject=current_id, week='week2')
now = datetime.now().strftime("%H:%M:%S")
print("Application finished: ", now)

