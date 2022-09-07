from typing import Dict
import warnings
import pandas as pd
from mad_gui import start_gui, BaseImporter, BaseExporter
from mad_gui.models import GlobalData
from PySide2.QtWidgets import QFileDialog
from mad_gui.components.dialogs import UserInformation
from pathlib import Path
import os
import datetime

class CustomImporter(BaseImporter):
    loadable_file_type = "*.*"

    @classmethod
    def name(cls) -> str:
        ################################################
        ###                   README                 ###
        ### Set your importer's name as return value ###
        ### This name will show up in the dropdown.  ###
        ################################################
        # warnings.warn("The importer has no meaningful name yet."
        #               " Simply change the return string and remove this warning.")
        return "PerCom2023"

    def load_sensor_data(self, file_path: str) -> Dict:
        ##################################################################
        ###                       README                               ###
        ### a) Use the argument `file_path` to load data. Transform    ###
        ###    it to a pandas dataframe (columns are sensor channels,  ###
        ###    as for example "acc_x". Assign it to sensor_data.       ###
        ###                                                            ###
        ### b) load the sampling rate (int or float)                   ###
        ##################################################################

        # warnings.warn("Please load sensor data from your source."
        #               " Just make sure, that sensor_data is a pandas.DataFrame."
        #               " Afterwards, remove this warning.")
        # sensor_data = pd.read_csv(file_path, names=["acc_x", "acc_y", "acc_z"])[1:]
        #
        sensor_data = pd.read_csv(file_path)[["acc_x", "acc_y", "acc_z"]]
        # warnings.warn("Please load the sampling frequency from your source in Hz"
        #               " Afterwards, remove this warning.")
        sampling_rate_hz = 25
        # sampling_rate_hz = 1 / sensor_data["time"].diff().mean()

        ##############################################################
        ###                      CAUTION                           ###
        ### If you only want to have one plot you do not need to   ###
        ### change the following lines! If you want several plots, ###
        ### just add another sensor like "IMU foot" to the `data`  ###
        ### dictionary, which again hase keys sensor_data and      ###
        ### and sampling_rate_hz for that plot.                    ###
        ##############################################################
        data = {
            "IMU Wrist": {
                "sensor_data": sensor_data,
                "sampling_rate_hz": sampling_rate_hz,
                "start_time": sensor_data.index[0]
            }
        }

        return data

    def get_start_time(self, *args, **kwargs) -> datetime.time:
        sensor_data_index = pd.read_csv(args[0], names=["acc_x", "acc_y", "acc_z"])[1:].index[0]
        return datetime.datetime.strptime(sensor_data_index, '%Y-%m-%d %H:%M:%S.%f').time()


class CustomExporter(BaseExporter):
    @classmethod
    def name(cls):
        return "Export annotations to csv (MaD GUI example)"

    def process_data(self, global_data: GlobalData):
        directory = QFileDialog().getExistingDirectory(
            None, "Save .csv results to this folder", str(Path(global_data.data_file).parent)
        )
        for plot_name, plot_data in global_data.plot_data.items():
            for label_name, annotations in plot_data.annotations.items():
                if len(annotations.data) == 0:
                    continue
                annotations.data.to_csv(
                    directory + os.sep + plot_name.replace(" ", "_") + "_" + label_name.replace(" ", "_") + ".csv"
                )

        UserInformation.inform(f"The results were saved to {directory}.")