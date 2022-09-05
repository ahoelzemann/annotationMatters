from mad_gui import start_gui
from mad_gui.plot_tools.labels import BaseRegionLabel

from custom_importer import CustomImporter
from custom_importer import CustomExporter
from custom_label import My_Label as labels
from mad_gui.config import BaseTheme
from PySide2.QtGui import QColor
from mad_gui.plugins.base import BaseImporter


class MyTheme(BaseTheme):
    # COLOR_DARK = QColor(0, 255, 100)
    # COLOR_LIGHT = QColor(255, 255, 255)
    FAU_PHILFAK_COLORS = {
        "dark": QColor(255, 0, 0),
        "medium": QColor(0, 255, 0),
        "light": QColor(0, 0, 255),
    }

class LayerOne(BaseRegionLabel):
    # This label will always be shown at the upper 20% of the plot view
    min_height = 0.8
    max_height = 1
    name = "Activity"
    color = [1, 255, 0, 100]
    descriptions = {"walking": None, "running": None, 'cycling': None}


start_gui(plugins=[CustomImporter, CustomExporter], theme=MyTheme, labels=[LayerOne])
