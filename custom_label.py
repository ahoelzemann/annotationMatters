from mad_gui.plot_tools.labels import BaseRegionLabel
from mad_gui import start_gui

class My_Label(BaseRegionLabel):
   # This label will always be shown at the lowest 20% of the plot view
   min_height = 0
   max_height = 0.2
   name = "Anomaly Label"

   # Snapping will be done, if you additionally pass a Settings object to the GUI,
   # which has an attribute SNAP_AXIS. See the README, the part of Adjusting Constants
   # for more information
   snap_to_min = False
   # snap_to_max = False  # if setting this to `True`, set `snap_to_min` to `False` or delete it

   # User will be asked to set the label's description when creating a label.
   # This can have an arbitrary amount of levels with nested dictionaries.
   # This es an example for a two-level description:
   descriptions = {"normal": None, "anomaly": ["too fast", "too slow"]}