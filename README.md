# A Matter of Annotation: An Empirical Study on In Situ and Self-Recall Activity Annotations from Wearable Sensors

This is the official Github repository of the paper A Matter of Annotation: An Empirical Study on In Situ and Self-Recall Activity Annotations from Wearable Sensors.

## Abstract:
Research into the detection of human activities from wearable sensors is a highly active field, benefiting numerous applications, from ambulatory monitoring of healthcare patients via fitness coaching to streamlining manual work processes.
We present an empirical study that compares 4 different commonly used annotation methods utilized in user studies that focus on in-the-wild data. These methods can be grouped in user-driven, in situ annotations - which are performed before or during the activity is recorded - and recall methods - where participants annotate their data in hindsight at the end of the day.
Our study illustrates that different labeling methodologies directly impact the annotations' quality, as well as the capabilities of a deep learning classifier trained with the data respectively. 
We noticed that in situ methods produce less but more precise labels than recall methods. Furthermore, we combined an activity diary with a visualization tool that enables the participant to inspect and label their activity data. Due to the introduction of such a tool were able to decrease missing annotations and increase the annotation consistency, and therefore the F1-score of the deep learning model by up to 8% (ranging between 82.1 and 90.4% F1-score). 
Furthermore, we discuss the advantages and disadvantages of the methods compared in our study, the biases they may could introduce and the consequences of their usage on human activity recognition studies and as well as possible solutions.

## Dataset 
The datasets generated during and/or analysed during the current study are available in the Zenodo repository: https://doi.org/10.5281/zenodo.7654684.
The repository contains additional information about the dataset.

## Available log files of experiments
Logs of our deep learning experiments across all evaluated participants are available on Weights and Biases https://tinyurl.com/4vxvfaed.

For further questions please contact alexander.hoelzemann@uni-siegen.de