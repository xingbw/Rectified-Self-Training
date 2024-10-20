# Rectifying Self-Training with Neighborhood Consistency and Proximity for Source Free Domain Adaptation


### [!] Codes are being updated, please wait for further release.


The codes are implemented with PyTorch 1.6 and Cuda 10.2.

### Dataset preparing

Download the datasets including VisDA-2017, Office-Home and Office-31. Then set the path of data list in the codes. 


### VisDA

First train the model on source domain, then conduct target adaptation without source data. Run the following commands:
> python src_pretrain.py
>
> python tar_adaptation_st.py

### Office-Home

Codes for Office-Home are in the 'office-home' folder, enter the folder and then run the following commands:

> sh train_src_officehome.sh
>
> sh train_tar_officehome.sh


### Office-31

Codes for Office-31 are in the 'office-31' folder, enter the folder and then run the following commands:

> sh train_src_office31.sh
>
> sh train_tar_office31.sh


#### Acknowledgement

The codes are based on [AaD](https://arxiv.org/abs/2205.04183),  thanks the authors for their efforts.
