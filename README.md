## Transfer Learned Neural Network for COVID-19 Detection

In this GitHub you can find the code for two COVID-19 detector networks. One which is build using transfer learning from a DenseNet-121 network pretrained on ImageNet, and one which is build using transfer learning from the CheXNet reproduction from [Bruce Chou](https://github.com/brucechou1983/CheXNet-Keras). 
Also the dataset we assembled can be found in this GitHub repository. The data in the dataset is derived from the [ieee8023](https://github.com/ieee8023/covid-chestxray-dataset) and the [CheXpert GitHub](https://stanfordmlgroup.github.io/competitions/chexpert/).

Both networks give great results, with the ImageNet pre-trained network having the edge with an accuracy of 0.98 on our test set!

A link to the corresponding blog post:

[Hack-MD Blog Post](https://hackmd.io/6iSDz7_DSwWz4yypc21yQw?view)

The pretrained CheXNet can be found in this repository:

[brucechou1983/CheXNet-Keras](https://github.com/brucechou1983/CheXNet-Keras)
**[Download Link](https://drive.google.com/open?id=19BllaOvs2x5PLV_vlWMy4i8LapLb2j6b)**

**Colab for this project:**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1x1hfAmBbmBZwFxbEMPaMBbBxhb_Eqyt1?usp=sharing)
