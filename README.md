# Skin-Cancer-Lesion-Segmentation

This project focuses on segmenting skin lesions from dermoscopic images using a modified SegNet-based deep learning model. To improve generalization and handle variations in lesion appearance, data preprocessing included resizing and normalization, and augmentation was implicitly leveraged through dataset shuffling and variability in training samples. The model was trained on the ISIC 2017 dataset, which contains diverse images with noise such as hair and low contrast, helping the network learn robust features. By exposing the model to varied input conditions during training, it becomes more resistant to overfitting and performs well on unseen data. The modified architecture reduces complexity while maintaining strong performance, achieving high Dice and Jaccard scores.

**Dataset**
![Image](https://github.com/AniketP04/Skin-Cancer-Lesion-Segmentation/blob/main/paper.jpg)

**Model Architecture**
![Image](https://github.com/AniketP04/Skin-Cancer-Lesion-Segmentation/blob/main/model.jpg)
