.
├── configs.py   # Configuration settings for the model.
├── data_loader.py  # Data loader for the model.
├── models  # Implementation of the proposed model.
│   ├── BoTNet.py
│   ├── __init__.py
│   ├── model.py
│   ├── __pycache__
│   ├── resnet.py
│   └── transformer.py
├── pretrained_model  # Directory for saving pre-trained models.
├── pretraining  # Using multi-gpu for pre-training the BoTNet model.
│   ├── BoTNet.py
│   ├── data_loader.py
│   ├── gengerate_dis_img # gengerate distortion images.
│   ├── log
│   ├── multi_gpu_train.py  # Script for multi-GPU pre-training.
│   ├── train_dpm.sh
│   └── train.py 
├── train_cross_datast.py  #  Cross-dataset testing code.
├── train_cross_datast.sh  # Script for cross-dataset testing.
├── train.py  # Single-dataset testing code.
├── train.sh  # Script for single-dataset testing.
└── tree.md  #File containing the project directory structure.