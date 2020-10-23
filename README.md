# CVND-Image-Captioning-Project

This implementation of the [CVND-Image-Captioning-Project](https://github.com/udacity/CVND---Image-Captioning-Project) is built for Udacity's Computer Vision Nanodegree.

## Dependencies:
***Libraries***

1. Create (and activate) a new environment, named `cv-nd` with Python 3.6. If prompted to proceed with the install `(Proceed [y]/n)` type y.

	- __Linux__ or __Mac__: 
	```
	conda create -n cv-nd python=3.6
	source activate cv-nd
	```
	- __Windows__: 
	```
	conda create --name cv-nd python=3.6
	activate cv-nd
	```
	
	At this point your command line should look something like: `(cv-nd) <User>:P1_Facial_Keypoints <user>$`. The `(cv-nd)` indicates that your environment has been activated, and you can proceed with further package installations.

2. Install PyTorch and torchvision; this should install the latest version of PyTorch.
	
	- __Linux__ or __Mac__: 
	```
	conda install pytorch torchvision -c pytorch 
	```
	- __Windows__: 
	```
	conda install pytorch-cpu -c pytorch
	pip install torchvision
	```

6. Install a few required pip packages, which are specified in the requirements text file (including OpenCV).
```
pip install -r requirements.txt
```

***Dstaset***
1. To download the COCO dataset, clone this repo: https://github.com/cocodataset/cocoapi  
```
git clone https://github.com/cocodataset/cocoapi.git
```

2. Setup the coco API (also described in the readme [here](https://github.com/cocodataset/cocoapi)) 
```
cd cocoapi/PythonAPI  
make  
cd ..
```

3. Download some specific data from here: http://cocodataset.org/#download (described below)

* Under **Annotations**, download:
  * **2014 Train/Val annotations [241MB]** (extract captions_train2014.json and captions_val2014.json, and place at locations cocoapi/annotations/captions_train2014.json and cocoapi/annotations/captions_val2014.json, respectively)  
  * **2014 Testing Image info [1MB]** (extract image_info_test2014.json and place at location cocoapi/annotations/image_info_test2014.json)

* Under **Images**, download:
  * **2014 Train images [83K/13GB]** (extract the train2014 folder and place at location cocoapi/images/train2014/)
  * **2014 Val images [41K/6GB]** (extract the val2014 folder and place at location cocoapi/images/val2014/)
  * **2014 Test images [41K/6GB]** (extract the test2014 folder and place at location cocoapi/images/test2014/)


**The project is structured as a series of Jupyter notebooks and `.py` files:**
- [0_Dataset.ipynb](https://github.com/Farida-Hamid/CVND-Image-Captioning-Project/blob/main/0_Dataset.ipynb)
- [1_Preliminaries.ipynb](https://github.com/Farida-Hamid/CVND-Image-Captioning-Project/blob/main/1_Preliminaries.ipynb)
- [2_Training.ipynb](https://github.com/Farida-Hamid/CVND-Image-Captioning-Project/blob/main/2_Training.ipynb)
- [3_Inference.ipynb](https://github.com/Farida-Hamid/CVND-Image-Captioning-Project/blob/main/3_Inference.ipynb)
- [model.py](https://github.com/Farida-Hamid/CVND-Image-Captioning-Project/blob/main/model.py)
- [vocabulary.py](https://github.com/Farida-Hamid/CVND-Image-Captioning-Project/blob/main/vocabulary.py)
- [data_loader.py](https://github.com/Farida-Hamid/CVND-Image-Captioning-Project/blob/main/data_loader.py)

## Results:
The final output is one senense per image. In [3_Inference.ipynb](https://github.com/Farida-Hamid/CVND-Image-Captioning-Project/blob/main/3_Inference.ipynb), results are given for some images chosen randomly.

Overall, the model identifies objects and describes them correctly but may give partly correct explanation of some images.

![Correctly captioned images](https://github.com/Farida-Hamid/CVND-Image-Captioning-Project/blob/main/images/good.png?raw=true "Title")

**Correctly captioned images**



![Almost correctly captioned images](https://github.com/Farida-Hamid/CVND-Image-Captioning-Project/blob/main/images/bad.png?raw=true "Title")

**Almost correctly captioned images**
