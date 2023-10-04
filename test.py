#!/usr/bin/env python
# coding: utf-8

# In[1]:

import requests
import json
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import requests
import json
from tqdm.notebook import tnrange, tqdm
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from os import walk
import cv2

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
# from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img, img_to_array 
from sklearn.metrics.pairwise import cosine_similarity

from scipy.sparse import hstack

from sklearn.preprocessing import normalize

# Convert the result to JSON and write to file
with open('output.json', mode='w', encoding='utf8') as outfile:
    outfile.write('{"out":"this is a test output"}')


# In[ ]:




