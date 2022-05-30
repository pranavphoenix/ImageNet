import cv2
from PIL import Image as im
import glob, os

import torch, torchvision
import torchvision.transforms.functional as F
from torchvision.io import read_image, ImageReadMode

subfolders = [ f.path for f in os.scandir('/home/Drive3/pranavjp/train/') if f.is_dir()]

# subfolders.remove('/home/Drive3/pranavjp/train/n02105855') 
# subfolders.remove('/home/Drive3/pranavjp/train/n03062245')
# subfolders.remove('/home/Drive3/pranavjp/train/n04371774')
# subfolders.remove('/home/Drive3/pranavjp/train/n03018349')
# subfolders.remove('/home/Drive3/pranavjp/train/n04336792') 

destination = '/home/Drive3/pranavjp/224/'

# subfolders = ['/home/Drive3/pranavjp/train/n02105855']

for folder in subfolders:

	os.mkdir(os.path.join(destination, folder.split('/')[-1]))

	filenames = glob.glob(os.path.join(folder, '*.JPEG'))

	# filenames.remove('/home/Drive3/pranavjp/train/n02105855/n02105855_2933.JPEG')

	for filename in filenames:

		print(filename)

		
		img = read_image(filename)

		dim = (224,224)

		
		resized = F.resize(img, dim)

		trans = torchvision.transforms.ToPILImage()

		resized = trans(resized)
		

		filename1 = filename.split('/')[-1]

		folder1 = filename.split('/')[-2]

		final = os.path.join(destination,folder1 + '/' + filename1)

		
		resized.save(final)

		print(final)

