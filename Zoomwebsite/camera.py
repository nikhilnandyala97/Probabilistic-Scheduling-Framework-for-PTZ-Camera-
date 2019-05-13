import cv2
import os;
import numpy as np;
import requests
from PIL import Image
from io import BytesIO

class VideoCamera(object):
	def get_frame(self):

		response = requests.get("http://admin:*1Password@10.42.0.199/web/tmpfs/auto.jpg")
		img = Image.open(BytesIO(response.content))
		frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
		#resp = urllib.request.urlopen("http://admin:*1Password@10.42.0.199/web/tmpfs/snap.jpg")
		#image = np.asarray(bytearray(resp.read()), dtype="uint8")
		#image = cv2.imdecode(image, cv2.IMREAD_COLOR)
		ret, jpeg = cv2.imencode('.jpg', frame)
		return jpeg.tobytes()
	
