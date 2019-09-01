import cv2

class FaceDetector(object):
	def __init__(self):
		self.face_xml = 'haarcascade_frontalface_default.xml'
		self.eye_xml = 'haarcascade_eye.xml'
		self.detect_eyes = False
		self.faces = None
		self.eyes = None
		self.show_result = False

	def classify(self,image):
		face_cascade = cv2.CascadeClassifier(self.face_xml)
		eye_cascade = cv2.CascadeClassifier(self.eye_xml)

		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# detect faces
		self.faces = face_cascade.detectMultiScale(image, 1.3, 5)

		self.eyes = []
		for (x, y, w, h) in self.faces:
			cv2.rectangle(image, (x,y), (x+w,y+h),(255,0,0),2)
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = image[y:y+h, x:x+w]
			if self.detect_eyes:
				self.eyes.append(eye_cascade.detectMultiScale(roi_gray))
				for (ex, ey, ew, eh) in self.eyes[-1]:
					cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh),(0,255,0),2)

		result = {'faces':self.faces,'eyes':self.eyes}

		if self.show_result:
			cv2.imshow('face detection result', image)
			cv2.waitKey(0)
			cv2.destroyAllWindows()

		
		return result