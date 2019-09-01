from face_detector import *
import argparse
import cv2

def str2bool(v):
	#susendberg's function
	return v.lower() in ("yes", "true", "t", "1")

def get_parser():
	parser = argparse.ArgumentParser(description='OpenCV implementation of face detection from images',
		epilog='For questions and bug reports, contact Jacky Ko <jkmailbox1991@gmail.com>')

	# register type keyword to registries
	parser.register('type','bool',str2bool)

	# add arguments
	parser.add_argument(
		'-v', '--verbose',
		dest='verbose',
		help='Show verbose output',
		default=False,
		action='store_true')
	parser.add_argument(
		'-e','--eye', 
		dest='eye', 
		help='Detect eyes (default = true)',
		default=True,
		action='store_true'
		)
	parser.add_argument(
		'--face_xml',
		dest='face_xml',
		help='File path to face detection xml config file (default = "haarcascade_frontalface_default.xml")',
		default='./haarcascade_frontalface_default.xml',
		type=str)
	parser.add_argument(
		'--eye_xml',
		dest='eye_xml',
		help='File path to eye detection xml config file (default = "haarcascade_eye.xml")',
		default='./haarcascade_eye.xml',
		type=str)
	parser.add_argument(
		'-i','--image',
		dest='image',
		help='File path to image',
		default='./images/1.jpg',
		type=str)
	parser.add_argument(
		'-s','--show',
		dest='show',
		help='Option to show detection result',
		default=True,
		type=bool)

	args = parser.parse_args()

	# print arguments if verbose
	if args.verbose:
		args_dict = vars(args)
		for key in sorted(args_dict):
			print("{} = {}".format(str(key), str(args_dict[key])))

	return args

def main(args):
	fd = FaceDetector()
	fd.face_xml = args.face_xml
	fd.eye_xml = args.eye_xml
	fd.show_result = args.show
	fd.detect_eyes = args.eye

	# read the image
	img = cv2.imread(args.image)

	fd.image = img
	result = fd.classify(img)

if __name__=="__main__":
	args = get_parser()
	main(args)