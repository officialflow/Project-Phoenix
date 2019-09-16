import cv2
fc = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img = cv2.imread("photo.jpg")

gimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = fc.detectMultiScale(gimg,
scaleFactor = 1.5,
minNeighbors = 5)

for x,y,w,h in faces:
	img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),8)
print(faces)

resize = cv2.resize(img,(1000,720))

cv2.imshow("gray",resize)

cv2.waitKey(0)

cv2.destroyAllWindows()

