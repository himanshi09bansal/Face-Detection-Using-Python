import cv2

#  Loading the image to be tested
img = cv2.imread("family.PNG", 1) 

# Haar cascade files
# Loading the classifier for frontal face
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


# Converting to grayscale as opencv expects detector takes in input gray scale images
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Face detection
faces = face_cascade.detectMultiScale(gray, 1.05, 5)

# Let us print the no. of faces found
print('Faces found: ', len(faces))


# Draw rectangles around the detected faces
for x,y,w,h in faces:
    img = cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 3)


# Display the image
cv2.imshow("Human", img)
#print(img)
cv2.waitKey(1000)
cv2.destroyAllWindows()
