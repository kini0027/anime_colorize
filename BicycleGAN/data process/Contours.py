import cv2
 
img = cv2.imread('14.png')
gray = cv2.cvtColor ( img , cv2.COLOR_BGR2GRAY )
cv2.imwrite("gray.jpg",gray)
ret , binary = cv2.threshold ( gray , 250 , 255 , cv2.THRESH_BINARY )
cv2.imwrite("binary.jpg",binary)
imggg=cv2.Canny(binary, 256, 256) 
cv2.imwrite("contours.jpg",imggg)
cv2.waitKey(0)

