import cv2
import sys

# 사진과 얼굴의 어떤 부분을 인식할까요?
imagePath = "C:/Users/msaca/assignment/opencvWithPython/Image.jpg" # 이미지 파일의 경로
cascPath = "haarcascade_frontalface_default.xml"

# haar cascade를 결정합니다
faceCascade = cv2.CascadeClassifier(cascPath)


title1, title2, title3 = 'image', 'Face_rectangle', 'Face_circle'
cv2.namedWindow(title1, cv2.WINDOW_NORMAL)
cv2.namedWindow(title2, cv2.WINDOW_NORMAL)
cv2.namedWindow(title3, cv2.WINDOW_NORMAL)

# 이미지를 읽어와 흑백사진으로 전환합니다
image = cv2.imread(imagePath)
image1 = image.copy()
image2 = image.copy()

if image is None:
    raise Exception(" 이미지 파일을 읽을 수 없습니다. 파일의 경로를 정확하게 입력해주세요. ")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# 흑백사진 상태에서 얼굴을 탐지합니다
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=10,
    minSize=(30, 30)
    # flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

# 몇개의 얼굴을 찾았는지 말합니다
print("Found {0} faces!".format(len(faces)))

# 얼굴에 초록색 사각형 그려주기 -> RGB (0, 255, 0)
for (x, y, w, h) in faces:
    cv2.rectangle(image1, (x, y), (x+w, y+h), (0, 255, 0), 2)

# 얼굴에 빨간색 원 그려주기 -> RGB (0, 0, 255)    
for (x, y, w, h) in faces:
    cv2.circle(image2,(x+int(w/2),y+int(h/2)), int(w/1.5), (0,0,255), 2)

# 얼굴에 사각형을 친 사진을 출력합니다
cv2.imwrite("Faces_rectangle.png", image1)
cv2.imwrite("Faces_circle.png", image2)

cv2.imshow(title1, image)
cv2.imshow(title2, image1)
cv2.imshow(title3, image2)
cv2.waitKey(0)