import cv2
import numpy as np
import os
import utils
from time import sleep

#################################################################
img_folder = "imgs"
widthImg = 800
heightImg = 600
questions = 10
choices = 5
ans = [0, 0, 2, 3, 4, 1, 2, 2, 3, 0]
template_ans = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
webcamFeed = True
camNum = 0
readed_sheet = False
count_sheet = 0
last_sheet = []
student_scores = []
thresh = 114  # last was 122
#################################################################

cap = cv2.VideoCapture(camNum)
cap.set(10, 150)

while readed_sheet == False:
    if webcamFeed:
        sucess, img = cap.read()
    else:
        img = cv2.imread(os.path.join(img_folder, "4.jpg"))
    # height, width, channels = img.shape
    # print(height, width, channels)
    img = cv2.resize(img, (widthImg, heightImg))
    img = cv2.rotate(img, cv2.ROTATE_180)

    # Preprocessing
    imgContours = img.copy()
    imgBiggestContour = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 10, 50)

    try:
        # Finding ALL CONTOURS
        contours, hierarchy = cv2.findContours(
            imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 1)

        # FIND RECTANGLES
        rectCon = utils.rectCountour(contours)
        biggestContour = utils.getCornerPoints(rectCon[0])
        # print(len(rectCon))
        # cv2.drawContours(imgContours, rectCon[0], -1, (0, 255, 255), 10)  # amarelo
        # cv2.drawContours(imgContours, rectCon[1], -1, (255, 0, 0), 10)  # azul
        # cv2.drawContours(imgContours, rectCon[2], -1, (34, 139, 34), 10)  # verde
        # cv2.drawContours(imgContours, rectCon[3], -1, (0, 255, 255), 10)
        # print(utils.getCornerPoints(rectCon[0]))

        if biggestContour.size != 0:
            cv2.drawContours(imgBiggestContour, biggestContour, -1, (0, 255, 255), 5)

            biggestContour = utils.reorder(biggestContour)
            # print(biggestContour.shape)

            pt1 = np.float32(biggestContour)
            pt2 = np.float32(
                [[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]]
            )
            matrix = cv2.getPerspectiveTransform(pt1, pt2)
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

            # Apply Threshold
            imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.threshold(imgWarpGray, thresh, 255, cv2.THRESH_BINARY_INV)[
                1
            ]

            # utils.splitBoxes(imgThresh)
            boxes = utils.splitBoxes2(imgThresh)
            # cv2.imshow("test", boxes[2])

            # Getting Pixel Values of each box
            myPixelVal = np.zeros((questions, choices))
            countC = 0
            countR = 0

            for image in boxes:
                totalPixels = cv2.countNonZero(image)
                myPixelVal[countR][countC] = totalPixels
                countC += 1
                if countC == choices:
                    countR += 1
                    countC = 0
            # print(myPixelVal)

            # Finding index values of the markings
            myIndex = []
            for x in range(0, questions):
                arr = myPixelVal[x]
                # print(arr)

                array_sorted = []
                for item in arr:
                    array_sorted.append(item)
                array_sorted.sort(reverse=True)
                # print(array_sorted)

                if (array_sorted[0] - array_sorted[1]) < 2000 or array_sorted[0] < 300:
                    myIndex.append(-1)
                else:
                    myIndexVal = np.where(arr == np.amax(arr))
                    myIndex.append(myIndexVal[0][0])
            print(myIndex)
            print(ans)

            # grading
            grading = []
            for x in range(0, questions):
                if ans[x] == myIndex[x]:
                    grading.append(1)
                else:
                    grading.append(0)
            print(grading)
            score = sum(grading)
            print(score)
            readed_sheet = True

        imgBlank = np.zeros_like(img)
        imageArray = (
            [img, imgGray, imgBlur, imgCanny],
            [imgContours, imgBiggestContour, imgWarpColored, imgThresh],
        )
    except:
        imgBlank = np.zeros_like(img)
        imageArray = (
            [img, imgGray, imgBlur, imgCanny],
            [imgBlank, imgBlank, imgBlank, imgBlank],
        )

    imgStacked = utils.stackImages(imageArray, 0.4)

    cv2.imshow("Stacked Images", imgStacked)

    if readed_sheet:
        # print("\a")
        if last_sheet == myIndex and myIndex != template_ans:
            count_sheet += 1
            print(count_sheet)
        else:
            last_sheet = myIndex
            count_sheet = 0

        if count_sheet == 30:
            print("\a")
            student_scores.append(score)
            sleep(3)
            count_sheet = 0

        readed_sheet = False

    k = cv2.waitKey(60)
    if k == 27:
        print(student_scores)
        break
    elif k == ord("a"):
        print("\a")
