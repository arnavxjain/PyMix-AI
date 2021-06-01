# Importing Modules
import cv2
import numpy as np
import face_recognition


def null():
    print("nullified")


# Loading Footballers
asset_01 = face_recognition.load_image_file("assets/Footballers/asset_01.jpg")
asset_02 = face_recognition.load_image_file("assets/Footballers/asset_02.jpg")
asset_03 = face_recognition.load_image_file("assets/Footballers/asset_03.jpg")
asset_04 = face_recognition.load_image_file("assets/Footballers/asset_04.jpg")
asset_05 = face_recognition.load_image_file("assets/Footballers/asset_05.jpg")
asset_06 = face_recognition.load_image_file("assets/Footballers/asset_06.jpg")
asset_07 = face_recognition.load_image_file("assets/Footballers/asset_07.jpg")
asset_08 = face_recognition.load_image_file("assets/Footballers/asset_08.jpg")
asset_09 = face_recognition.load_image_file("assets/Footballers/asset_09.jpg")
asset_10 = face_recognition.load_image_file("assets/Footballers/asset_10.jpg")

# Pushing the football assets into a list
footballAssets = [asset_01, asset_02, asset_03, asset_04, asset_05, asset_06, asset_07, asset_08, asset_09, asset_10]

encodedFootballAssets = []

# Map through the list to make every image RGB
for x in range(len(footballAssets)):
    footballAssets[x] = cv2.cvtColor(footballAssets[x], cv2.COLOR_BGR2RGB)
    footbalObj = {
        "img": footballAssets[x],
        "faceLocation": face_recognition.face_locations(footballAssets[x])[0]
    }
    encodedFootballAssets.append(footbalObj)

# for x in range(len(footballAssets)):
#     cv2.imshow("Frame", footballAssets[x])
#     cv2.waitKey(0)

# Accessing the camera
cap = cv2.VideoCapture(0)

# Creating the camera loop
while True:
    # Capture frame-by-frame
    ret, camera = cap.read()

    # Display the resulting frame
    cv2.createButton("Back", null, None, cv2.QT_PUSH_BUTTON, 1)
    cv2.imshow('Frame', camera)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
