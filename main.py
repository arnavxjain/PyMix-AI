# Importing Modules
import cv2
import numpy as np
import face_recognition


def null():
    print("nullified")

# Declaring the variable for {x} digit for the file to be saved
token = 0

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
coloredFootballAssets = []

# Map through the list to make every image RGB
for x in range(len(footballAssets)):
    footballAssets[x] = cv2.cvtColor(footballAssets[x], cv2.COLOR_BGR2RGB)
    footbalObj = {
        "img": footballAssets[x],
        "faceLocation": face_recognition.face_locations(footballAssets[x])[0]
    }
    coloredFootballAssets.append(footbalObj)


cap = cv2.VideoCapture(0)

# Creating the camera loop
while True:
    # Capture frame-by-frame
    ret, camera = cap.read()

    # Display the resulting frame
    cv2.putText(camera, "Press Space To Capture", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(camera, "Press Esc To Exit", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    k = cv2.waitKey(1)

    cv2.imshow('Camera', camera)

    # If spacebar is pressed 👇
    if k%256 == 32:
        imageName = f"pymix-filter-{token}.png"
        cv2.imwrite(imageName, camera)

        scores = []
        # Code to compare faces
        for x in range(len(coloredFootballAssets)): # Map through every asset to compare with snapshot
            snapshot = cv2.imread("pymix-filter-0.png")
            # -------------------------------------------------------------------------
            encodedSnapshot = face_recognition.face_encodings(snapshot)[0]
            encodedAsset = face_recognition.face_encodings(coloredFootballAssets[x]["img"])
            # -------------------------------------------------------------------------
            results = face_recognition.compare_faces(encodedSnapshot, encodedAsset)
            faceDistance = face_recognition.face_distance(encodedSnapshot, encodedAsset)
            # print(results, faceDistance)
            roundedFaceDistance = round(faceDistance[0], 3)
            # -------------------------------------------------------------------------
            scores.append(roundedFaceDistance)

        print(scores)
        lowestScore = min(scores)
        lowestScoreIndex = scores.index(lowestScore)
        print(lowestScoreIndex)
        cv2.imshow("Match", coloredFootballAssets[lowestScoreIndex]["img"])
        cv2.waitKey(0)

    elif k%256 == 27:
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
