# Importing Modules
import cv2
import numpy as np
import face_recognition
import os
import time

flag = ["Celebrities", "Celebrity"]

def null():
    print("nullified")

# Declaring the variable for {x} digit for the file to be saved
token = 0

# Loading Assets
asset_01 = face_recognition.load_image_file(f"assets/{flag[0]}/asset_01.jpg")
asset_02 = face_recognition.load_image_file(f"assets/{flag[0]}/asset_02.jpg")
asset_03 = face_recognition.load_image_file(f"assets/{flag[0]}/asset_03.jpg")
asset_04 = face_recognition.load_image_file(f"assets/{flag[0]}/asset_04.jpg")
asset_05 = face_recognition.load_image_file(f"assets/{flag[0]}/asset_05.jpg")
asset_06 = face_recognition.load_image_file(f"assets/{flag[0]}/asset_06.jpg")
asset_07 = face_recognition.load_image_file(f"assets/{flag[0]}/asset_07.jpg")
asset_08 = face_recognition.load_image_file(f"assets/{flag[0]}/asset_08.jpg")
asset_09 = face_recognition.load_image_file(f"assets/{flag[0]}/asset_09.jpg")
asset_10 = face_recognition.load_image_file(f"assets/{flag[0]}/asset_10.jpg")

# Pushing the assets into a list
mainAssets = [asset_01, asset_02, asset_03, asset_04, asset_05, asset_06, asset_07, asset_08, asset_09, asset_10]
coloredAssets = []

# Map through the list to make every image RGB
for x in range(len(mainAssets)):
    mainAssets[x] = cv2.cvtColor(mainAssets[x], cv2.COLOR_BGR2RGB)
    assetObj = {
        "img": mainAssets[x],
        "faceLocation": face_recognition.face_locations(mainAssets[x])[0]
    }
    coloredAssets.append(assetObj)


cap = cv2.VideoCapture(0)

# Creating the camera loop
while True:
    # Capture frame-by-frame
    ret, camera = cap.read()

    # Display the resulting frame
    cv2.putText(camera, "Press Space To Capture", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(camera, "Press Esc To Exit", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow('Camera', camera)
    k = cv2.waitKey(1)

    # If spacebar is pressed 👇
    if k%256 == 32:
        try:
            imageName = f"pymix-filter-{token}.png"
            cv2.imwrite(imageName, camera)
            cv2.putText(camera, "Loading...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            scores = []

            # Code to compare faces
            for x in range(len(coloredAssets)):  # Map through every asset to compare with snapshot
                snapshot = cv2.imread("pymix-filter-0.png")
                # -------------------------------------------------------------------------
                encodedSnapshot = face_recognition.face_encodings(snapshot)[0]
                encodedAsset = face_recognition.face_encodings(coloredAssets[x]["img"])
                # -------------------------------------------------------------------------
                results = face_recognition.compare_faces(encodedSnapshot, encodedAsset)
                faceDistance = face_recognition.face_distance(encodedSnapshot, encodedAsset)
                # print(results, faceDistance)

                # Rounding the face distances
                roundedFaceDistance = round(faceDistance[0], 3)
                # -------------------------------------------------------------------------
                scores.append(roundedFaceDistance)
                print("...")

            lowestScore = min(scores)
            lowestScoreIndex = scores.index(lowestScore)
            assetMatch = coloredAssets[lowestScoreIndex]["img"]
            cv2.putText(assetMatch, f"We Found Your {flag[1]} Look Alike!", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow("Match", assetMatch)
            cv2.waitKey(0)

        except Exception as e:
            cv2.putText(camera, "Please Try And Show Your Entire Face Properly", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    elif k%256 == 27:
        os.remove("pymix-filter-0.png")
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
