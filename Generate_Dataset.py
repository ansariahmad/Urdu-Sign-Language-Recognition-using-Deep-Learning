import cv2
import os

# Create directories for each sign
num_of_signs = 36
base_dir = 'train'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)
for i in range(1, num_of_signs+1):
    sign_dir = os.path.join(base_dir, f'sign_{i}')
    if not os.path.exists(sign_dir):
        os.makedirs(sign_dir)

# Define the ROI box
roi = (100, 100, 450, 450)

# Initialize camera
cap = cv2.VideoCapture(1)

# Set mouse callback function
cv2.namedWindow('frame')

# Initialize variables
drawing = False
roi_selected = True
img_count = 0
sign_count = 1

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame is captured
    if not ret:
        break

    # Draw ROI if selected
    if roi_selected:
        cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)

    # Display the resulting frame with counter
    cv2.putText(frame, f'Sign {sign_count}, Captured: {img_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('frame', frame)

    # Start capturing pictures of ROI when key 'c' is pressed
    if cv2.waitKey(1) & 0xFF == ord('c') and roi_selected:
        while (True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Check if frame is captured
            if not ret:
                break

            # Crop image to ROI
            img = frame[roi[1]:roi[3], roi[0]:roi[2]]

            # Save image to directory
            img_path = os.path.join(base_dir, f'sign_{sign_count}', f'sign_{sign_count}_{img_count}.jpg')
            cv2.imwrite(img_path, img)
            img_count += 1

            # Display the resulting frame with ROI and counter
            cv2.putText(frame, f'Sign {sign_count}, Captured: {img_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('frame', frame)
            cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)

            # Stop capturing pictures when key 's' is pressed
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break

            # If 1000 pictures captured, display message and prompt user to capture pictures for the next sign
            if img_count == 1000:
                print(f'Sign {sign_count} captured!')
                img_count = 0
                sign_count += 1
                if sign_count > num_of_signs:
                    break
                else:
                    key = input(f'Press any key to start capturing pictures for sign {sign_count}...')
                    if key == "q":
                        cap.release()
                        cv2.destroyAllWindows()
                    else:
                        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
