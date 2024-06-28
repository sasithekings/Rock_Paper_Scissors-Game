import cv2
import numpy as np
import tensorflow as tf
import random
import time
import pyttsx3
import threading
import os

# Load the pre-trained hand gesture recognition model
model = tf.keras.models.load_model('rock_paper_scissors_model.h5')

# Constants for the bounding box
BOX_START_X = 100
BOX_START_Y = 100
BOX_SIZE = 300

# Ensure the directory for saving captured images exists
os.makedirs('captured_images', exist_ok=True)

# Function to detect hand gesture within the bounding box and save the captured image
def detect_hand(frame):
    # Crop the frame to the bounding box
    cropped_frame = frame[BOX_START_Y:BOX_START_Y+BOX_SIZE, BOX_START_X:BOX_START_X+BOX_SIZE]

    # Convert image to tf.Tensor
    image_tensor = tf.convert_to_tensor(cropped_frame, dtype=tf.uint8)

    # Convert image to float32 and normalize pixel values
    image_tensor = tf.image.convert_image_dtype(image_tensor, tf.float32)

    # Resize the image to the model's expected input shape
    image_tensor = tf.image.resize(image_tensor, [224, 224])

    # Expand dimensions to match the model's expected input shape
    image_tensor = tf.expand_dims(image_tensor, axis=0)

    # Make predictions using the pre-trained model
    predictions = model.predict(image_tensor)
    
    # Convert prediction to hand gesture label
    gestures = ['rock', 'paper', 'scissors','none']
    predicted_gesture_index = np.argmax(predictions)
    detected_gesture = gestures[predicted_gesture_index]

    return detected_gesture

# Function to determine winner
def determine_winner(user_choice, computer_choice):
    if user_choice == computer_choice:
        return 'Tie'
    elif (user_choice == 'rock' and computer_choice == 'scissors') or \
         (user_choice == 'paper' and computer_choice == 'rock') or \
         (user_choice == 'scissors' and computer_choice == 'paper'):
        return 'User'
    else:
        return 'Computer'

# Function to announce countdown
def announce_countdown():
    engine = pyttsx3.init()
    for i in range(3, 0, -1):
        engine.say(str(i))
        engine.runAndWait()
        time.sleep(1)

# Function to play the game
def play_game():
    global game_started, user_choice, computer_choice, winner, frame

    announce_countdown()
    user_choice = detect_hand(frame)
    computer_choice = random.choice(['rock', 'paper', 'scissors'])
    winner = determine_winner(user_choice, computer_choice)
    game_started = False

# Main function
def main():
    global game_started, user_choice, computer_choice, winner, frame
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to capture video.")
        return

    game_started = False
    user_choice = None
    computer_choice = None
    winner = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw the bounding box on the frame
        cv2.rectangle(frame, (BOX_START_X, BOX_START_Y), (BOX_START_X+BOX_SIZE, BOX_START_Y+BOX_SIZE), (0, 255, 0), 2)

        if not game_started:
            if user_choice and computer_choice and winner:
                cv2.putText(frame, f"Your choice: {user_choice}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Computer's choice: {computer_choice}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Winner: {winner}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Rock Paper Scissors', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and not game_started:
            game_started = True
            threading.Thread(target=play_game).start()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
