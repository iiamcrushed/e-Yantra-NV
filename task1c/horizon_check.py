import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

cap = cv.VideoCapture('public/clip_01.mp4')

ret, frame = cap.read()
cap.release()       

if ret:
    hls = cv.cvtColor(frame, cv.COLOR_BGR2HLS)
    
    lower_white = np.array([0, 200, 0], dtype=np.uint8)
    upper_white = np.array([179, 255, 255], dtype=np.uint8)
    white_mask = cv.inRange(hls, lower_white, upper_white)

    lower_yellow = np.array([15, 120, 100], dtype=np.uint8)
    upper_yellow = np.array([35, 255, 255], dtype=np.uint8)
    yellow_mask = cv.inRange(hls, lower_yellow, upper_yellow)

    combined_color_mask = cv.bitwise_or(white_mask, yellow_mask)
# Based on visually estimating your frame

    vertices = np.array([[
        [24, 420],   # Your bottom-left
        [300, 130],  # Your top-left
        [450, 130],  # Your top-right
        [640, 200],  # The extra anchor point to catch the right yellow line
        [640, 420]   # Your bottom-right
    ]], dtype=np.int32)

  
    roi_mask = np.zeros_like(combined_color_mask)
    cv.fillPoly(roi_mask, vertices, 255)

    cropped_lanes = cv.bitwise_and(combined_color_mask, roi_mask)

    src = np.float32([
        [260, 150],  # Top-left (anchored to the yellow line)
        [380, 150],  # Top-right (anchored to the dashed line)
        [530, 420],  # Bottom-right (anchored to the dashed line)
        [24, 420]    # Bottom-left (anchored to the yellow line)
    ])


    dst = np.float32([
        [150, 0],      
        [490, 0],    
        [490, 480],  
        [150, 480]     
    ])
    
    M = cv.getPerspectiveTransform(src, dst)
    Minv=cv.getPerspectiveTransform(dst, src)
    warped = cv.warpPerspective(cropped_lanes, M, (640,480))
    
    bottom_half = warped[240:, :]
    
    histogram = np.sum(bottom_half, axis = 0)
    midpoint = int(histogram.shape[0]/2)
    
    left_x_warped=np.argmax(histogram[:midpoint])
    right_x_warped = np.argmax(histogram[:midpoint]) + midpoint
    
    # --- Visual Testing ---
    # Convert the black-and-white warped image to color so we can draw colored lines on it
    warped_color = cv.cvtColor(warped, cv.COLOR_GRAY2RGB)

    # Draw a red line down the detected left lane center
    cv.line(warped_color, (left_x_warped, 0), (left_x_warped, 480), (255, 0, 0), 3)
    
    # Draw a blue line down the detected right lane center
    cv.line(warped_color, (right_x_warped, 0), (right_x_warped, 480), (0, 0, 255), 3)

    center_x_warped = (left_x_warped + right_x_warped) / 2
    warped_point = np.array([[[center_x_warped, 480]]], dtype=np.float32)
    original_point = cv.perspectiveTransform(warped_point, Minv)
    center_x_original = original_point[0][0][0]

    left_half_yellow=yellow_mask[:, :320]

    yellow_pixel_count=cv.countNonZero(left_half_yellow)
    if yellow_pixel_count>100:
        lane = 1

    # Display the result
    plt.imshow(warped_color)
    plt.title(f"Left Peak: {left_x_warped} | Right Peak: {right_x_warped}")
    plt.show()

cap.release()