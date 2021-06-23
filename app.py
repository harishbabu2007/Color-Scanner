from flask import Flask, Response, render_template
import cv2
import numpy as np

app = Flask(__name__)

video = cv2.VideoCapture(0)


def camera(video):
        while True:
            success, image = video.read()

            hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            #blue detection
            low_blue = np.array([94, 80, 2], np.uint8)
            high_blue = np.array([126, 255, 255], np.uint8)
            blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
            
            blue_contours, b_ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            
            for b_contour in blue_contours:
                area_b = cv2.contourArea(b_contour)

                if area_b > 20000:
                    [x,y,w,h] = cv2.boundingRect(b_contour)
                    rect = cv2.rectangle(image, (x,y), (x+w,y+h), (255, 0, 0), 3)
                    cv2.putText(rect, 'blue detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # red detection
            low_red = np.array([0,50,50], np.uint8)
            high_red = np.array([11,255,255], np.uint8)
            red_mask = cv2.inRange(hsv_frame, low_red, high_red)
            
            red_contours, r_ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            for r_contour in red_contours:
                area_r = cv2.contourArea(r_contour)

                if area_r > 7000:
                    [x,y,w,h] = cv2.boundingRect(r_contour)
                    rect = cv2.rectangle(image, (x,y), (x+w,y+h), (0, 0, 255), 3)
                    cv2.putText(rect, 'red detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # green detection
            low_green = np.array([25, 52, 72], np.uint8)
            high_green = np.array([102, 255, 255], np.uint8)
            green_mask = cv2.inRange(hsv_frame, low_green, high_green)

            green_contours, g_ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            for g_contour in green_contours:
                area_g = cv2.contourArea(g_contour)

                if area_g > 9000:
                    [x,y,w,h] = cv2.boundingRect(g_contour)
                    rect = cv2.rectangle(image, (x,y), (x+w,y+h), (0, 255, 0), 3)
                    cv2.putText(rect, 'green detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # yellow detection
            low_yellow = np.array([22, 93, 0], np.uint8)
            high_yellow = np.array([45, 255, 255], np.uint8)
            yellow_mask = cv2.inRange(hsv_frame, low_yellow, high_yellow)

            yellow_contours, y_ = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            for y_contour in yellow_contours:
                area_y = cv2.contourArea(y_contour)

                if area_y > 9000:
                    [x,y,w,h] = cv2.boundingRect(y_contour)
                    rect = cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,255), 3)
                    cv2.putText(rect, 'yellow detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

            
            ret, jpeg = cv2.imencode('.jpg', image)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/video_feed')
def video_feed():
    global video
    return Response(camera(video),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
    