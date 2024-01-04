import cv2
import numpy as np

# 定义红、绿、蓝色的HSV范围
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])

lower_green = np.array([40, 50, 50])
upper_green = np.array([90, 255, 255])

lower_blue = np.array([100, 50, 50])
upper_blue = np.array([140, 255, 255])


def color_detection(frame):
    # 转换图像为HSV色彩空间
    gs_frame = cv2.GaussianBlur(frame, (5, 5), 0)  # 高斯模糊
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)   #转换为HSV图像
    erode_hsv = cv2.erode(hsv, None, iterations=2)  #腐蚀
    inRange_red =cv2.inRange(erode_hsv,lower_red,upper_red)
    inRange_green = cv2.inRange(erode_hsv, lower_green, upper_green)
    inRange_blue = cv2.inRange(erode_hsv, lower_blue, upper_blue)
    # 绘制矩形边框
    if np.any(inRange_red):
        cnts_red = cv2.findContours(inRange_red.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        c_red = max(cnts_red, key=cv2.contourArea)
        rect_red = cv2.minAreaRect(c_red)
        box_red = cv2.boxPoints(rect_red)
        cv2.drawContours(frame, [np.int0(box_red)], -1, (0, 255, 255), 2)
        x, y, w, h = cv2.boundingRect(c_red)
        cv2.putText(frame, 'Red', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    if np.any(inRange_green):
        cnts_green = cv2.findContours(inRange_green.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        c_green = max(cnts_green, key=cv2.contourArea)
        rect_green = cv2.minAreaRect(c_green)
        box_green = cv2.boxPoints(rect_green)
        cv2.drawContours(frame, [np.int0(box_green)], -1, (0, 255, 255), 2)
        x, y, w, h = cv2.boundingRect(c_green)
        cv2.putText(frame, 'Green', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    if np.any(inRange_blue):
        cnts_blue = cv2.findContours(inRange_blue.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        c_blue = max(cnts_blue, key=cv2.contourArea)
        rect_blue = cv2.minAreaRect(c_blue)
        box_blue = cv2.boxPoints(rect_blue)
        cv2.drawContours(frame, [np.int0(box_blue)], -1, (0, 255, 255), 2)
        x, y, w, h = cv2.boundingRect(c_blue)
        cv2.putText(frame, 'Blue', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return frame


# 打开摄像头
cap = cv2.VideoCapture(0)
# cv2.namedWindow('camera', cv2.WINDOW_AUTOSIZE)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 调用颜色识别函数
    result_frame = color_detection(frame)

    # 显示结果
    cv2.imshow('Color Detection', result_frame)

    # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()
