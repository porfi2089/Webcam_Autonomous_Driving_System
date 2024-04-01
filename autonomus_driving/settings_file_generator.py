import tkinter as tk
from tkinter import ttk
import cv2 as cv
from cv2 import aruco
import json
import numpy as np
from PIL import Image, ImageTk
import os

settings_path = r"autonomus_driving\settings.json"

def image_resize(image, width = None, height = None, inter = cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


def load_unwrap_data(filepath='unwrap_data.json'):
    with open(filepath, 'r') as f:
        unwrapData = json.loads(f.read())

    global unwrap_cent
    unwrap_cent = np.array(unwrapData['centers'])

load_unwrap_data(r'autonomus_driving\unwrap_data.json')
# Create GUI
root = tk.Tk()
root.title("Settings Editor")
root.geometry("800x600")

r_presed = False
def on_r_keypress(event):
    global r_presed
    r_presed = True

root.bind('<r>', on_r_keypress)

w_presed = False
def on_w_keypress(event):
    global w_presed
    w_presed = True

root.bind('<w>', on_w_keypress)

# JSON parameters labels and entries
labels = ["Line Reductions:", "Cam Index:", "Threshold 1:", "Threshold 2:", "Frame Width:", "Frame Height:",
          "Adaptive Setting 1:", "Adaptive Setting 2:", "Serial Baudrate:",
          "Horizontal Line Threshold:", "line_width"]

menu_frame = ttk.Frame(root)
image_frame = ttk.Frame(root)
image_frame.place(relx=0.4, rely=0, relwidth=0.6, relheight=1)
menu_frame.place(relx=0, rely=0, relwidth=0.4, relheight=1)
menu_frame.grid_columnconfigure((0,1), weight=1)
menu_frame.grid_rowconfigure(list(range(len(labels)+1)), weight=1)
entries = {}
broken_file = False
try:
    with open(settings_path, "r") as file:
        data = json.load(file)
        for label in labels:
            if label not in data:
                raise Exception
except:
    broken_file = True
if not os.path.exists(settings_path) or broken_file:
    with open(settings_path, "w") as file:
        data = {
            "line_reductions": 1,
            "cam_index": 0,
            "thresholds": [100, 255],
            "frame_size": [640, 480],
            "adaptive_settings": [11, 11],
            "Serial_settings": [9600, "COM1"],
            "horizontal_line_threshold": 240,
            "line_width": 80
        }
        json.dump(data, file, indent=4)
    
line_reductions_label = tk.Label(menu_frame, text=labels[0])
line_reductions_label.grid(row=0, column=0, sticky="e")
line_reductions_entry = tk.Entry(menu_frame)
line_reductions_entry.grid(row=0, column=1, padx=5, pady=5)
entries[labels[0]] = line_reductions_entry

cam_index_label = tk.Label(menu_frame, text=labels[1])
cam_index_label.grid(row=1, column=0, sticky="e")
cam_index_entry = tk.Entry(menu_frame)
cam_index_entry.grid(row=1, column=1, padx=5, pady=5)
entries[labels[1]] = cam_index_entry

threshold1_label = tk.Label(menu_frame, text=labels[2])
threshold1_label.grid(row=2, column=0, sticky="e")
threshold1_entry = tk.Entry(menu_frame)
threshold1_entry.grid(row=2, column=1, padx=5, pady=5)
entries[labels[2]] = threshold1_entry

threshold2_label = tk.Label(menu_frame, text=labels[3])
threshold2_label.grid(row=3, column=0, sticky="e")
threshold2_entry = tk.Entry(menu_frame)
threshold2_entry.grid(row=3, column=1, padx=5, pady=5)
entries[labels[3]] = threshold2_entry

frame_width_label = tk.Label(menu_frame, text=labels[4])
frame_width_label.grid(row=4, column=0, sticky="e")
frame_width_entry = tk.Entry(menu_frame)
frame_width_entry.grid(row=4, column=1, padx=5, pady=5)
entries[labels[4]] = frame_width_entry

frame_height_label = tk.Label(menu_frame, text=labels[5])
frame_height_label.grid(row=5, column=0, sticky="e")
frame_height_entry = tk.Entry(menu_frame)
frame_height_entry.grid(row=5, column=1, padx=5, pady=5)
entries[labels[5]] = frame_height_entry

adaptive_setting1_label = tk.Label(menu_frame, text=labels[6])
adaptive_setting1_label.grid(row=6, column=0, sticky="e")
adaptive_setting1_entry = tk.Entry(menu_frame)
adaptive_setting1_entry.grid(row=6, column=1, padx=5, pady=5)
entries[labels[6]] = adaptive_setting1_entry

adaptive_setting2_label = tk.Label(menu_frame, text=labels[7])
adaptive_setting2_label.grid(row=7, column=0, sticky="e")
adaptive_setting2_entry = tk.Entry(menu_frame)
adaptive_setting2_entry.grid(row=7, column=1, padx=5, pady=5)
entries[labels[7]] = adaptive_setting2_entry

serial_baudrate_label = tk.Label(menu_frame, text=labels[8])
serial_baudrate_label.grid(row=8, column=0, sticky="e")
serial_baudrate_entry = tk.Entry(menu_frame)
serial_baudrate_entry.grid(row=8, column=1, padx=5, pady=5)
entries[labels[8]] = serial_baudrate_entry

serial_port_label = tk.Label(menu_frame, text="serial_port:")
serial_port_label.grid(row=9, column=0, sticky="e")
serial_port_entry = ttk.Combobox(menu_frame, values=["COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", "COM10", "COM11", "COM12", "COM13", "COM14", "COM15"])
serial_port_entry.grid(row=9, column=1, padx=5, pady=5)

horizontal_line_threshold_label = tk.Label(menu_frame, text=labels[9])
horizontal_line_threshold_label.grid(row=10, column=0, sticky="e")
horizontal_line_threshold_entry = tk.Entry(menu_frame)
horizontal_line_threshold_entry.grid(row=10, column=1, padx=5, pady=5)
entries[labels[9]] = horizontal_line_threshold_entry

line_width_label = tk.Label(menu_frame, text=labels[10])
line_width_label.grid(row=11, column=0, sticky="e")
line_width_entry = tk.Entry(menu_frame)
line_width_entry.grid(row=11, column=1, padx=5, pady=5)
entries[labels[10]] = line_width_entry

with open(settings_path, "r") as file:
        data = json.load(file)
        line_reductions_entry.insert(0, data["line_reductions"])
        cam_index_entry.insert(0, data["cam_index"])
        threshold1_entry.insert(0, data["thresholds"][0])
        threshold2_entry.insert(0, data["thresholds"][1])
        frame_width_entry.insert(0, data["frame_size"][0])
        frame_height_entry.insert(0, data["frame_size"][1])
        adaptive_setting1_entry.insert(0, data["adaptive_settings"][0])
        adaptive_setting2_entry.insert(0, data["adaptive_settings"][1])
        serial_baudrate_entry.insert(0, data["Serial_settings"][0])
        serial_port_entry.insert(0, data["Serial_settings"][1])
        horizontal_line_threshold_entry.insert(0, data["horizontal_line_threshold"])
        line_width_entry.insert(0, data["line_width"])

# Function to update JSON file
def update_json():
    incompleteFlag = False
    for entry, label in zip(entries.values(), labels):
        try :
            int(entry.get())
        except:
            entry.delete(0, "end")
            entry.insert(0, "Enter Value")
            entry.config(fg="red")
            incompleteFlag = True
    
    if not incompleteFlag:
        data = {
            "line_reductions": int(line_reductions_entry.get()),
            "cam_index": int(cam_index_entry.get()),
            "thresholds": [int(threshold1_entry.get()), int(threshold2_entry.get())],
            "frame_size": [int(frame_width_entry.get()), int(frame_height_entry.get())],
            "adaptive_settings": [int(adaptive_setting1_entry.get()), int(adaptive_setting2_entry.get())],
            "Serial_settings": [int(serial_baudrate_entry.get()), serial_port_entry.get()],
            "horizontal_line_threshold": int(horizontal_line_threshold_entry.get()),
            "line_width": int(line_width_entry.get())
        }
        with open(settings_path, "w") as file:
            json.dump(data, file, indent=4)

# Update Button
update_button = tk.Button(root, text="Update", command=update_json)
update_button.grid(row=len(labels)+1, columnspan=2, pady=10)

# OpenCV display
cap = cv.VideoCapture(0)
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
cv_panel = tk.Label(image_frame)
cv_panel.pack(fill="both", expand=True)

marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_250)

param_markers = aruco.DetectorParameters()

pts2 = [[0, 0], [0, 480], [640, 0], [640, 480]]
matrix = cv.getPerspectiveTransform(np.float32(unwrap_cent), np.float32(pts2))

def update_frame():
    global r_presed
    global w_presed
    _, frame = cap.read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame = cv.warpPerspective(frame, matrix, (640, 480))
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    marker_corners, marker_IDs, reject = aruco.detectMarkers(
        gray_frame, marker_dict, parameters=param_markers
    )
    centers = []
    marker_IDs_ref = []
    if marker_corners:
        for ids, corners in zip(marker_IDs, marker_corners):
            cv.polylines(
                frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
            )
            corners = corners.reshape(4, 2)
            corners = corners.astype(int)
            top_right = corners[0].ravel()
            top_left = corners[1].ravel()
            bottom_right = corners[2].ravel()
            bottom_left = corners[3].ravel()
            center = [int((top_left[0] + bottom_left[0])/2), int((top_left[1] + bottom_left[1])/2)]
            centers.append(center)
            cv.circle(frame, center, 3, (0, 0, 255), -1)
            cv.putText(
                frame,
                f"id: {ids[0]}",
                top_right,
                cv.FONT_HERSHEY_PLAIN,
                1.3,
                (200, 100, 0),
                2,
                cv.LINE_AA,
            )
            marker_IDs_ref.append(ids[0])
    if marker_IDs is not None:
        if len(marker_IDs) == 1:
            if r_presed:
                print("Marker IDs: ", marker_IDs)
                horizontal_line_threshold_entry.delete(0, "end")
                horizontal_line_threshold_entry.insert(0, centers[0][1])
                r_presed = False
            if w_presed:
                print("Single Marker Detected")
                w_presed = False
        if r_presed and len(marker_IDs) > 1:
            print("Multiple Markers Detected")
            r_presed = False
        if len(marker_IDs) == 2:
            distance = int(np.sqrt((centers[0][0] - centers[1][0])**2 + (centers[0][1] - centers[1][1])**2))
            cv.line(frame, centers[0], centers[1], (0, 255, 0), 2)
            cv.putText(frame, f"Distance: {distance}", (centers[0][0], centers[0][1]-10), cv.FONT_HERSHEY_PLAIN, 1.3, (0, 255, 0), 2, cv.LINE_AA)
            if w_presed:
                line_width_entry.delete(0, "end")
                line_width_entry.insert(0, distance)
                print("Distance between markers: ", distance)
                w_presed = False
    else:
        if r_presed or w_presed:
            print("No Markers Detected")
            r_presed = False
            w_presed = False
        

    try:
        frame = image_resize(frame, width=(cv_panel.winfo_width()-25))
        if frame.shape[0] > cv_panel.winfo_height()-25:
            frame = image_resize(frame, height=(cv_panel.winfo_height()-25))
    except:
        pass
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    cv_panel.imgtk = imgtk
    cv_panel.configure(image=imgtk)
    cv_panel.after(100, update_frame)

update_frame()

# Start GUI loop
root.mainloop()
