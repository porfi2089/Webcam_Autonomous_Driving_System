import tkinter as tk
from tkinter import ttk
import cv2
import json
import numpy as np


# Create GUI
root = tk.Tk()
root.title("Settings Editor")
root.geometry("800x600")
# JSON parameters labels and entries
labels = ["Line Reductions:", "Cam Index:", "Threshold 1:", "Threshold 2:", "Frame Width:", "Frame Height:",
          "Adaptive Setting 1:", "Adaptive Setting 2:", "Serial Baudrate:",
          "Horizontal Line Threshold:"]

menu_frame = ttk.Frame(root)
image_frame = ttk.Frame(root)
image_frame.place(relx=0.4, rely=0, relwidth=0.6, relheight=1)
menu_frame.place(relx=0, rely=0, relwidth=0.4, relheight=1)
menu_frame.grid_columnconfigure((0,1), weight=1)
menu_frame.grid_rowconfigure(list(range(len(labels)+1)), weight=1)
entries = {}

line_reductions_label = tk.Label(menu_frame, text=labels[0])
line_reductions_label.grid(row=0, column=0, sticky="e")
line_reductions_entry = tk.Entry(menu_frame)
line_reductions_entry.grid(row=0, column=1, padx=5, pady=5)
line_reductions_entry.insert(0, "Enter Value")
entries[labels[0]] = line_reductions_entry

cam_index_label = tk.Label(menu_frame, text=labels[1])
cam_index_label.grid(row=1, column=0, sticky="e")
cam_index_entry = tk.Entry(menu_frame)
cam_index_entry.grid(row=1, column=1, padx=5, pady=5)
cam_index_entry.insert(0, "Enter Value")
entries[labels[1]] = cam_index_entry

threshold1_label = tk.Label(menu_frame, text=labels[2])
threshold1_label.grid(row=2, column=0, sticky="e")
threshold1_entry = tk.Entry(menu_frame)
threshold1_entry.grid(row=2, column=1, padx=5, pady=5)
threshold1_entry.insert(0, "Enter Value")
entries[labels[2]] = threshold1_entry

threshold2_label = tk.Label(menu_frame, text=labels[3])
threshold2_label.grid(row=3, column=0, sticky="e")
threshold2_entry = tk.Entry(menu_frame)
threshold2_entry.grid(row=3, column=1, padx=5, pady=5)
threshold2_entry.insert(0, "Enter Value")
entries[labels[3]] = threshold2_entry

frame_width_label = tk.Label(menu_frame, text=labels[4])
frame_width_label.grid(row=4, column=0, sticky="e")
frame_width_entry = tk.Entry(menu_frame)
frame_width_entry.grid(row=4, column=1, padx=5, pady=5)
frame_width_entry.insert(0, "Enter Value")
entries[labels[4]] = frame_width_entry

frame_height_label = tk.Label(menu_frame, text=labels[5])
frame_height_label.grid(row=5, column=0, sticky="e")
frame_height_entry = tk.Entry(menu_frame)
frame_height_entry.grid(row=5, column=1, padx=5, pady=5)
frame_height_entry.insert(0, "Enter Value")
entries[labels[5]] = frame_height_entry

adaptive_setting1_label = tk.Label(menu_frame, text=labels[6])
adaptive_setting1_label.grid(row=6, column=0, sticky="e")
adaptive_setting1_entry = tk.Entry(menu_frame)
adaptive_setting1_entry.grid(row=6, column=1, padx=5, pady=5)
adaptive_setting1_entry.insert(0, "Enter Value")
entries[labels[6]] = adaptive_setting1_entry

adaptive_setting2_label = tk.Label(menu_frame, text=labels[7])
adaptive_setting2_label.grid(row=7, column=0, sticky="e")
adaptive_setting2_entry = tk.Entry(menu_frame)
adaptive_setting2_entry.grid(row=7, column=1, padx=5, pady=5)
adaptive_setting2_entry.insert(0, "Enter Value")
entries[labels[7]] = adaptive_setting2_entry

serial_baudrate_label = tk.Label(menu_frame, text=labels[8])
serial_baudrate_label.grid(row=8, column=0, sticky="e")
serial_baudrate_entry = tk.Entry(menu_frame)
serial_baudrate_entry.grid(row=8, column=1, padx=5, pady=5)
serial_baudrate_entry.insert(0, "Enter Value")
entries[labels[8]] = serial_baudrate_entry

serial_port_label = tk.Label(menu_frame, text="serial_port:")
serial_port_label.grid(row=9, column=0, sticky="e")
serial_port_entry = ttk.Combobox(menu_frame, values=["COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", "COM10", "COM11", "COM12", "COM13", "COM14", "COM15"])
serial_port_entry.grid(row=9, column=1, padx=5, pady=5)
serial_port_entry.insert(0, "Enter Value")

horizontal_line_threshold_label = tk.Label(menu_frame, text=labels[9])
horizontal_line_threshold_label.grid(row=10, column=0, sticky="e")
horizontal_line_threshold_entry = tk.Entry(menu_frame)
horizontal_line_threshold_entry.grid(row=10, column=1, padx=5, pady=5)
horizontal_line_threshold_entry.insert(0, "Enter Value")
entries[labels[9]] = horizontal_line_threshold_entry


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
            "horizontal_line_threshold": [int(horizontal_line_threshold_entry.get())]
        }
        with open("settings.json", "w") as file:
            json.dump(data, file, indent=4)

# Update Button
update_button = tk.Button(root, text="Update", command=update_json)
update_button.grid(row=len(labels)+1, columnspan=2, pady=10)

# OpenCV display
cap = cv2.VideoCapture(1)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cv_panel = tk.Label(image_frame, width=frame_width//2, height=frame_height//2)
cv_panel.pack(fill="both", expand=True)

def update_frame():
    try:
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow("frame", frame)
        img = cv2.resize(frame, (frame_width//2, frame_height//2))
        img_array = np.array(img)
        img = tk.PhotoImage(data=img_array.tobytes())
        cv_panel.img = img
        cv_panel.config(image=img)
    except:
        pass
    cv_panel.after(10, update_frame)

update_frame()

# Start GUI loop
root.mainloop()
