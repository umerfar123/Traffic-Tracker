# Traffic Tracker 
This project is a Python-based application designed to accurately count vehicles 🚗 in real-time video footage. It includes OpenCV for image processing, 
YOLOv8 for object detection, and the SORT algorithm for robust object tracking. By combining these technologies, 
the Traffic Tracker provides a reliable and efficient solution for various traffic monitoring applications.

## Key Features

* Real-time Vehicle Detection: Utilizes YOLOv8 to detect vehicles in incoming video frames at high speeds.
* Accurate Vehicle Counting: Employs advanced object tracking techniques to ensure vehicles are counted precisely, avoiding double-counting or missed counts.
* Robust Object Tracking: Incorporates the [SORT](https://github.com/abewley/sort) algorithm to maintain object identities across frames, even in challenging conditions.

## How it Works

* Video Capture: The application reads video frames from the specified source.
* Object Detection: YOLOv8 is used to detect vehicles within each frame, returning bounding boxes and confidence scores.
* Object Tracking: The SORT algorithm associates detected objects across frames to maintain their identities.
* Vehicle Counting: The application increments the vehicle count whenever a tracked object crosses a predefined line.
* Visualization: The results are displayed in real-time, including detected objects, tracked trajectories, and the current vehicle count.

## Installation

1. Clone this repository
   
   ```
     git clone https://github.com/umerfar123/Traffic-Tracker.git
   ```

2. Install dependencies using

   ```python
    pip install -r requirements.txt
   ```

> [!NOTE]
> You should have installed visual studio c++ development kit to install certain libraries.

3. Run the main file

    ```python
    python run main.py
   ```

## Demo

https://github.com/user-attachments/assets/fba097c3-aeac-4e3a-a6e8-4fede84a6b5d

## Contributions

Contributions to this project are welcome! Feel free to fork the repository, make improvements, and submit pull requests.

## License

This project is licensed under the MIT License.


   
