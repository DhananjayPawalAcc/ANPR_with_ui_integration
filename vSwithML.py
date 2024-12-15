import flet as ft
import cv2
import base64
import threading
from modelFactory import ANPRModel, YOLOv11SegmentationModel, YOLOv11DetectionModel  # Import the ML models

class WindowStreamer:
    def __init__(self, cam_name, cam_details):
        self.cam_name = cam_name
        self.cam_details = cam_details
        self.connections = {}  # To manage multiple sources and their states
        self.streaming_windows = {}

        # Initialize the model based on cam_details
        self.model = self.initialize_model()

    def initialize_model(self):
        """Initialize the ML model based on camera details."""
        model_used = self.cam_details.get('model_used', '')
        if model_used == 'ANPRModel':
            return ANPRModel()
        elif model_used == 'YOLOv11DetectionModel':
            return YOLOv11DetectionModel()
        elif model_used == 'YOLOv11SegmentationModel':
            return YOLOv11SegmentationModel()
        else:
            return None  # Default to no model if not specified

    def create_streaming_window(self):
        return ft.Container(
            width=720,
            height=480,
            content=ft.Image(
                src="assets/disconnect.png",
                width=50,
                height=50,
                fit=ft.ImageFit.CONTAIN,
            ),
            margin=ft.margin.all(10),
            clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
            bgcolor=ft.colors.BLACK,
            border=ft.border.all(2, ft.colors.OUTLINE),
            expand=True,
        )

    def create_connect_button(self, source_id):
        return ft.ElevatedButton(
            text="Connect",
            style=ft.ButtonStyle(
                color={
                    ft.MaterialState.DEFAULT: ft.colors.WHITE,
                    ft.MaterialState.HOVERED: ft.colors.WHITE,
                },
                bgcolor={
                    ft.MaterialState.DEFAULT: ft.colors.GREEN,
                    ft.MaterialState.HOVERED: ft.colors.GREEN_700,
                },
            ),
            on_click=lambda e: self.toggle_connection(e, source_id)
        )

    def toggle_connection(self, e, source_id):
        if source_id not in self.connections:
            self.connections[source_id] = {
                "is_connected": False,
                "cap": None
            }

        connection = self.connections[source_id]
        connection["is_connected"] = not connection["is_connected"]

        if connection["is_connected"]:
            self.connect(e.control, source_id)
        else:
            self.disconnect(e.control, source_id)

    def connect(self, connect_button, source_id):
        # Update button to "Disconnect"
        connect_button.text = "Disconnect"
        connect_button.style.bgcolor = {
            ft.MaterialState.DEFAULT: ft.colors.RED,
            ft.MaterialState.HOVERED: ft.colors.RED_700,
        }
        connect_button.update()

        source = source_id  # Assign the source (e.g., 0 for webcam or file path)
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print(f"Failed to open video source: {source}")
            return

        # Store the video capture object in the connections dictionary
        self.connections[source_id]["cap"] = cap

        # Start a background thread to read frames
        threading.Thread(target=self.read_frames, args=(source_id,), daemon=True).start()

    def disconnect(self, connect_button, source_id):
        # Update button to "Connect"
        connect_button.text = "Connect"
        connect_button.style.bgcolor = {
            ft.MaterialState.DEFAULT: ft.colors.GREEN,
            ft.MaterialState.HOVERED: ft.colors.GREEN_700,
        }
        connect_button.update()

        connection = self.connections.get(source_id, None)
        if connection and connection["cap"]:
            connection["cap"].release()
            connection["cap"] = None

        # Update streaming window to show disconnected state
        window = self.streaming_windows.get(source_id, None)
        if window:
            window.content.src = "assets/disconnect.png"
            window.update()

    def read_frames(self, source_id):
        connection = self.connections[source_id]
        cap = connection["cap"]
        window = self.streaming_windows[source_id]

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 10  # Default to 30 FPS if metadata is unavailable
        frame_delay = int(1000 / fps)
        print(f"Video FPS: {fps}, Frame delay: {frame_delay} ms")

        while connection["is_connected"]:
            ret, frame = cap.read()

            if not ret:
                print(f"Cannot read frames from source: {source_id}")
                break

            try:
                # # Process the frame (this is where you can integrate model logic)
                # processed_frame = self.process_frame(frame)

                if self.cam_details['model_used'] == 'ANPRModel':
                    frameDict1 = self.model.det_objects(frameDict)
                    frameDict2 = self.model.det_plates_ocr(frameDict1)
                    res_frame = self.model.plot_bounding_boxes(frame,frameDict2)
                elif self.cam_details['model_used'] == 'YOLOv11DetectionModel':
                    res_frame = self.model.predict(frame)
                else:
                    res_frame = frame

                # Encode the processed frame to base64
                _, buffer = cv2.imencode(".jpg", res_frame)
                img_str = base64.b64encode(buffer).decode("utf-8")

                # Update the streaming window with the new frame
                window.content.src_base64 = f"{img_str}"
                window.update()

                # Introduce delay to match the video's frame rate
                cv2.waitKey(frame_delay)

            except Exception as e:
                print(f"Error processing frame for source {source_id}: {e}")
                break

    def process_frame(self, frame):
        """Process the frame using the selected model."""
        if self.model:
            if isinstance(self.model, ANPRModel):
                frame = self.model.detect_license_plate(frame)  # Assume method in ANPRModel
            elif isinstance(self.model, YOLOv11DetectionModel):
                frame = self.YOLOv11DetectionModel()  # Assume method in YOLOv11DetectionModel
            elif isinstance(self.model, YOLOv11SegmentationModel):
                frame = self.model.segment_objects(frame)  # Assume method in YOLOv11SegmentationModel
        return frame

    def build(self, source_id, cam_name):
        streaming_window = self.create_streaming_window()
        self.streaming_windows[source_id] = streaming_window
        connect_button = self.create_connect_button(source_id)

        return ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text(
                        value=cam_name,  # Display the camera name
                        style=ft.TextStyle(size=16, weight="bold"),  # Bold, larger font
                    ),
                    streaming_window,  # Streaming window
                    connect_button,  # Connect button
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=10,  # Spacing between elements
                expand=True,  # Allow the column to expand
            ),
            padding=20,
            border_radius=10,
            bgcolor=ft.colors.SURFACE,
            shadow=ft.BoxShadow(
                spread_radius=1,
                blur_radius=10,
                color=ft.colors.with_opacity(0.3, ft.colors.SHADOW),
            ),
            expand=True,  # Allow the container to expand
        )
