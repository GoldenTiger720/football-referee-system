import sys
import requests
from PyQt5 import QtCore, QtGui, QtWidgets
from datetime import datetime, timedelta, timezone

class VideoLinkWidget(QtWidgets.QWidget):
    def __init__(self, video_link):
        super().__init__()
        self.layout = QtWidgets.QHBoxLayout()

        self.checkbox = QtWidgets.QCheckBox()
        self.id_label = QtWidgets.QLabel(video_link['id'])
        self.id_label.setStyleSheet("color: #555;")  # Set ID color to darker gray
        self.name_label = QtWidgets.QLabel(video_link['name'])

        # Calculate start time plus 2 hours and format it
        start_time = datetime.fromisoformat(video_link['start_time'].replace("Z", "+00:00")) + timedelta(hours=2)
        formatted_start_time = start_time.strftime('%Y-%m-%d %H:%M')
        self.start_time_label = QtWidgets.QLabel(formatted_start_time)
        
        self.field_label = QtWidgets.QLabel("Field: 1")
        self.countdown_label = QtWidgets.QLabel("Loading...")

        self.layout.addWidget(self.checkbox)
        self.layout.addWidget(self.id_label)
        self.layout.addWidget(self.name_label)
        self.layout.addWidget(self.start_time_label)
        self.layout.addWidget(self.field_label)
        self.layout.addWidget(self.countdown_label)

        self.setLayout(self.layout)

        # Setup countdown timer
        self.setup_countdown(start_time)

        # Apply green background to countdown label if the event is Football or Soccer
        if video_link['name'] in ["Football", "Soccer"]:
            self.countdown_label.setStyleSheet("background-color: #4CAF50; color: #fff;")

    def setup_countdown(self, start_time):
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(lambda: self.update_countdown(start_time))
        self.timer.start(1000)

    def update_countdown(self, start_time):
        now = datetime.now(timezone.utc)
        diff = start_time - now
        if diff.total_seconds() > 0:
            # Format the timedelta to display up to seconds (not milliseconds)
            time_str = str(diff).split('.')[0]
            self.countdown_label.setText(time_str)
            # Check if the countdown is 5 seconds away and the checkbox is checked
            if diff.total_seconds() <= 5 and self.checkbox.isChecked():
                print(f"Event {self.name_label.text()} is about to start in 5 seconds!")
        else:
            self.countdown_label.setText("Event started")
            self.timer.stop()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Links Viewer")
        self.setGeometry(100, 100, 800, 600)

        # Dark theme
        self.setStyleSheet("QWidget { background-color: #333; color: #fff; }")

        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        self.load_video_links()

    def load_video_links(self):
        url = 'https://backendsportunity2017.com/graphql'
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        query = """
        {
          viewer {
            VideoLinks(VideoLinksInput: {
              starttime: "2024-04-29T00:01:00Z",
              endtime: "2024-04-29T23:00:00Z"
            }) {
              id
              name
              start_time
            }
          }
        }
        """
        response = requests.post(url, json={'query': query}, headers=headers)
        if response.status_code == 200:
            data = response.json()
            video_links = data['data']['viewer']['VideoLinks']
            for video_link in video_links:
                widget = VideoLinkWidget(video_link)
                self.layout.addWidget(widget)
        else:
            error_label = QtWidgets.QLabel(f"Failed to fetch data: {response.status_code}")
            self.layout.addWidget(error_label)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
