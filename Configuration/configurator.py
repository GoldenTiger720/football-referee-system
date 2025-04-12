import sys
import json
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QLineEdit, QPushButton, QMessageBox

def load_config(filename):
    """Load JSON configuration from a file."""
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except Exception as e:
        QMessageBox.critical(None, "Error", f"Failed to load configuration: {str(e)}")
        return None

def save_config(filename, config):
    """Save JSON configuration to a file."""
    try:
        with open(filename, 'w') as file:
            json.dump(config, file, indent=4)
        QMessageBox.information(None, "Success", "Configuration saved successfully.")
    except Exception as e:
        QMessageBox.critical(None, "Error", f"Failed to save configuration: {str(e)}")

class ConfigEditor(QWidget):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.config = load_config(filename)
        if not self.config:
            sys.exit(1)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Configuration Editor')
        self.setGeometry(300, 300, 300, 200)

        # Layouts
        layout = QVBoxLayout()
        source_layout = QHBoxLayout()
        mode_layout = QHBoxLayout()
        field_layout = QHBoxLayout()
        start_pos_layout = QHBoxLayout()
        button_layout = QHBoxLayout()

        # Source selection
        source_label = QLabel('Source:')
        self.source_combo = QComboBox()
        self.source_combo.addItems(['file', 'rtsp'])
        self.source_combo.setCurrentText(self.config.get('source', 'file'))
        source_layout.addWidget(source_label)
        source_layout.addWidget(self.source_combo)

        # Mode selection
        mode_label = QLabel('Mode:')
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['streaming', 'recording'])
        self.mode_combo.setCurrentText(self.config.get('mode', 'streaming'))
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo)

        # Field selection
        field_label = QLabel('Field:')
        self.field_combo = QComboBox()
        self.field_combo.addItems(['1', '2'])
        self.field_combo.setCurrentText(str(self.config.get('field', 1)))
        field_layout.addWidget(field_label)
        field_layout.addWidget(self.field_combo)

        # Starting position
        start_pos_label = QLabel('Starting Position:')
        self.start_pos_line = QLineEdit(str(self.config.get('starting_position_sec', 0)))
        start_pos_layout.addWidget(start_pos_label)
        start_pos_layout.addWidget(self.start_pos_line)

        # Buttons
        save_button = QPushButton('Save')
        save_button.clicked.connect(self.save_changes)
        exit_button = QPushButton('Exit')
        exit_button.clicked.connect(self.close)
        button_layout.addWidget(save_button)
        button_layout.addWidget(exit_button)

        # Set main layout
        layout.addLayout(source_layout)
        layout.addLayout(mode_layout)
        layout.addLayout(field_layout)
        layout.addLayout(start_pos_layout)
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def save_changes(self):
        self.config['source'] = self.source_combo.currentText()
        self.config['mode'] = self.mode_combo.currentText()
        self.config['field'] = int(self.field_combo.currentText())
        try:
            starting_pos = int(self.start_pos_line.text())
            if 0 <= starting_pos <= 10000:
                self.config['starting_position_sec'] = starting_pos
                save_config(self.filename, self.config)
            else:
                QMessageBox.critical(self, "Error", "Starting position must be between 0 and 10000.")
        except ValueError:
            QMessageBox.critical(self, "Error", "Starting position must be an integer.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    editor = ConfigEditor('feed_config.json')
    editor.show()
    sys.exit(app.exec_())
