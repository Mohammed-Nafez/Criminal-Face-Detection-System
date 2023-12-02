import sys
import json
import os
import subprocess
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QToolButton, QLineEdit, QVBoxLayout, QHBoxLayout, QWidget, QTextBrowser, QPushButton, QLabel
from PyQt5.QtGui import QIcon, QPixmap

class Ui_Form5(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1400, 900)

        # Create a horizontal layout for the top bar
        top_bar_layout = QtWidgets.QHBoxLayout()

        # Create a search bar widget and add it to the top bar layout (on the left side)
        self.search_bar = QLineEdit(Form)
        self.search_bar.setObjectName("search_bar")
        self.search_bar.setPlaceholderText("Search for names")

        # Create the search button but initially disable it
        self.search_button = QPushButton("Search", Form)
        self.search_button.setObjectName("search_button")
        self.search_button.clicked.connect(self.search_names)
        self.search_button.setEnabled(False)  # Initially disabled

        # Connect the textChanged signal of the search bar to enable/disable the search button
        self.search_bar.textChanged.connect(self.toggle_search_button)

        top_bar_layout.addWidget(self.search_bar)
        top_bar_layout.addWidget(self.search_button)

        # Create the scroll area and its contents
        self.scrollArea = QtWidgets.QScrollArea(Form)
        self.scrollArea.setGeometry(QtCore.QRect(10, 50, 1381, 831))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")

        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 1300, 831))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")

        self.verticalLayout = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout.setObjectName("verticalLayout")

        # Load data from a JSON file (replace 'data.json' with your file)
        with open('registration_data.json', 'r') as json_file:
            data = json.load(json_file)

        self.buttons = {}

        for person in data:
            # Check if the folder exists in the "matches" directory
            folder_name = person['Full Name']
            folder_path = os.path.join('matches', folder_name)
            has_folder = os.path.exists(folder_path)

            person_layout = QtWidgets.QHBoxLayout()
            person_layout.setObjectName(f"layout_{person['Full Name']}")

            image_label = QtWidgets.QLabel()
            image_label.setMinimumSize(QtCore.QSize(250, 300))
            image_label.setMaximumSize(QtCore.QSize(250, 300))
            image_label.setAlignment(QtCore.Qt.AlignCenter)
            image_label.setObjectName(f"label_{person['Full Name']}")

            pixmap = QtGui.QPixmap(person['Image Path'])
            pixmap = pixmap.scaledToWidth(320)
            image_label.setPixmap(pixmap)

            info_container = QtWidgets.QWidget()
            info_container.setObjectName(f"info_container_{person['Full Name']}")
            info_container.setStyleSheet("background: white;")

            info_text = QtWidgets.QTextBrowser(info_container)
            info_text.setAlignment(QtCore.Qt.AlignCenter)
            info_text.setObjectName(f"info_{person['Full Name']}")
            info_text.setPlainText(f"Name: {person['Full Name']}\n"
                                    f"Birthday: {person['Birthday']}\n"
                                    f"ID: {person['ID']}\n"
                                    f"Gender: {person['Gender']}\n"
                                    f"Phone Number: {person['Phone Number']}\n"
                                    f"Address: {person['Address']}")

            font = QtGui.QFont()
            font.setPointSize(17)
            info_text.setFont(font)

            spacer_left = QtWidgets.QSpacerItem(10, 10, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
            spacer_right = QtWidgets.QSpacerItem(10, 10, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)

            info_layout = QtWidgets.QVBoxLayout(info_container)
            info_layout.addItem(spacer_left)
            info_layout.addWidget(info_text)
            info_layout.addItem(spacer_right)

            found_button = QtWidgets.QPushButton("Found", Form)
            found_button.setObjectName(f"found_button_{person['Full Name']}")
            found_button.clicked.connect(lambda checked, name=folder_name: self.execute_open(name))

            delete_button = QtWidgets.QPushButton("Delete", Form)
            delete_button.setObjectName(f"delete_button_{person['Full Name']}")
            delete_button.clicked.connect(lambda checked, name=person['Full Name']: self.delete_person(name))

            not_found_button = QtWidgets.QPushButton("Not Found", Form)
            not_found_button.setObjectName(f"not_found_button_{person['Full Name']}")

            self.buttons[person['Full Name']] = {'found_button': found_button, 'not_found_button': not_found_button}

            person_layout.addWidget(image_label)
            person_layout.addWidget(info_container)
            person_layout.addWidget(found_button)
            person_layout.addWidget(delete_button)  # Add the delete button
            person_layout.addWidget(not_found_button)

            self.verticalLayout.addLayout(person_layout)

            if has_folder:
                found_button.setEnabled(True)
                not_found_button.setEnabled(False)
            else:
                found_button.setEnabled(False)
                not_found_button.setEnabled(True)

        self.scrollArea.setStyleSheet("background-image: url('register.jpg');")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        # Create a back button and set its properties
        self.back_button = QToolButton(Form)
        self.back_button.setGeometry(QtCore.QRect(10, 10, 30, 30))
        self.back_button.setIcon(QIcon("back.png"))
        self.back_button.setIconSize(QtCore.QSize(28, 28))
        self.back_button.setStyleSheet("background-color: white;")
        self.back_button.clicked.connect(self.go_to_GUI)

        # Add the top bar layout to the main layout
        main_layout = QtWidgets.QVBoxLayout(Form)
        main_layout.addLayout(top_bar_layout)
        main_layout.addWidget(self.scrollArea)
        main_layout.addWidget(self.back_button)

    def toggle_search_button(self):
        # Enable the search button if there is text in the search bar, otherwise disable it
        if self.search_bar.text().strip():
            self.search_button.setEnabled(True)
        else:
            self.search_button.setEnabled(False)

    def delete_person(self, name):
        # Load data from a JSON file
        with open('registration_data.json', 'r') as json_file:
            data = json.load(json_file)

        # Find the person to delete by name
        person_to_delete = None
        for person in data:
            if person['Full Name'] == name:
                person_to_delete = person
                break

        if person_to_delete:
            # Delete the person from the list
            data.remove(person_to_delete)

            # Save the updated data back to the JSON file
            with open('registration_data.json', 'w') as json_file:
                json.dump(data, json_file, indent=4)

            # Delete the person's image file
            image_path = os.path.join('images', f"{name}.jpg")
            if os.path.exists(image_path):
                os.remove(image_path)

            # Remove the person's layout from the scroll area
            person_layout = self.verticalLayout.findChild(QtWidgets.QHBoxLayout, f"layout_{name}")
            if person_layout:
                for i in reversed(range(person_layout.count())):
                    widget = person_layout.itemAt(i).widget()
                    if widget is not None:
                        widget.deleteLater()

            # Refresh the scroll area
            self.scrollArea.update()

        else:
            QtWidgets.QMessageBox.information(Form, "Person Not Found", "The person was not found in the data.")


    def search_names(self):
        # Get the search text from the search bar
        search_text = self.search_bar.text().strip().lower()

        # Load data from a JSON file
        with open('registration_data.json', 'r') as json_file:
            data = json.load(json_file)

        results_found = False  # Flag to track if any results were found

        for person in data:
            # Check if the person's name contains the search text
            if search_text in person['Full Name'].lower():
                results_found = True  # Set the flag to True as a result was found

                # Scroll to the person's location in the scroll area
                person_layout = self.verticalLayout.findChild(QtWidgets.QHBoxLayout, f"layout_{person['Full Name']}")
                if person_layout:
                    scroll_pos = person_layout.geometry().top()
                    self.scrollArea.verticalScrollBar().setValue(scroll_pos)

        if not results_found:
            # Handle case when no results are found
            QtWidgets.QMessageBox.information(Form, "No Results", "No results found for the search.")



    def execute_open(self, folder_name):
        # Open the directory based on the folder_name
        directory_path = os.path.join('matches', folder_name)
        if os.path.exists(directory_path):
            operating_system = sys.platform
            try:
                if operating_system.startswith('win'):  # Windows
                    subprocess.Popen(['explorer', directory_path], shell=True)
                elif operating_system.startswith('darwin'):  # macOS
                    subprocess.Popen(['open', directory_path])
                elif operating_system.startswith('linux'):  # Linux
                    subprocess.Popen(['xdg-open', directory_path])
                else:
                    print("Unsupported operating system.")
            except Exception as e:
                print(f"An error occurred: {str(e)}")
        else:
            QtWidgets.QMessageBox.information(Form, "Directory Not Found", "Directory not found for this person.")

    def go_to_GUI(self):
        Form.close()
        upload_process = subprocess.Popen([sys.executable, "GUI.py"])
        upload_process.wait()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form5()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
