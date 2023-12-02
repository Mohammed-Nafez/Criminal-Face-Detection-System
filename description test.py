from PyQt5 import QtCore, QtGui, QtWidgets
import hashlib
import sys

# Import the generated UI code
from Signup import Ui_Form3  # Replace with the actual name of your generated UI module

class SignUpApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form3()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.sign_up)

        # Set password masking for lineEdit_3 and lineEdit_4
        self.ui.lineEdit_3.setEchoMode(QtWidgets.QLineEdit.Password)
        self.ui.lineEdit_4.setEchoMode(QtWidgets.QLineEdit.Password)

    def sign_up(self):
        employee_id = self.ui.lineEdit.text()
        password = self.ui.lineEdit_3.text()  # Get password from lineEdit_3
        confirm_password = self.ui.lineEdit_4.text()  # Get confirm password from lineEdit_4

        if not employee_id or not password or not confirm_password:
            QtWidgets.QMessageBox.critical(self, "Error", "All fields are required.")
            return

        if password != confirm_password:
            QtWidgets.QMessageBox.critical(self, "Error", "Passwords do not match.")
            return

        # Read existing IDs from the file
        with open("Employee_ID.txt", "r") as file:
            existing_ids = [line.strip() for line in file.readlines()]

        if employee_id not in existing_ids:
            QtWidgets.QMessageBox.critical(self, "Error", "Employee ID not found.")
            return

        # Hash the password
        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        # Save the ID and hashed password in a file
        with open("Employee_Passwords.txt", "a") as file:
            file.write(f"{employee_id} {hashed_password}\n")

        QtWidgets.QMessageBox.information(self, "Success", "Account created successfully.")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    signup_app = SignUpApp()
    Form = QtWidgets.QWidget()
    ui = Ui_Form3()
    ui.setupUi(Form)
    Form.show()
    #signup_app.show()
    sys.exit(app.exec_())
