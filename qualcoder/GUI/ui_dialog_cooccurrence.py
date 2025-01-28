# Form implementation generated from reading ui file 'ui_dialog_cooccurrence.ui'
#
# Created by: PyQt6 UI code generator 6.5.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_Dialog_Coocurrence(object):
    def setupUi(self, Dialog_Coocurrence):
        Dialog_Coocurrence.setObjectName("Dialog_Coocurrence")
        Dialog_Coocurrence.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        Dialog_Coocurrence.resize(694, 543)
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog_Coocurrence)
        self.verticalLayout.setContentsMargins(6, -1, 6, -1)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox = QtWidgets.QGroupBox(parent=Dialog_Coocurrence)
        self.groupBox.setMinimumSize(QtCore.QSize(0, 75))
        self.groupBox.setMaximumSize(QtCore.QSize(16777215, 75))
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.label_selections = QtWidgets.QLabel(parent=self.groupBox)
        self.label_selections.setGeometry(QtCore.QRect(10, 3, 261, 22))
        self.label_selections.setMinimumSize(QtCore.QSize(0, 22))
        self.label_selections.setMaximumSize(QtCore.QSize(16777213, 22))
        self.label_selections.setWordWrap(True)
        self.label_selections.setObjectName("label_selections")
        self.pushButton_export = QtWidgets.QPushButton(parent=self.groupBox)
        self.pushButton_export.setGeometry(QtCore.QRect(290, 1, 32, 32))
        self.pushButton_export.setMinimumSize(QtCore.QSize(32, 32))
        self.pushButton_export.setMaximumSize(QtCore.QSize(32, 32))
        self.pushButton_export.setText("")
        self.pushButton_export.setObjectName("pushButton_export")
        self.checkBox_hide_blanks = QtWidgets.QCheckBox(parent=self.groupBox)
        self.checkBox_hide_blanks.setGeometry(QtCore.QRect(340, 10, 201, 20))
        self.checkBox_hide_blanks.setObjectName("checkBox_hide_blanks")
        self.verticalLayout.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(parent=Dialog_Coocurrence)
        self.groupBox_2.setTitle("")
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout.setContentsMargins(6, 6, 6, 6)
        self.gridLayout.setObjectName("gridLayout")
        self.tableWidget = QtWidgets.QTableWidget(parent=self.groupBox_2)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        self.gridLayout.addWidget(self.tableWidget, 0, 0, 1, 1)
        self.verticalLayout.addWidget(self.groupBox_2)

        self.retranslateUi(Dialog_Coocurrence)
        QtCore.QMetaObject.connectSlotsByName(Dialog_Coocurrence)

    def retranslateUi(self, Dialog_Coocurrence):
        _translate = QtCore.QCoreApplication.translate
        Dialog_Coocurrence.setWindowTitle(_translate("Dialog_Coocurrence", "Code Co-occurence"))
        self.label_selections.setToolTip(_translate("Dialog_Coocurrence", "<html><head/><body><p>Show the overlapping codes.</p></body></html>"))
        self.label_selections.setText(_translate("Dialog_Coocurrence", "Code co-occurence"))
        self.pushButton_export.setToolTip(_translate("Dialog_Coocurrence", "<html><head/><body><p>Export to file</p></body></html>"))
        self.checkBox_hide_blanks.setText(_translate("Dialog_Coocurrence", "Hide blank lines"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog_Coocurrence = QtWidgets.QDialog()
    ui = Ui_Dialog_Coocurrence()
    ui.setupUi(Dialog_Coocurrence)
    Dialog_Coocurrence.show()
    sys.exit(app.exec())
