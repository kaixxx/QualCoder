# Form implementation generated from reading ui file 'ui_manage_references.ui'
#
# Created by: PyQt6 UI code generator 6.5.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_Dialog_manage_references(object):
    def setupUi(self, Dialog_manage_references):
        Dialog_manage_references.setObjectName("Dialog_manage_references")
        Dialog_manage_references.resize(899, 491)
        self.gridLayout = QtWidgets.QGridLayout(Dialog_manage_references)
        self.gridLayout.setObjectName("gridLayout")
        self.label_2 = QtWidgets.QLabel(parent=Dialog_manage_references)
        self.label_2.setMinimumSize(QtCore.QSize(0, 30))
        self.label_2.setMaximumSize(QtCore.QSize(16777215, 30))
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 1)
        self.splitter = QtWidgets.QSplitter(parent=Dialog_manage_references)
        self.splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.splitter.setObjectName("splitter")
        self.tableWidget_refs = QtWidgets.QTableWidget(parent=self.splitter)
        self.tableWidget_refs.setObjectName("tableWidget_refs")
        self.tableWidget_refs.setColumnCount(0)
        self.tableWidget_refs.setRowCount(0)
        self.tableWidget_files = QtWidgets.QTableWidget(parent=self.splitter)
        self.tableWidget_files.setObjectName("tableWidget_files")
        self.tableWidget_files.setColumnCount(0)
        self.tableWidget_files.setRowCount(0)
        self.gridLayout.addWidget(self.splitter, 1, 0, 1, 1)
        self.groupBox = QtWidgets.QGroupBox(parent=Dialog_manage_references)
        self.groupBox.setMinimumSize(QtCore.QSize(0, 70))
        self.groupBox.setMaximumSize(QtCore.QSize(16777215, 70))
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.checkBox_hide_files = QtWidgets.QCheckBox(parent=self.groupBox)
        self.checkBox_hide_files.setGeometry(QtCore.QRect(20, 10, 301, 23))
        self.checkBox_hide_files.setObjectName("checkBox_hide_files")
        self.checkBox_hide_refs = QtWidgets.QCheckBox(parent=self.groupBox)
        self.checkBox_hide_refs.setGeometry(QtCore.QRect(20, 40, 301, 23))
        self.checkBox_hide_refs.setObjectName("checkBox_hide_refs")
        self.pushButton_delete_ref = QtWidgets.QPushButton(parent=self.groupBox)
        self.pushButton_delete_ref.setGeometry(QtCore.QRect(560, 8, 30, 30))
        self.pushButton_delete_ref.setText("")
        self.pushButton_delete_ref.setObjectName("pushButton_delete_ref")
        self.pushButton_link = QtWidgets.QPushButton(parent=self.groupBox)
        self.pushButton_link.setGeometry(QtCore.QRect(340, 8, 30, 30))
        self.pushButton_link.setText("")
        self.pushButton_link.setObjectName("pushButton_link")
        self.pushButton_unlink_files = QtWidgets.QPushButton(parent=self.groupBox)
        self.pushButton_unlink_files.setGeometry(QtCore.QRect(380, 8, 30, 30))
        self.pushButton_unlink_files.setText("")
        self.pushButton_unlink_files.setObjectName("pushButton_unlink_files")
        self.pushButton_delete_unused_refs = QtWidgets.QPushButton(parent=self.groupBox)
        self.pushButton_delete_unused_refs.setGeometry(QtCore.QRect(600, 8, 30, 30))
        self.pushButton_delete_unused_refs.setText("")
        self.pushButton_delete_unused_refs.setObjectName("pushButton_delete_unused_refs")
        self.pushButton_edit_ref = QtWidgets.QPushButton(parent=self.groupBox)
        self.pushButton_edit_ref.setGeometry(QtCore.QRect(420, 8, 30, 30))
        self.pushButton_edit_ref.setText("")
        self.pushButton_edit_ref.setObjectName("pushButton_edit_ref")
        self.pushButton_import = QtWidgets.QPushButton(parent=self.groupBox)
        self.pushButton_import.setGeometry(QtCore.QRect(460, 8, 30, 30))
        self.pushButton_import.setText("")
        self.pushButton_import.setObjectName("pushButton_import")
        self.pushButton_auto_link = QtWidgets.QPushButton(parent=self.groupBox)
        self.pushButton_auto_link.setGeometry(QtCore.QRect(500, 8, 30, 30))
        self.pushButton_auto_link.setText("")
        self.pushButton_auto_link.setObjectName("pushButton_auto_link")
        self.gridLayout.addWidget(self.groupBox, 4, 0, 1, 1)

        self.retranslateUi(Dialog_manage_references)
        QtCore.QMetaObject.connectSlotsByName(Dialog_manage_references)
        Dialog_manage_references.setTabOrder(self.tableWidget_refs, self.tableWidget_files)
        Dialog_manage_references.setTabOrder(self.tableWidget_files, self.checkBox_hide_files)
        Dialog_manage_references.setTabOrder(self.checkBox_hide_files, self.checkBox_hide_refs)
        Dialog_manage_references.setTabOrder(self.checkBox_hide_refs, self.pushButton_link)
        Dialog_manage_references.setTabOrder(self.pushButton_link, self.pushButton_unlink_files)
        Dialog_manage_references.setTabOrder(self.pushButton_unlink_files, self.pushButton_edit_ref)
        Dialog_manage_references.setTabOrder(self.pushButton_edit_ref, self.pushButton_import)
        Dialog_manage_references.setTabOrder(self.pushButton_import, self.pushButton_delete_ref)
        Dialog_manage_references.setTabOrder(self.pushButton_delete_ref, self.pushButton_delete_unused_refs)

    def retranslateUi(self, Dialog_manage_references):
        _translate = QtCore.QCoreApplication.translate
        Dialog_manage_references.setWindowTitle(_translate("Dialog_manage_references", "Reference manager"))
        self.label_2.setText(_translate("Dialog_manage_references", "Assign selected file(s) to selected reference. Press L or Link button below."))
        self.checkBox_hide_files.setText(_translate("Dialog_manage_references", "Hide assigned files"))
        self.checkBox_hide_refs.setText(_translate("Dialog_manage_references", "Hide assigned references"))
        self.pushButton_delete_ref.setToolTip(_translate("Dialog_manage_references", "<html><head/><body><p>Delete selected reference</p></body></html>"))
        self.pushButton_link.setToolTip(_translate("Dialog_manage_references", "<html><head/><body><p>Link selected files to selected reference</p></body></html>"))
        self.pushButton_unlink_files.setToolTip(_translate("Dialog_manage_references", "<html><head/><body><p>Unlink selected files from references</p></body></html>"))
        self.pushButton_delete_unused_refs.setToolTip(_translate("Dialog_manage_references", "<html><head/><body><p>Delete all references that are not assigned to files</p></body></html>"))
        self.pushButton_edit_ref.setToolTip(_translate("Dialog_manage_references", "<html><head/><body><p>Edit reference</p></body></html>"))
        self.pushButton_import.setToolTip(_translate("Dialog_manage_references", "<html><head/><body><p>Import references. RIS format.</p></body></html>"))
        self.pushButton_auto_link.setToolTip(_translate("Dialog_manage_references", "<html><head/><body><p>Automatically link references to unassigned file names.</p><p>Word matching uses words from refernce title.</p><p>Strong matches with 70% or more matching words  are linked.</p><p>Linking may be incorrect. </p><p>Review after applying the function.</p></body></html>"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog_manage_references = QtWidgets.QDialog()
    ui = Ui_Dialog_manage_references()
    ui.setupUi(Dialog_manage_references)
    Dialog_manage_references.show()
    sys.exit(app.exec())
