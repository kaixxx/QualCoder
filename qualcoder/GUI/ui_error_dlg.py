# Form implementation generated from reading ui file 'c:\Users\kai\Documents\Programmierung\2023QualCoder\GUI_UIs\ui_error_dlg.ui'
#
# Created by: PyQt6 UI code generator 6.7.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_ErrorDlg(object):
    def setupUi(self, ErrorDlg):
        ErrorDlg.setObjectName("ErrorDlg")
        ErrorDlg.resize(447, 317)
        ErrorDlg.setSizeGripEnabled(True)
        ErrorDlg.setModal(True)
        self.verticalLayout = QtWidgets.QVBoxLayout(ErrorDlg)
        self.verticalLayout.setObjectName("verticalLayout")
        self.widget_main = QtWidgets.QWidget(parent=ErrorDlg)
        self.widget_main.setObjectName("widget_main")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget_main)
        self.verticalLayout_2.setContentsMargins(9, 9, 9, 9)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.widget_error_message = QtWidgets.QWidget(parent=self.widget_main)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_error_message.sizePolicy().hasHeightForWidth())
        self.widget_error_message.setSizePolicy(sizePolicy)
        self.widget_error_message.setObjectName("widget_error_message")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget_error_message)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(18)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_icon = QtWidgets.QLabel(parent=self.widget_error_message)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_icon.sizePolicy().hasHeightForWidth())
        self.label_icon.setSizePolicy(sizePolicy)
        self.label_icon.setMinimumSize(QtCore.QSize(64, 64))
        self.label_icon.setObjectName("label_icon")
        self.horizontalLayout.addWidget(self.label_icon)
        self.label_error_message = QtWidgets.QLabel(parent=self.widget_error_message)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_error_message.sizePolicy().hasHeightForWidth())
        self.label_error_message.setSizePolicy(sizePolicy)
        self.label_error_message.setWordWrap(True)
        self.label_error_message.setObjectName("label_error_message")
        self.horizontalLayout.addWidget(self.label_error_message)
        self.verticalLayout_2.addWidget(self.widget_error_message)
        self.widget_traceback = QtWidgets.QWidget(parent=self.widget_main)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_traceback.sizePolicy().hasHeightForWidth())
        self.widget_traceback.setSizePolicy(sizePolicy)
        self.widget_traceback.setObjectName("widget_traceback")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.widget_traceback)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        spacerItem = QtWidgets.QSpacerItem(20, 18, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        self.verticalLayout_4.addItem(spacerItem)
        self.line = QtWidgets.QFrame(parent=self.widget_traceback)
        self.line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout_4.addWidget(self.line)
        self.label_2 = QtWidgets.QLabel(parent=self.widget_traceback)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_4.addWidget(self.label_2)
        self.plainTextEdit = QtWidgets.QPlainTextEdit(parent=self.widget_traceback)
        self.plainTextEdit.setEnabled(True)
        self.plainTextEdit.setStyleSheet("background: transparent;")
        self.plainTextEdit.setReadOnly(True)
        self.plainTextEdit.setBackgroundVisible(False)
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.verticalLayout_4.addWidget(self.plainTextEdit)
        self.verticalLayout_2.addWidget(self.widget_traceback)
        self.verticalLayout.addWidget(self.widget_main)
        self.buttonBox = QtWidgets.QDialogButtonBox(parent=ErrorDlg)
        self.buttonBox.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.buttonBox.setCenterButtons(False)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(ErrorDlg)
        self.buttonBox.accepted.connect(ErrorDlg.accept) # type: ignore
        self.buttonBox.rejected.connect(ErrorDlg.reject) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(ErrorDlg)

    def retranslateUi(self, ErrorDlg):
        _translate = QtCore.QCoreApplication.translate
        ErrorDlg.setWindowTitle(_translate("ErrorDlg", "Uncaught exception"))
        self.label_error_message.setText(_translate("ErrorDlg", "<Error Message> "))
        self.label_2.setText(_translate("ErrorDlg", "Error traceback (most recent call last):"))
        self.plainTextEdit.setPlainText(_translate("ErrorDlg", "<traceback>"))