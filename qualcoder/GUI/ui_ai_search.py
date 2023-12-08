# Form implementation generated from reading ui file 'GUI_UIs\ui_ai_search.ui'
#
# Created by: PyQt6 UI code generator 6.4.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_Dialog_AiSearch(object):
    def setupUi(self, Dialog_AiSearch):
        Dialog_AiSearch.setObjectName("Dialog_AiSearch")
        Dialog_AiSearch.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        Dialog_AiSearch.resize(978, 580)
        self.gridLayout_2 = QtWidgets.QGridLayout(Dialog_AiSearch)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.buttonBox = QtWidgets.QDialogButtonBox(parent=Dialog_AiSearch)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.StandardButton.Cancel|QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout_2.addWidget(self.buttonBox, 1, 0, 1, 1)
        self.groupBox_2 = QtWidgets.QGroupBox(parent=Dialog_AiSearch)
        self.groupBox_2.setStyleSheet("QGroupBox {border: none}")
        self.groupBox_2.setTitle("")
        self.groupBox_2.setFlat(False)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout.setContentsMargins(-1, 5, -1, 2)
        self.gridLayout.setObjectName("gridLayout")
        self.label_3 = QtWidgets.QLabel(parent=self.groupBox_2)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 0, 0, 1, 1)
        self.tabWidget = QtWidgets.QTabWidget(parent=self.groupBox_2)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_code_search = QtWidgets.QWidget()
        self.tab_code_search.setObjectName("tab_code_search")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.tab_code_search)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.treeWidget = QtWidgets.QTreeWidget(parent=self.tab_code_search)
        self.treeWidget.setObjectName("treeWidget")
        self.treeWidget.headerItem().setText(0, "Code Tree")
        self.verticalLayout_2.addWidget(self.treeWidget)
        self.checkBox_send_memos = QtWidgets.QCheckBox(parent=self.tab_code_search)
        self.checkBox_send_memos.setChecked(False)
        self.checkBox_send_memos.setObjectName("checkBox_send_memos")
        self.verticalLayout_2.addWidget(self.checkBox_send_memos)
        self.tabWidget.addTab(self.tab_code_search, "")
        self.tab_free_search = QtWidgets.QWidget()
        self.tab_free_search.setObjectName("tab_free_search")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.tab_free_search)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label = QtWidgets.QLabel(parent=self.tab_free_search)
        self.label.setObjectName("label")
        self.verticalLayout_3.addWidget(self.label)
        self.lineEdit_free_topic = QtWidgets.QLineEdit(parent=self.tab_free_search)
        self.lineEdit_free_topic.setObjectName("lineEdit_free_topic")
        self.verticalLayout_3.addWidget(self.lineEdit_free_topic)
        self.label_2 = QtWidgets.QLabel(parent=self.tab_free_search)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_3.addWidget(self.label_2)
        self.textEdit_free_description = QtWidgets.QTextEdit(parent=self.tab_free_search)
        self.textEdit_free_description.setObjectName("textEdit_free_description")
        self.verticalLayout_3.addWidget(self.textEdit_free_description)
        self.tabWidget.addTab(self.tab_free_search, "")
        self.gridLayout.addWidget(self.tabWidget, 1, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(parent=self.groupBox_2)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 0, 1, 1, 1)
        self.splitter = QtWidgets.QSplitter(parent=self.groupBox_2)
        self.splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.splitter.setObjectName("splitter")
        self.splitter_vert = QtWidgets.QSplitter(parent=self.splitter)
        self.splitter_vert.setOrientation(QtCore.Qt.Orientation.Vertical)
        self.splitter_vert.setObjectName("splitter_vert")
        self.listWidget_files = QtWidgets.QListWidget(parent=self.splitter_vert)
        self.listWidget_files.setObjectName("listWidget_files")
        self.listWidget_cases = QtWidgets.QListWidget(parent=self.splitter_vert)
        self.listWidget_cases.setObjectName("listWidget_cases")
        self.widget = QtWidgets.QWidget(parent=self.splitter_vert)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy)
        self.widget.setMinimumSize(QtCore.QSize(0, 24))
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_attributeselect = QtWidgets.QPushButton(parent=self.widget)
        self.pushButton_attributeselect.setObjectName("pushButton_attributeselect")
        self.horizontalLayout.addWidget(self.pushButton_attributeselect)
        self.label_attributes = QtWidgets.QLabel(parent=self.widget)
        self.label_attributes.setMaximumSize(QtCore.QSize(400, 24))
        self.label_attributes.setText("")
        self.label_attributes.setObjectName("label_attributes")
        self.horizontalLayout.addWidget(self.label_attributes)
        self.gridLayout.addWidget(self.splitter, 1, 1, 1, 1)
        self.gridLayout_2.addWidget(self.groupBox_2, 0, 0, 1, 1)

        self.retranslateUi(Dialog_AiSearch)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Dialog_AiSearch)
        Dialog_AiSearch.setTabOrder(self.listWidget_files, self.listWidget_cases)

    def retranslateUi(self, Dialog_AiSearch):
        _translate = QtCore.QCoreApplication.translate
        Dialog_AiSearch.setWindowTitle(_translate("Dialog_AiSearch", "AI search"))
        self.label_3.setText(_translate("Dialog_AiSearch", "What do you want to search for?"))
        self.treeWidget.setToolTip(_translate("Dialog_AiSearch", "Select the code for which you want to find more data"))
        self.checkBox_send_memos.setStatusTip(_translate("Dialog_AiSearch", "Send not only the name but also the memo associated with a code to the AI?"))
        self.checkBox_send_memos.setText(_translate("Dialog_AiSearch", "Send memo to AI"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_code_search), _translate("Dialog_AiSearch", "Code search"))
        self.label.setText(_translate("Dialog_AiSearch", "Topic or phenomenon to search for:"))
        self.lineEdit_free_topic.setToolTip(_translate("Dialog_AiSearch", "Enter a good descriptive name for what you are looking for."))
        self.label_2.setText(_translate("Dialog_AiSearch", "Description:"))
        self.textEdit_free_description.setToolTip(_translate("Dialog_AiSearch", "Give a short description so that the AI can better understand what you are looking for"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_free_search), _translate("Dialog_AiSearch", "Free search"))
        self.label_4.setText(_translate("Dialog_AiSearch", "Where do you want to search?"))
        self.pushButton_attributeselect.setToolTip(_translate("Dialog_AiSearch", "Filter with the help of attributes"))
        self.pushButton_attributeselect.setText(_translate("Dialog_AiSearch", "Select Attributes"))
