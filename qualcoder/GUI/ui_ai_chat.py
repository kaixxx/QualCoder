# Form implementation generated from reading ui file 'c:\Users\kai\Documents\Programmierung\2023QualCoder\GUI_UIs\ui_ai_chat.ui'
#
# Created by: PyQt6 UI code generator 6.7.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_Dialog_ai_chat(object):
    def setupUi(self, Dialog_ai_chat):
        Dialog_ai_chat.setObjectName("Dialog_ai_chat")
        Dialog_ai_chat.resize(946, 579)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Dialog_ai_chat.sizePolicy().hasHeightForWidth())
        Dialog_ai_chat.setSizePolicy(sizePolicy)
        self.gridLayout = QtWidgets.QGridLayout(Dialog_ai_chat)
        self.gridLayout.setObjectName("gridLayout")
        self.widget_left = QtWidgets.QWidget(parent=Dialog_ai_chat)
        self.widget_left.setObjectName("widget_left")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget_left)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.listWidget_chat_list = QtWidgets.QListWidget(parent=self.widget_left)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.listWidget_chat_list.sizePolicy().hasHeightForWidth())
        self.listWidget_chat_list.setSizePolicy(sizePolicy)
        self.listWidget_chat_list.setToolTip("")
        self.listWidget_chat_list.setObjectName("listWidget_chat_list")
        self.verticalLayout_2.addWidget(self.listWidget_chat_list)
        self.widget_left_buttons = QtWidgets.QWidget(parent=self.widget_left)
        self.widget_left_buttons.setObjectName("widget_left_buttons")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget_left_buttons)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_new_analysis = QtWidgets.QPushButton(parent=self.widget_left_buttons)
        self.pushButton_new_analysis.setObjectName("pushButton_new_analysis")
        self.horizontalLayout.addWidget(self.pushButton_new_analysis)
        self.pushButton_delete = QtWidgets.QPushButton(parent=self.widget_left_buttons)
        self.pushButton_delete.setEnabled(False)
        self.pushButton_delete.setObjectName("pushButton_delete")
        self.horizontalLayout.addWidget(self.pushButton_delete)
        self.pushButton_help = QtWidgets.QPushButton(parent=self.widget_left_buttons)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_help.sizePolicy().hasHeightForWidth())
        self.pushButton_help.setSizePolicy(sizePolicy)
        self.pushButton_help.setText("")
        self.pushButton_help.setIconSize(QtCore.QSize(16, 16))
        self.pushButton_help.setObjectName("pushButton_help")
        self.horizontalLayout.addWidget(self.pushButton_help)
        self.verticalLayout_2.addWidget(self.widget_left_buttons)
        self.gridLayout.addWidget(self.widget_left, 0, 0, 1, 1)
        self.widget_chat = QtWidgets.QWidget(parent=Dialog_ai_chat)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_chat.sizePolicy().hasHeightForWidth())
        self.widget_chat.setSizePolicy(sizePolicy)
        self.widget_chat.setObjectName("widget_chat")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.widget_chat)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setVerticalSpacing(0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.progressBar_ai = QtWidgets.QProgressBar(parent=self.widget_chat)
        self.progressBar_ai.setMinimumSize(QtCore.QSize(0, 6))
        self.progressBar_ai.setMaximumSize(QtCore.QSize(16777215, 6))
        self.progressBar_ai.setMaximum(0)
        self.progressBar_ai.setProperty("value", 0)
        self.progressBar_ai.setTextVisible(False)
        self.progressBar_ai.setObjectName("progressBar_ai")
        self.gridLayout_2.addWidget(self.progressBar_ai, 1, 0, 1, 2)
        self.plainTextEdit_question = QtWidgets.QPlainTextEdit(parent=self.widget_chat)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.plainTextEdit_question.sizePolicy().hasHeightForWidth())
        self.plainTextEdit_question.setSizePolicy(sizePolicy)
        self.plainTextEdit_question.setMaximumSize(QtCore.QSize(16777215, 80))
        self.plainTextEdit_question.setPlainText("")
        self.plainTextEdit_question.setPlaceholderText("")
        self.plainTextEdit_question.setObjectName("plainTextEdit_question")
        self.gridLayout_2.addWidget(self.plainTextEdit_question, 3, 0, 1, 1)
        self.scrollArea_ai_output = QtWidgets.QScrollArea(parent=self.widget_chat)
        self.scrollArea_ai_output.setWidgetResizable(True)
        self.scrollArea_ai_output.setObjectName("scrollArea_ai_output")
        self.scrollArea_ai_output_contents = QtWidgets.QWidget()
        self.scrollArea_ai_output_contents.setGeometry(QtCore.QRect(0, 0, 664, 467))
        self.scrollArea_ai_output_contents.setObjectName("scrollArea_ai_output_contents")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.scrollArea_ai_output_contents)
        self.verticalLayout.setContentsMargins(30, 6, 30, 6)
        self.verticalLayout.setObjectName("verticalLayout")
        self.ai_output = QtWidgets.QLabel(parent=self.scrollArea_ai_output_contents)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ai_output.sizePolicy().hasHeightForWidth())
        self.ai_output.setSizePolicy(sizePolicy)
        self.ai_output.setAutoFillBackground(False)
        self.ai_output.setInputMethodHints(QtCore.Qt.InputMethodHint.ImhNone)
        self.ai_output.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.ai_output.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.ai_output.setText("")
        self.ai_output.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.ai_output.setScaledContents(False)
        self.ai_output.setAlignment(QtCore.Qt.AlignmentFlag.AlignBottom|QtCore.Qt.AlignmentFlag.AlignLeading|QtCore.Qt.AlignmentFlag.AlignLeft)
        self.ai_output.setWordWrap(True)
        self.ai_output.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.LinksAccessibleByKeyboard|QtCore.Qt.TextInteractionFlag.LinksAccessibleByMouse|QtCore.Qt.TextInteractionFlag.TextBrowserInteraction|QtCore.Qt.TextInteractionFlag.TextSelectableByKeyboard|QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.ai_output.setObjectName("ai_output")
        self.verticalLayout.addWidget(self.ai_output)
        self.scrollArea_ai_output.setWidget(self.scrollArea_ai_output_contents)
        self.gridLayout_2.addWidget(self.scrollArea_ai_output, 0, 0, 1, 2)
        self.pushButton_question = QtWidgets.QPushButton(parent=self.widget_chat)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_question.sizePolicy().hasHeightForWidth())
        self.pushButton_question.setSizePolicy(sizePolicy)
        self.pushButton_question.setMinimumSize(QtCore.QSize(80, 0))
        self.pushButton_question.setMaximumSize(QtCore.QSize(80, 16777215))
        font = QtGui.QFont()
        font.setPointSize(30)
        self.pushButton_question.setFont(font)
        self.pushButton_question.setText("")
        self.pushButton_question.setIconSize(QtCore.QSize(32, 32))
        self.pushButton_question.setShortcut("")
        self.pushButton_question.setFlat(False)
        self.pushButton_question.setObjectName("pushButton_question")
        self.gridLayout_2.addWidget(self.pushButton_question, 3, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 6, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Preferred)
        self.gridLayout_2.addItem(spacerItem, 2, 0, 1, 1)
        self.gridLayout.addWidget(self.widget_chat, 0, 1, 1, 1)

        self.retranslateUi(Dialog_ai_chat)
        QtCore.QMetaObject.connectSlotsByName(Dialog_ai_chat)

    def retranslateUi(self, Dialog_ai_chat):
        _translate = QtCore.QCoreApplication.translate
        Dialog_ai_chat.setWindowTitle(_translate("Dialog_ai_chat", "AI Chat"))
        self.pushButton_new_analysis.setToolTip(_translate("Dialog_ai_chat", "Create a new chat"))
        self.pushButton_new_analysis.setText(_translate("Dialog_ai_chat", "New"))
        self.pushButton_delete.setToolTip(_translate("Dialog_ai_chat", "Delete the selected chat"))
        self.pushButton_delete.setText(_translate("Dialog_ai_chat", "Delete"))
        self.pushButton_help.setToolTip(_translate("Dialog_ai_chat", "Help"))
        self.plainTextEdit_question.setToolTip(_translate("Dialog_ai_chat", "Enter your question here and press Enter to continue the chat"))
