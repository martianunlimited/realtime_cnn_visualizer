from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage

import numpy as np
import sys
import cv2

from ui_main import Ui_MainWindow


class FeaturesMap(QLabel):
    cell_changed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self._rows = 1
        self._cols = 1
        self.raw_idx = 0

    def mousePressEvent(self, event):
        self.raw_idx = self.getRawNumber(event.pos())
        self.cell_changed.emit(self.raw_idx)

    def setGridSize(self, size):
        self._rows, self._cols = size

    def getRawNumber(self, pos):
        cubeWidth = self.width() // self._rows
        cubeHeight = self.height() // self._cols
        cur_row = pos.x() // cubeWidth
        cur_col = pos.y() // cubeHeight
        return self._rows * cur_col + cur_row

    def resetIdx(self):
        self.raw_idx = 0


class DenseMap(QLabel):
    cell_changed = pyqtSignal(int)

    def __init__(self, parent):
        super().__init__()
        self._rows = 1
        self._cols = 1
        self.raw_idx = 0
        self.cell_limit = 1

    def setGridSize(self, shape):
        self._cols, self._rows, _ = shape

    def setCellNumbers(self, number):
        self.cell_limit = number

    def mousePressEvent(self, event):
        self.raw_idx = self.getRawNumber(event.pos())
        self.cell_changed.emit(self.raw_idx)

    def getRawNumber(self, pos):
        p_height = self.pixmap().height()
        s_height = self.height()
        if p_height < s_height:
            cubeWidth = self.width() / self._rows
            dif = (s_height - p_height) // 2
            cubeHeight = p_height / self._cols
            cur_row = pos.x() // cubeWidth
            cur_col = (pos.y() - dif) // cubeHeight
        else:
            cubeWidth = self.width() / self._rows
            cubeHeight = self.height() / self._cols
            cur_row = pos.x() // cubeWidth
            cur_col = pos.y() // cubeHeight
        return int(self._rows * cur_col + cur_row)

    def resetIdx(self):
        self.raw_idx = 0


class Ui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self._show_colored = True

        self.conv_layers = ["Conv1", "Layer2", "Layer3"] * 8
        self.dense_layers = ["FC1", "Layer2", "Layer3"] * 8
        # self.fillLayers(self.conv_layers, self.dense_layers)

        self.convMap = FeaturesMap()
        # self.ui.scrollAreaMap.setWidget(self.convMap)

        self.currentConv = self.conv_layers[0]
        self.currentDense = self.dense_layers[0]

        # self.convMap.cell_changed.connect(self.changeMapNum)
        # self.ui.comboBoxConv.currentTextChanged.connect(self.ConvLayerChanged)
        # self.ui.comboBoxFC.currentTextChanged.connect(self.DenseLayerChanged)

        self.paused = False
        # self.denseMap = DenseMap(self.ui.scrollAreaDense)
        # self.ui.scrollAreaDense.setWidget(self.denseMap)
        # self.denseMap.cell_changed.connect(self.changeDenseNum)

    @pyqtSlot()
    def PausePlay(self):
        if self.paused:
            self.ui.pushButtonPlay.setText("Pause")
        else:
            self.ui.pushButtonPlay.setText("Play")
        self.paused = not self.paused

    @pyqtSlot(int)
    def changeColorMode(self, state):
        self._show_colored = bool(state)

    @pyqtSlot(int)
    def changeMapNum(self, num):
        self.ui.labelMapNum.setText(str(num))

    @pyqtSlot(int)
    def changeDenseNum(self, num):
        if num < self.denseMap.cell_limit:
            self.ui.labelDenseNum.setText(str(num))
        else:
            self.ui.labelDenseNum.setText("Out")

    @pyqtSlot(str)
    def ConvLayerChanged(self, layer_name):
        self.currentConv = layer_name
        self.ui.labelMapName.setText(self.currentConv)
        self.ui.labelMapNum.setText("0")
        self.convMap.resetIdx()

    @pyqtSlot(str)
    def DenseLayerChanged(self, layer_name):
        self.currentDense = layer_name
        self.ui.labelDenseName.setText(layer_name)
        self.ui.labelDenseNum.setText("0")
        # self.denseMap.resetIdx()

    def fillLayers(self, conv_layers, dense_layers):
        # self.ui.comboBoxConv.clear()
        # self.ui.comboBoxFC.clear()
        if conv_layers:
            # self.ui.comboBoxConv.addItems(conv_layers)
            self.conv_layers = conv_layers
        if dense_layers:
            # self.ui.comboBoxFC.addItems(dense_layers)
            self.dense_layers = dense_layers

    # def setButtons(self, buttons):
    #     widget = QWidget()
    #     layout = QHBoxLayout()
    #     for button in buttons:
    #         btn = QPushButton(button)
    #         btn.setFlat(True)
    #         btn.clicked.connect(self.btnClicked)
    #         layout.addWidget(btn)
    #     widget.setLayout(layout)
    #     self.ui.scrollArea.setWidget(widget)
    #
    #     self.buttons = list(sorted(buttons))
    #     self.currentConv = self.buttons[0]

    def loadActivationScrollMap(self, map, cell_numbers):
        label = self.denseMap
        label.setCellNumbers(cell_numbers)
        label.setGridSize(map.shape)
        img = cv2.cvtColor(map, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape
        bytesPerLine = 3 * width
        qImg = QImage(img, width, height, bytesPerLine, QImage.Format_RGB888)
        # scroll_area = self.ui.scrollAreaDense
        # width = scroll_area.width() - scroll_area.verticalScrollBar().width() - 2
        # pixmap = QPixmap(qImg).scaledToWidth(width)
        # label.setGeometry(0, 0, pixmap.width(), pixmap.height())
        # label.setPixmap(pixmap)

    def loadActivationMap(self, map):
        img = cv2.cvtColor(map, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape
        bytesPerLine = 3 * width
        qImg = QImage(img, width, height, bytesPerLine, QImage.Format_RGB888)
        label = self.ui.labelMapDense
        width, height = label.width(), label.height()
        pixmap = QPixmap(qImg).scaled(width, height)
        label.setPixmap(pixmap)

    def loadMap(self, image, size):
        img = cv2.resize(
            image,
            (self.convMap.width(), self.convMap.height()),
            interpolation=cv2.INTER_NEAREST,
        )
        if self._show_colored:
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
            height, width, _ = img.shape
            bytesPerLine = 3 * width
            qImg = QImage(img, width, height, bytesPerLine, QImage.Format_RGB888)
        else:
            height, width = img.shape
            qImg = QImage(img, width, height, QImage.Format_Grayscale8)
        self.convMap.setPixmap(QPixmap(qImg))
        self.convMap.setGridSize(size)

    def setGradCam(self, image):
        label = self.ui.labelGradCam
        img = cv2.resize(image, (label.width(), label.height()))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape
        bytesPerLine = 3 * width
        qImg = QImage(img, width, height, bytesPerLine, QImage.Format_RGB888)
        label.setPixmap(QPixmap(qImg))

    def setPredTop(self, txt):
        label = self.ui.labelPred_top
        label.setText(txt)

    def setPredAll(self, predList, probas):
        labels = [
            self.ui.labelPred1,
            self.ui.labelPred2,
            self.ui.labelPred3,
            self.ui.labelPred4,
            self.ui.labelPred5,
        ]
        for idx, pred in enumerate(predList):
            txt = f"{pred:20} {probas[idx]:.3f}"
            labels[idx].setText(txt)

    def setDenseValue(self, val):
        self.ui.labelDenseVal.setText(str(val))

    def loadRealImage(self, image):
        label = self.ui.labelInput
        img = cv2.resize(image, (label.width(), label.height()))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape
        bytesPerLine = 3 * width
        qImg = QImage(img, width, height, bytesPerLine, QImage.Format_RGB888)
        label.setPixmap(QPixmap(qImg))

    def loadCell(self, image):
        img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_NEAREST)
        if self._show_colored:
            img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
            height, width, _ = img.shape
            bytesPerLine = 3 * width
            qImg = QImage(img, width, height, bytesPerLine, QImage.Format_RGB888)
        else:
            height, width = img.shape
            qImg = QImage(img, width, height, QImage.Format_Grayscale8)

        self.ui.labelZoomed.setPixmap(QPixmap(qImg))

    def loadSubMap(self, image, row, col):
        img = np.uint8(
            (1.0 - (image - np.min(image)) * 1.0 / (np.max(image) - np.min(image)))
            * 255
        )
        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_NEAREST)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        height, width, _ = img.shape
        bytesPerLine = 3 * width
        qImg = QImage(img, width, height, bytesPerLine, QImage.Format_RGB888)

        labels = [
            [
                self.ui.labelMap11,
                self.ui.labelMap12,
                self.ui.labelMap13,
                self.ui.labelMap14,
                self.ui.labelMap15,
            ],
            [
                self.ui.labelMap21,
                self.ui.labelMap22,
                self.ui.labelMap23,
                self.ui.labelMap24,
                self.ui.labelMap25,
            ],
            [
                self.ui.labelMap31,
                self.ui.labelMap32,
                self.ui.labelMap33,
                self.ui.labelMap34,
                self.ui.labelMap35,
            ],
            [
                self.ui.labelMap41,
                self.ui.labelMap42,
                self.ui.labelMap43,
                self.ui.labelMap44,
                self.ui.labelMap45,
            ],
            [
                self.ui.labelMap51,
                self.ui.labelMap52,
                self.ui.labelMap53,
                self.ui.labelMap54,
                self.ui.labelMap55,
            ],
            [
                self.ui.labelMap61,
                self.ui.labelMap62,
                self.ui.labelMap63,
                self.ui.labelMap64,
                self.ui.labelMap65,
            ],
            [
                self.ui.labelMap71,
                self.ui.labelMap72,
                self.ui.labelMap73,
                self.ui.labelMap74,
                self.ui.labelMap75,
            ],
        ]
        labels[row][col].setPixmap(QPixmap(qImg))


def run_ui():
    app = QApplication(sys.argv)
    ui = Ui()
    # ui.setGeometry(500, 300, 300, 400)
    ui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_ui()
