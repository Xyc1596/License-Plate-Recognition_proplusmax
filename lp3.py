import os
import sys
import time
import numpy as np
import traceback
import cv2
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import QLineEdit, QPushButton, QWidget, QFileDialog, QMessageBox, QVBoxLayout, \
                              QTableWidget, QTableWidgetItem, QApplication

rcParams['font.sans-serif'] = ['SimHei']    # 选择合适的中文字体
rcParams['axes.unicode_minus'] = False      # 解决负号 '-' 显示为方块的问题
workDir = os.path.abspath(os.path.dirname(sys.argv[0]))
templateDir = os.path.join(workDir, 'refer1')
imgPath:os.PathLike = os.path.join(workDir, 'test.bmp')
charSet:list[str] = []  # 0~9: 数字, 10~33: 字母, 34~64: 汉字
templates:dict[str, list] = {}
img_thresh:cv2.Mat = None
"""二值图像"""
filtered_contours:list[cv2.Mat] = []
"""字符轮廓"""

def LoadTemplate() -> dict[str,int]:
    """加载模板"""
    global charSet, templates
    charSet = [fn for fn in os.listdir(templateDir)]
    tp_len:dict[str, int] = {}
    print(f"\033[33m【加载模板】: \033[0m{templateDir}")
    t1 = time.time()
    try:
        for c in charSet:
            templates[c] = []
            for img in os.listdir(cdir:=os.path.join(templateDir, c)):
                tp = cv2.imdecode(np.fromfile(os.path.join(cdir,img), dtype=np.uint8), 1)
                _, tp = cv2.threshold(cv2.cvtColor(tp, cv2.COLOR_RGB2GRAY), 0, 255, cv2.THRESH_OTSU)
                templates[c].append(tp)
            tp_len[c] = len(templates[c])
            print(f"\t已加载 [{c}] 总数: {tp_len[c]}")
    except Exception as e:
        print(f"\033[31m【加载失败】{e}\n{traceback.print_exc()}\n\033[0m")
        return tp_len
    else:
        print(f"""\033[32m【加载完成】
    用时: \033[0m{time.time()-t1:.6f} sec\033[32m
    模板总数: \033[0m{sum(tp_len.values())}\n""")
    finally:
        return tp_len

def LoadImg() -> None | plt.Figure:
    """读取车牌图像并显示"""
    global img_thresh, filtered_contours
    print(f"\033[33m【加载图片】{imgPath}\033[0m\n")
    img:cv2.Mat = cv2.imread(imgPath)
    if img is None:
        print(f"\033[31m【加载失败】请检查图像路径是否正确\n\033[0m")
        return None

    fig, ax = plt.subplots(3,2)
    fig:plt.Figure
    ax:list[list[plt.Axes]]

    ax[0][0].imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    ax[0][0].set_title("车牌图像")
    ax[0][1].imshow(img_gray:=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    ax[0][1].set_title("灰度图")
    ax[1][0].imshow(img_blurred:=cv2.GaussianBlur(img_gray, (5,5), 0))
    ax[1][0].set_title("高斯模糊")

    # 使用Canny边缘检测
    edged = cv2.Canny(img_blurred, 30, 150)

    # 找到轮廓
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 假设车牌是面积最大的矩形
    img_contour = img.copy()
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    plate_contour = None

    # 遍历轮廓，寻找可能的车牌
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:  # 车牌大多是矩形
            plate_contour = approx
            break

    if plate_contour is None:
        print(f"\033[31m【定位失败】未找到车牌!\033[0m")
    else:
        ax[1][1].imshow(cv2.drawContours(img_contour, [plate_contour], -1, (0, 255, 0), 3))
        ax[1][1].set_title("定位结果")

        # 提取车牌区域
        x, y, w, h = cv2.boundingRect(plate_contour)
        plate_image = img_gray[y:y+h, x:x+w]

        _, img_thresh = cv2.threshold(plate_image, 150, 255, cv2.THRESH_BINARY_INV)
        img_thresh_inv = ~img_thresh
        ax[2][0].imshow(img_thresh, cmap='gray')
        ax[2][0].set_title("二值化车牌")

        # 使用轮廓分割字符
        char_contours, _ = cv2.findContours(img_thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        char_contours:list[cv2.Mat]
        print(f'\033[32m【定位成功】\n\t找到的字符数量: \033[0m{len(char_contours)}')
        img_contour = cv2.cvtColor(img_thresh_inv, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_contour, char_contours, -1, (0, 255, 0), 1)
        ax[2][1].imshow(img_contour)
        ax[2][1].set_title("找到的轮廓")

        # 过滤轮廓
        min_area = 100      # 最小轮廓面积
        max_area = 4000     # 最大轮廓面积（可以根据需要调整）
        filtered_contours = [ct for ct in char_contours if min_area < cv2.contourArea(ct) < max_area]
        print(f'\t\033[32m过滤后的字符数量: \033[0m{len(filtered_contours)}\n')
    return fig

def Recognize() -> tuple[str, plt.Figure]:
    """匹配车牌字符并显示
    Returns:
        `[str,plt.Figure]`:
            `str`: 匹配结果字符串\n
            `plt.Figure`: 显示图像
    """
    def _match(char_img, idx) -> str:
        match idx:
            case 0:     # 第一个字符只能为汉字
                templates_sliced = list(templates.items())[34:]
            case 1:     # 第二个字符只能为字母
                templates_sliced = list(templates.items())[10:34]
            case _:     # 第三个字符起可为数字或英文
                templates_sliced = list(templates.items())[:34]
        best_score = {}
        for tp_char, tp_imgs in templates_sliced:
            score = []
            for tp_img in tp_imgs:
                score.append(cv2.matchTemplate(char_img, tp_img, cv2.TM_CCOEFF)[0][0])  #相关系数匹配，返回值愈大，匹配值越高
            best_score[tp_char] = max(score)
        return max(best_score, key=lambda c:best_score[c])

    fig, ax = plt.subplots(1, len(filtered_contours))
    ax:list[plt.Axes]
    results:list[str] = []
    for idx, contour in enumerate(filtered_contours):
        x, y, w, h = cv2.boundingRect(contour)
        char_img = cv2.resize(img_thresh[y:y+h, x:x+w], (20,20))    # 调整为20x20
        ax[idx].imshow(char_img, cmap='gray')
        ax[idx].set_title(f"字符{idx+1}")
        print(f"\033[33m【正在识别】\033[0m字符 {idx+1}: ", end="")
        t1 = time.time()
        results.append(_match(~char_img, idx))
        print(f"[{results[idx]}]\t用时: {(time.time()-t1):.6f} sec")

    res_str = "".join(results)
    print(f"\033[32m【识别完成】\033[0m{res_str}\n")
    return res_str, fig



class MainWindow:
    def __init__(self) -> None:
        self.ui:QWidget = QUiLoader().load(os.path.join(workDir, 'lp3.ui'))
        self.box_imgpath:QLineEdit = self.ui.box_imgpath
        self.box_tpdir:QLineEdit = self.ui.box_tpdir
        self.btn_imgpath:QPushButton = self.ui.btn_imgpath
        self.btn_tpdir:QPushButton = self.ui.btn_tpdir
        self.wgt_fig1:QVBoxLayout = self.ui.wgt_fig1
        self.wgt_fig2:QVBoxLayout = self.ui.wgt_fig2
        self.box_rec:QLineEdit = self.ui.box_rec
        self.btn_loadimg:QPushButton = self.ui.btn_loadimg
        self.btn_loadtp:QPushButton = self.ui.btn_loadtp
        self.btn_rec:QPushButton = self.ui.btn_rec
        self.table_tp:QTableWidget = self.ui.table_tp

        self.btn_imgpath.clicked.connect(self.GetImgPath)
        self.btn_tpdir.clicked.connect(self.GetTpDir)
        self.btn_loadimg.clicked.connect(self.LoadImg)
        self.btn_loadtp.clicked.connect(self.LoadTp)
        self.btn_rec.clicked.connect(self.Recognize)
        self.box_imgpath.setText(imgPath)
        self.box_tpdir.setText(templateDir)

    @staticmethod
    def ClearLayout(layout:QVBoxLayout) -> None:
        """清除Layout中所有子控件"""
        while layout.count():
            layout.removeItem(item:=layout.takeAt(0))
            del item

    @classmethod
    def DrawFig(cls, layout:QVBoxLayout, fig:plt.Figure) -> None:
        """在Layout中绘制Matplotlib图像"""
        cls.ClearLayout(layout)
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        canvas.draw()

    def GetImgPath(self):
        filepath, _ = QFileDialog.getOpenFileName(self.ui, "选择车牌图像", workDir, "位图(*.bmp,*.png)")
        if filepath:
            self.box_imgpath.setText(filepath)

    def GetTpDir(self):
        filepath = QFileDialog.getExistingDirectory(self.ui, "选择模板路径")
        if filepath:
            self.box_tpdir.setText(filepath)

    def LoadImg(self):
        """加载车牌图像"""
        global imgPath
        imgPath = os.path.abspath(self.box_imgpath.text())
        fig = LoadImg()
        if fig:
            self.DrawFig(self.wgt_fig1, fig)
            self.ClearLayout(self.wgt_fig2)
        else:
            QMessageBox.critical(self.ui, "加载图片失败",
                                 f'无法加载图片："{imgPath}"，请检查控制台错误报告。')

    def LoadTp(self):
        """加载模板"""
        global templateDir
        templateDir = os.path.abspath(self.box_tpdir.text())
        tp_len = LoadTemplate()
        self.table_tp.clearContents()
        self.table_tp.setRowCount(len(tp_len)+1)
        for idx, item in enumerate(tp_len.items()):
            self.table_tp.setItem(idx, 0, QTableWidgetItem(item[0]))
            self.table_tp.setItem(idx, 1, QTableWidgetItem(str(item[1])))
        self.table_tp.setItem(len(tp_len), 0, QTableWidgetItem("【总计】"))
        self.table_tp.setItem(len(tp_len), 1, QTableWidgetItem(str(sum(tp_len.values()))))
        if(len(tp_len) < 65):
            QMessageBox.critical(self.ui, "加载模板失败",
                                 f'未能加载完整模板："{imgPath}"，请检查控制台错误报告。')

    def Recognize(self):
        """识别车牌"""
        if self.wgt_fig1.count() < 1:   # 未加载图片
            self.LoadImg()
        if len(templates)<1 or len(list(templates.values())[0])<1:    # 未加载模板
            self.LoadTp()
        result, fig = Recognize()
        self.box_rec.setText(result)
        self.DrawFig(self.wgt_fig2, fig)



if __name__ == '__main__':
    os.system("")   # 使CMD支持彩色输出
    app = QApplication([])
    window = MainWindow()
    window.ui.show()
    app.exec_()