"""
    本代码用于: Scene Recognition 的 demo 编程实现
    创建时间: 2022 年 02 月 08 日
    创建人: MorningStar
    最后一次修改时间: 2022 年 02 月 08 日
"""
# ==================== 导入必要的包 ==================== #
# ----- 系统操作相关的包 ----- #
import time
import sys
import os 

# ----- 导入后端处理的包 ----- # 
from flask import Flask, render_template, request, url_for, redirect
from werkzeug.utils import secure_filename

# ----- 导入自定义的包 ----- # 
# import tools.hello_world as hello_world
import tools.Scene_Recognition_demo as Scene_Recognition_demo

# ==================== 设置常量参数 ==================== #
UPLOAD_FOLDER = "static/upload"
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


# ==================== 函数实现 ==================== #
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ----- 定义 home 界面 ----- #
@app.route('/')
def home():
    return render_template('home.html')


# ----- 定义检查上传文件是否合格 ----- #
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# ----- 上传文件 ----- #
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            global img_test
            img_test = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            return render_template('home.html', img_test=img_test)


# ----- 运行程序 ----- # 
@app.route('/question', methods=['GET', 'POST'])
def run():
    if request.method == 'POST':
        # 测试程序 #
        # IO_predict, categories_pred, attributes_pred, img_CAM, time = hello_world.hello_world()

        # 真实运行 # 
        IO_predict, categories_pred, attributes_pred, img_CAM, time = Scene_Recognition_demo.Scene_Recognition(img_test)

        return render_template('home.html', img_test=img_test, 
                                IO_predict=IO_predict, 
                                categories_pred=categories_pred, 
                                attributes_pred=attributes_pred, 
                                img_CAM=img_CAM, time=time)


# ==================== 主函数运行 ==================== #
if __name__ == '__main__':
    # ----- 开始计时 ----- #
    T_Start = time.time()

    # ----- 运行程序 ----- # 
    app.run('0.0.0.0', port=5000)

    # ----- 结束计时 ----- #
    T_End = time.time()
    T_Sum = T_End  - T_Start
    T_Hour = int(T_Sum/3600)
    T_Minute = int((T_Sum%3600)/60)
    T_Second = round((T_Sum%3600)%60, 2)
    # print("程序运行时间: {}时{}分{}秒".format(T_Hour, T_Minute, T_Second))
    # print("程序已结束 !")