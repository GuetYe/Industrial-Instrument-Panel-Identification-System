import io
import torch
from PIL import Image
from flask import Flask, jsonify, request, render_template,redirect,url_for
from datetime import datetime
from SQLmodels import PanelDatabase
import SQLconfig
from SQLexts import db
from flask_migrate import Migrate
from sqlalchemy import or_
import ReadData

import base64
import cv2

import numpy as np
import yolov7.detect as detect
import Unet.segmentation as segmentation

app = Flask(__name__)       # app为一个Flask类
app.config.from_object(SQLconfig)       # MySQL基础设置
# CORS(app)  # 解决跨域问题
# 把app绑定到db上
db.init_app(app)

migrate = Migrate(app,db)



# select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



def get_prediction(image,filename,image_bytes):
    # print(image_bytes)
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)  # 转换成openCV格式
    # cv2.imshow("image", image_bytes)
    # try:
    value_min = float(request.values["value_min"])  # 得到输入最小值
    value_max = float(request.values["value_max"])  # 得到输入最大值
    date_info = detect.detect_init(data=image,flask_mode=True)  # 检测初始化
    image = detect.run(info=date_info)  # 检测
    # cv2转PIL，为了后续能用语义分割模型
    image_PIL = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # openCv->PIL
    # 语义分割
    img_pointer, img_scale, img_seg = segmentation.predict(image_PIL)   # 语义分割
    scale = ReadData.read(image, img_scale, img_pointer, value_min, value_max)  # 读数
    scale = round(scale, 3) # 保留三位小数

    # 如果已经存在图片，则直接更新读数;否则创建新的仪表盘数据
    panel_model = PanelDatabase.query.filter_by(panel=filename).first()
    if panel_model:
            # MySQL存入数据
            panel_model.value = scale
            panel_model.creat_time = datetime.now()
            panel_model.min_value=value_min
            panel_model.max_value=value_max
            panel_model.image_data=image_bytes
            db.session.commit()
    else:
            #MySQL更改数据
            captcha_model = PanelDatabase(panel=filename, value=scale,image_data=image_bytes
                                          ,min_value=value_min,max_value=value_max)
            db.session.add(captcha_model)
            db.session.commit()

    return_info = {"result":["识别结果："+str(scale)]}   # 返回识别结果
    # except Exception as e:
    #     return_info = {"result": [str(e)]}
    # print(return_info)
        # json格式返回至网页

@app.route("/identify", methods=["POST"])
def identify():
    value_min = float(request.values["value_min"])  # 得到输入最小值
    value_max = float(request.values["value_max"])  # 得到输入最大值
    number = int(request.values["number"])+1
    filename = request.values["filename"]
    image_bytes = request.values["img"].strip("data:;base64,")
    image_bytes = base64.b64decode(image_bytes)
    image = Image.open(io.BytesIO(image_bytes))  # 用PIL打开图像
    # 语义分割
    img_pointer, img_scale, img_seg = segmentation.predict(image)  # 语义分割

    # cv2.imshow("scale", img_scale)
    # cv2.imshow("pointer", img_pointer)
    # cv2.imshow("blend", img_seg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)  # 转换成openCV格式
    scale = ReadData.read(image, img_scale, img_pointer, value_min, value_max)  # 读数


    scale = round(scale, 3)  # 保留三位小数

    # 如果已经存在图片，则直接更新读数;否则创建新的仪表盘数据
    panel_model = PanelDatabase.query.filter_by(panel=filename,number=number).first()
        # MySQL存入数据
    panel_model.value = scale
    panel_model.creat_time = datetime.now()
    panel_model.min_value = value_min
    panel_model.max_value = value_max
    panel_model.image_data = image_bytes
    db.session.commit()
    print("max:",value_max,"min:",value_min,"识别结果为",scale)
    return_info = {"result":["识别结果："+str(scale)]}   # 返回识别结果
    return jsonify(return_info)

@app.route("/detection", methods=["GET","POST"])
@torch.no_grad()
def detection():
        count = 0
        img_list = []
        count_list = []
        image = request.files["images"]   # 获得网站上传输的文件
        filename = image.filename   # 获取图像文件名
        image_bytes = image.read()  # 读取图像数据
        image = Image.open(io.BytesIO(image_bytes)) # 用PIL打开图像

            # info = get_prediction(image,filename,image_bytes)   # 识别
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)  # 转换成openCV格式
        date_info = detect.detect_init(data=image,flask_mode=True)  # 检测初始化
        im_list = detect.run(info=date_info)  # 检测
        for im in im_list:
             bs = cv2.imencode(".jpg",im)[1].tobytes()
             # 如果已经存在图片，则直接更新读数;否则创建新的仪表盘数据
             panel_model = PanelDatabase.query.filter_by(panel=filename,number=str(count+1)).first()
             if panel_model:
                # MySQL存入数据
                panel_model.creat_time = datetime.now()
                panel_model.image_data = bs
                db.session.commit()
             else:
                # MySQL更改数据
                captcha_model = PanelDatabase(panel=filename, value=0,image_data=bs
                                                  ,min_value=0,max_value=0,number=str(count+1))
                db.session.add(captcha_model)
                db.session.commit()
             count_list.append(count)
             count += 1
             img_list.append(base64.b64encode(bs).decode('ascii'))

        return render_template("detection.html",img_list=img_list,filename=filename,count_list=count_list)

@app.route("/", methods=["GET", "POST"])    # 主页
def root():
        return render_template("home_page.html") # 使用identify模板

@app.route("/inquire", methods=["GET", "POST"])
def inquire():
    panels = PanelDatabase.query.order_by(db.text("creat_time")).all() # 按照时间先后显示panels数据
    return render_template("inquire.html",panels=panels)    # 使用inquire模板

@app.route("/inquire/<int:panel_id>", methods=["GET", "POST"])
def panel_detail(panel_id):
    if request.method == 'POST':
        return redirect(url_for("inquire"))
    else:
        panel = PanelDatabase.query.get(panel_id)   #通过id获取表盘信息
        panel.image_data = base64.b64encode(panel.image_data).decode('ascii')   # 解码图片信息
        return render_template("detail.html",panel=panel)


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')