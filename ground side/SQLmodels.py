from SQLexts import db
from datetime import datetime

class PanelDatabase(db.Model):
    __tablename__ = "panel_data"
    id = db.Column(db.Integer,primary_key=True,autoincrement=True)    # 映射到数据库当中的列

    panel = db.Column(db.String(200),nullable=False)
    number = db.Column(db.Integer, nullable=False)
    value = db.Column(db.Float,nullable=False)
    image_data = db.Column(db.BLOB)
    creat_time = db.Column(db.DateTime,default=datetime.now)
    min_value = db.Column(db.Float, nullable=False)
    max_value = db.Column(db.Float, nullable=False)