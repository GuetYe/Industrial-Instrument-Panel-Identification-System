#数据库配置变量
HOSTNAME = '127.0.0.1'
PORT     = '3306'
DATABASE = 'PannelDetection'
USERNAME = 'root'
PASSWORD = '246789'
DB_URI = 'mysql+pymysql://{}:{}@{}:{}/{}?charset=utf8'\
    .format(USERNAME,PASSWORD,HOSTNAME,PORT,DATABASE)
SQLALCHEMY_DATABASE_URI = DB_URI
SQLALCHEMY_TRACK_MODIFICATIONS= True