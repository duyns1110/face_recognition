from flask import Flask, request, render_template
from datetime import datetime, date
import os
import pandas as pd
from sqlite3 import Error
import cv2
import classifier
import align_dataset_mtcnn
import face_reg_attendance
import sqlite3
from imutils.video import VideoStream

### Defining Flask App
app = Flask(__name__)

try:
    con = sqlite3.connect('Students.db')
except Error:
    print(Error)

def check_table(con):
    cursorObj = con.cursor()
    list_of_table = cursorObj.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='Students';").fetchall()
    if list_of_table == []:
        print("Table not found!")
        return True
    else:
        print("Table exists")
        return False


def sql_table(con):
    cursorObj = con.cursor()
    if check_table(con):
        cursorObj.execute("CREATE TABLE Students(mssv integer PRIMARY KEY, name text)")
        con.commit()
    con.close()

sql_table(con)
def datetoday():
    return date.today().strftime("%m_%d_%y")


def datetoday2():
    return date.today().strftime("%d-%B-%Y")


#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')


if f'Attendance-{datetoday()}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday()}.csv', 'w') as f:
        f.write('Name,Roll,Time')


def number_of_Student():
    con = sqlite3.connect('Students.db')
    cursorObj = con.cursor()
    list_of_students = cursorObj.execute("SELECT * FROM Students").fetchall()
    con.close()
    return len(list_of_students)


def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday()}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l


def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday()}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday()}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')


def check_data(con, mssv):
        cursorObj = con.cursor()
        data = cursorObj.execute(f"SELECT * FROM Students WHERE mssv={mssv}").fetchall()
        if data == []:
            print("Thong tin chua co trong database")
            return True
        else:
            print("Thong tin da co trong database")
            return False


def insert_student(con, mssv, name):
        cursorObj = con.cursor()
        cursorObj.execute('INSERT INTO Students(mssv, name) VALUES(?, ?)', (mssv, name))
        con.commit()


######################### ROUTING FUNCTIONS ##########################
#### Our main page
@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html',  names=names, rolls=rolls, times=times, l=l,
                           totalreg=number_of_Student(), datetoday2=datetoday2())


#### This function will run when we click on Take Attendance Button
@app.route('/start', methods=['GET'])
def start():
    if 'facemodel.pkl' not in os.listdir('Models'):
        return render_template('home.html', totalreg=number_of_Student(), datetoday2=datetoday2(),
                               mess='There is no trained model in the static folder. Please add a new face to continue.')
    face_reg_attendance.main()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=number_of_Student(),
                           datetoday2=datetoday2())


#### This function will run when we add a new user
@app.route('/add', methods=['GET', 'POST'])
def add():
    try:
        con = sqlite3.connect('Students.db')
    except Error:
        print(Error)

    # Nhập tên và mssv
    name = request.form['newusername']
    mssv = request.form['newuserMSSV']

    names, rolls, times, l = extract_attendance()
    print((len(mssv) == 8), check_data(con, mssv))
    if not ((len(mssv) == 8) & check_data(con, mssv)):
        return render_template('home.html', totalreg=number_of_Student(), datetoday2=datetoday2(),
                               mess='MSSV is not valid!!!.')

    # Tạo bảng và nạp dữ liệu với tên được chuẩn hóa
    insert_student(con, mssv, name)
    con.close()

    # Xử lý tên để đặt file ảnh
    name = name.split(" ")
    newname = None
    for x in name:
        if newname == None:
            newname = x
        else:
            newname += ('_' + x)

    # Bắt đầu lấy ảnh
    label = newname + '_' + mssv
    cam = VideoStream(src=0).start()

    # Biến đếm để xác định số ảnh đã chụp
    i = 0
    while True:
        # Capture frame-by-frame
        frame = cam.read()
        frame = cv2.flip(frame, 1)

        # Vẽ khung giữa màn hình để người dùng đưa mặt vào
        centerH = frame.shape[0] // 2;
        centerW = frame.shape[1] // 2;
        sizeboxW = 300;
        sizeboxH = 400;

        # Tạo thư mục nếu chưa có
        imagefolder = 'DataSet/FaceData/raw/' + str(label)
        print(os.path.isdir(imagefolder))
        if not os.path.isdir(imagefolder):
            os.makedirs(imagefolder)
        print(imagefolder)
        print(os.path.isdir(imagefolder))

        # Lưu dữ liệu
        if cv2.waitKey(1) & 0xFF == 32:
            if not cv2.imwrite(imagefolder + "/" + str(newname) + '_' + str(i) + ".jpg", frame):
                raise Exception("Could not write image")
            i += 1
        cv2.rectangle(frame, (centerW - sizeboxW // 2, centerH - sizeboxH // 2),
                          (centerW + sizeboxW // 2, centerH + sizeboxH // 2),
                          (255, 255, 255), 5)
        cv2.putText(frame, f'Images Captured: {i}/10', (30, 30), cv2.FONT_HERSHEY_SIMPLEX
                        , 1, (255, 0, 255), 2, cv2.LINE_AA)

        # Hiển thị
        cv2.imshow('frame', frame)

        if (cv2.waitKey(1) == ord('q') or i == 10):
            break

    # When everything done, release the capture
    cam.stream.release()
    cv2.destroyAllWindows()
    print("Take face from picture")
    align_dataset_mtcnn.take_face()
    if number_of_Student() >= 2:
        classifier.classifier()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l,
                           totalreg=number_of_Student(), datetoday2=datetoday2())


##### Our main funcion which rún the FLasks App
if __name__ == '__main__':
    app.run(debug=True)