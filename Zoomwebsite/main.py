from flask import Flask, render_template, Response
from camera import VideoCamera

import os;

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/UserInp/<ui>')
def userInput(ui):
    direct='/home/teja/Desktop/S2'
    os.chdir(direct)
    File=open('UserInput.txt','w')
    File.write(ui)
    File.close()
    return render_template('index.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(threaded=True, debug=True)
