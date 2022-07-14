from flask import Flask, render_template, request,g
from werkzeug.utils import secure_filename
import pandas as pd
import shutil
import os
import time
import subprocess
from importlib import reload
app = Flask(__name__)

@app.route("/file_upload", methods = ['GET','POST'])
def file_upload():

    if os.path.exists("./static/user_root_test/a/"):
        file1 = './static/user_root_test/a/user_root_test_img.png'
    if os.path.exists("./static/user_root_test/b/"):
        file2 = './static/user_root_test/b/user_root_test_img.png'


    if os.path.exists("./root_test/a/"):
        for file in os.scandir("./root_test/a/"):
            os.remove(file.path)
        # file3 = './root_test/a/'
    if os.path.exists("./root_test/b/"):
        for file in os.scandir("./root_test/b/"):
            os.remove(file.path)
        # file4 = './root_test/b/'
    # if os.path.exists("./fake_images/"):
    #     file5 = './fake_images/fake_img.png'
    if os.path.exists("./static/user_fake_images/"):
        for file in os.scandir("./static/user_fake_images/"):
            os.remove(file.path)
    # if os.path.exists("./static/user_fake_images/"):
    #     file6 = './static/user_fake_images/fake_img.png'
    if os.path.exists("./fake_images/"):
        for file in os.scandir("./fake_images/"):
            os.remove(file.path)
    if os.path.exists(file1):
        os.remove(file1)
    if os.path.exists(file2):
        os.remove(file2)
    # if os.path.exists(file3):
    #     os.remove(file3)
    # if os.path.exists(file4):
    #     os.remove(file4)
    # if os.path.exists(file5):
    #     os.remove(file5)
    # if os.path.exists(file6):
    #     os.remove(file6)
    # #


    if request.method =='POST':
        f1 = request.files['file']



        # f.save('./user_drawing_img/'+secure_filename(f.filename))
        f1.save('./root_test/a/' + secure_filename(f1.filename))
        origin = "C:\\Users\\qorgh2akfl\\Desktop\\flask_server\\root_test\\a\\"+ secure_filename(f1.filename)
        copy = "C:\\Users\\qorgh2akfl\\Desktop\\flask_server\\root_test\\b\\"+ secure_filename(f1.filename)
        copy2 = "C:\\Users\\qorgh2akfl\\Desktop\\flask_server\\static\\user_root_test\\a\\"+ secure_filename("user_root_test_img"+".png")
        copy3 = "C:\\Users\\qorgh2akfl\\Desktop\\flask_server\\static\\user_root_test\\b\\" + secure_filename("user_root_test_img"+".png")
        shutil.copy(origin, copy)
        shutil.copy(origin, copy2)
        shutil.copy(origin, copy3)
        #f1.save('./root_test/b/' + secure_filename(f1.filename))

        #return '파일이 저장되었습니다'
        return render_template('save_img.html')
    else:
        return render_template('file_upload.html')



# @app.route('/save_img', methods = ['GET','POST'])
# def save_img():
#     if request.method == 'POST':
#         keyword = request.form['text']
#
#         with open("./text/save.txt", "w", encoding='utf-8') as f:
#             f.write("%s" % (keyword))
#
#     return render_template('save_img.html')



@app.route('/')
def main():
    if os.path.exists("./static/user_root_test/a/"):
        file1 = './static/user_root_test/a/user_root_test_img.png'
    if os.path.exists("./static/user_root_test/b/"):
        file2 = './static/user_root_test/b/user_root_test_img.png'
    if os.path.exists("./crawl_img/"):
        for file in os.scandir("./crawl_img/"):
            os.remove(file.path)
    if os.path.exists("./text/"):
        for file in os.scandir("./text/"):
            os.remove(file.path)
    if os.path.exists("./feature_image/"):
        for file in os.scandir("./feature_image/"):
            os.remove(file.path)
    if os.path.exists("./static/similar_site/"):
        for file in os.scandir("./static/similar_site/"):
            os.remove(file.path)
    if os.path.exists("./root_test/a/"):


        for file in os.scandir("./root_test/a/"):
            os.remove(file.path)

        # file3 = './root_test/a/'
    if os.path.exists("./root_test/b/"):
        for file in os.scandir("./root_test/b/"):
            os.remove(file.path)
        # file4 = './root_test/b/'
    if os.path.exists("./static/similar_site/"):
        for file in os.scandir("./static/similar_site/"):
            os.remove(file.path)
    if os.path.exists("./fake_images/"):
        for file in os.scandir("./fake_images/"):
            os.remove(file.path)

    if os.path.exists("./static/user_fake_images/"):
        file6 = './static/user_fake_images/fake_img.png'

    if os.path.exists(file1):
        os.remove(file1)
    if os.path.exists(file2):
        os.remove(file2)
    # if os.path.exists(file3):
    #     os.remove(file3)
    # if os.path.exists(file4):
    #     os.remove(file4)
    # if os.path.exists(file5):
    #     os.remove(file5)
    if os.path.exists(file6):
        os.remove(file6)
    return render_template('start.html')


@app.route("/fake_img", methods = ['GET','POST'])
def fake_img():
    if request.method == 'POST':
        keyword = request.form['text']

        with open("./text/save.txt", "w", encoding='utf-8') as f:
            f.write("%s" % (keyword))
    import make_pic

    make_pic

    # if os.path.exists('./static/user_root_test/a/user_root_test_img.png'):
    #     make_pic
    # if (g == 1):
    #     make_pic
    #     print(g)
    # for i in range(5):
    #     print("lllllllllllllll")
    #     import make_pic
    #     make_pic
    print(os.path.exists('./static/user_fake_images/fake_img.png'))
    if (os.path.exists('./static/user_fake_images/fake_img.png')==False):
        import make_pic2
        make_pic2



    print("pl")
    print(os.path.exists('./static/user_fake_images/fake_img.png'))
    print(os.path.exists("./fake_images/fake_img.png"))


    reload(make_pic)

    return render_template('fake_img.html')



    #return render_template('no_3.html')

@app.route('/site')
def site():
    import crawl
    crawl
    import img_model
    img_model
    # reload(crawl)
    # reload(img_model)
    site=img_model.var
    print("site : ",site)
    label = []
    data = []
    csv_test = pd.read_csv('C:\\Users\\qorgh2akfl\\Desktop\\flask_server\\df.csv')
    f_row = csv_test.loc[csv_test["name"] == site]
    name = f_row["name"].astype("string")
    tel = f_row["tel"].astype("string")
    address = f_row["address"].astype("string")
    latitude = f_row["latitude"].astype("string")
    longitude = f_row["longitude"].astype("string")

    name = name.to_string(index=False)
    tel = tel.to_string(index=False)
    address = address.to_string(index=False)
    latitude = latitude.to_string(index=False)
    longitude = longitude.to_string(index=False)
    print(site)
    print(name)
    print(tel)
    print(address)
    print(latitude)
    print(longitude)


    # print(name[0])
    #return render_template('site.html' )
    return render_template('site.html', name=name, tel=tel, address=address, latitude=latitude,
                           longitude=longitude)
    #return render_template('site.html', name=name[1], tel=tel[1], address=address[1], latitude=latitude[1], longitude=longitude[1])
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug = True)
