from flask import Flask, render_template, request, redirect, url_for
import sqlite3
from datetime import datetime
import re 

app = Flask(__name__)

def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/',methods=['GET'])
def index(active_slide=1):

    conn = get_db_connection()
    posts = conn.execute('SELECT * FROM posts').fetchall()
    conn.close()

    recent_posts = posts[len(posts)-3:] if len(posts) > 2 else posts

    recent_posts.reverse()

    recent_posts_data = {'created': [], 'title': [], 'content': []}

    display_len = 3 if len(posts) > 2 else 1
    i = 0

    while i < display_len:
        recent_posts_data['created'].append(posts[len(posts)- i - 1]['created'])
        recent_posts_data['title'].append(posts[len(posts)- i - 1]['title'])
        recent_posts_data['content'].append(posts[len(posts)- i - 1]['content'])
        i += 1

    for j in range(len(recent_posts_data['content'])):
        curr_content = recent_posts_data['content'][j]
        if len(curr_content.split(' ')) > 50:
            new_content_list = re.split(r'\s', curr_content, maxsplit=49)
            new_content = ' '.join(new_content_list[:49])
            recent_posts_data['content'][j] = new_content
    
    recent_posts = []
    for p in range(len(recent_posts_data['title'])):
        recent_posts.append([recent_posts_data['created'][p],recent_posts_data['title'][p], recent_posts_data['content'][p]])

    slide_activity = ["carousel-item slide1", "carousel-item slide2", "carousel-item slide3", "carousel-item slide4",
                       "carousel-item slide5"]

    slide_activity[int(active_slide)-1] = "carousel-item active slide"+str(active_slide)


    return render_template('index.html', posts=recent_posts, slide_activity=slide_activity)

@app.route('/posts')
def posts():
    conn = get_db_connection()
    posts = conn.execute('SELECT * FROM posts').fetchall()
    conn.close()
    posts.reverse()
    return render_template('posts.html',posts=posts)

@app.route('/post/<post_name>', methods=['GET','POST'])
def post(post_name):
    conn = get_db_connection()
    res = conn.execute("SELECT created, title, content FROM posts WHERE title = (?)", (post_name,))
    created, title, content = res.fetchone()
    conn.close()
    print(title)

    #print(post)

    return render_template('post.html', created=created, title=title, content=content)

@app.route('/projects/imaithination')
def projects_imaithination():
    project = {'title':'iMAiTHination',
               'visual1':'imathination_index.png',
               'visual2':'sample_image.png',
               'inspiration':'We have all participated in some form of teaching, whether it be tutoring, volunteering, helping out during office hours and PSOs as a TA, even simply explaining a problem to our friends. In our various experiences relating concepts in simpler terms, we\'ve noticed that explaining and learning is often easier done visually. However, when attempting to formulate a math problem into an image can be difficult at first, particularly for younger students who are just beginning to learn the entire new language that is mathematics. We wanted to create a platform to help stimulate visual imagination (or, iMaiTHination ;) ) for math problems, to not only provide an entertaining boost for the present math, but to also cultivate the connection between symbols on paper and what we see through our own eyes. Additionally, our web application would be helpful for children whose second language is English and struggle to get past any language barrier when learning math.',
               'what it does': 'Our application can be split into three parts: reading in data, processing, and generating the image. We allow the user the option to either type in their word problem, or upload a file of the problem. Once the user inputs their data in the desired format, our application will parse the data, and then display an image corresponding to their math word problem.',
               'how it works': 'We developed the web application with a Flask backend framework and html, css, and js for the frontend. We used Spacy to perform NLP on each word problem, identifying the subjects corresponding to the numerical quantities, the numerical quantities themselves, the operation involved in the problem, and the subtype of operation (for example, addition problems sometimes calculate the "total" of two values, however other times calculate "more than" of one value). Finally, we used PIL, and openai to create the final image from that we display back to the user.'}
    return render_template('project.html', project=project)

@app.route('/projects/object-tracker')
def projects_object_tracker():
    project = {'title':'Custom Object Tracking Algorithm',
               'inspiration':'fjdsklfsjksl',
               'what it does': 'jfdkslfjd',
               'how it works': 'jfkdslfj'}
    return render_template('project.html', project=project)

@app.route('/projects/RadarPaper')
def projects_radar_derivations():
    project = {'title':'Custom Object Tracking Algorithm',
               'inspiration':'fjdsklfsjksl',
               'what it does': 'jfdkslfjd',
               'how it works': 'jfkdslfj'}
    return render_template('project.html', project=project)

@app.route('/add/Lovehope36!!')
def add():
    return render_template('add_post.html')

@app.route('/addpost',methods=['POST'])
def addpost():
    title = request.form['title']
    blurb = request.form['blurb']
    content = request.form['content']

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO posts (title, blurb, content) VALUES (?, ?, ?)",
            (title, blurb, content)
            )

    conn.commit()
    conn.close()

    return redirect(url_for('index'))

# @app.route('/post/<int:post_id>')
# def project(project_id):
#     return render_template('post.html', post=post)

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)