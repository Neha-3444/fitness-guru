from flask import Flask, render_template, request, url_for, redirect, session, Response
from pymongo import MongoClient
import bcrypt
import ollama
import os
import json
import numpy as np
from numpy.linalg import norm
import logging
import re
import cv2
import mediapipe as mp
import threading
import numpy as np


app = Flask(__name__)
app.secret_key = "testing"
camera = cv2.VideoCapture(0)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
class ShoulderPressEstimator:
    def __init__(self):
        self.rep_count = 0
        self.is_press_up = False
        self.is_press_down = True
        self.feedback_msgs = []

    def check_form(self, landmarks):
        feedback = []

        min_position_angle = 90  # Elbow angle at minimum position
        max_position_angle = 180  # Elbow angle at maximum position

        angle_right_elbow = self.calculate_angle(landmarks[11], landmarks[13], landmarks[15])
        angle_left_elbow = self.calculate_angle(landmarks[12], landmarks[14], landmarks[16])

        right_shoulder_y = landmarks[11].y
        left_shoulder_y = landmarks[12].y
        right_elbow_y = landmarks[13].y
        left_elbow_y = landmarks[14].y

        tolerance = 0.05

        if angle_right_elbow < min_position_angle:
            feedback.append("Right elbow should be perpendicular at the minimum position")
        if angle_left_elbow < min_position_angle:
            feedback.append("Left elbow should be perpendicular at the minimum position")
        if angle_right_elbow > max_position_angle or angle_left_elbow > max_position_angle:
            feedback.append("Extend arms fully at the top position")

        right_shoulder_wrist_dist = np.linalg.norm(np.array([landmarks[11].x, landmarks[11].y]) - np.array([landmarks[15].x, landmarks[15].y]))
        left_shoulder_wrist_dist = np.linalg.norm(np.array([landmarks[12].x, landmarks[12].y]) - np.array([landmarks[16].x, landmarks[16].y]))

        if right_shoulder_wrist_dist < 0.2 or left_shoulder_wrist_dist < 0.2:
            feedback.append("Keep arms away from your face")

        back_angle_tolerance = 35
        angle_back = self.calculate_angle(landmarks[11], landmarks[23], landmarks[25])
        if angle_back < (180 - back_angle_tolerance) or angle_back > (180 + back_angle_tolerance):
            feedback.append("Keep your back straight")

        if right_elbow_y > (right_shoulder_y + tolerance):
            feedback.append("Right elbow shouldn't go below shoulder")
        if left_elbow_y > (left_shoulder_y + tolerance):
            feedback.append("Left elbow shouldn't go below shoulder")

        press_up_angle_threshold = 150
        press_down_angle_threshold = 90

        if angle_right_elbow > (press_up_angle_threshold - tolerance) and angle_left_elbow > (press_up_angle_threshold - tolerance):
            if not self.is_press_up:
                self.is_press_up = True
                self.is_press_down = False
                self.feedback_msgs.append("Press up")
        elif angle_right_elbow < (press_down_angle_threshold + tolerance) and angle_left_elbow < (press_down_angle_threshold + tolerance):
            if not self.is_press_down and self.is_press_up:
                self.is_press_down = True
                self.is_press_up = False
                self.rep_count += 1
                self.feedback_msgs.append("Press down")
                self.feedback_msgs.append(f"Rep count: {self.rep_count}")

        return feedback, angle_right_elbow, angle_left_elbow

    def calculate_angle(self, a, b, c):
        a = [a.x, a.y, a.z]
        b = [b.x, b.y, b.z]
        c = [c.x, c.y, c.z]

        ba = [a[i] - b[i] for i in range(3)]
        bc = [c[i] - b[i] for i in range(3)]

        cosine_angle = sum(ba[i] * bc[i] for i in range(3)) / ((sum(ba[i] ** 2 for i in range(3)) ** 0.5) * (sum(bc[i] ** 2 for i in range(3)) ** 0.5))
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)

pose_estimator = ShoulderPressEstimator()

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.resize(frame, (1000, 700))
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(img_rgb)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark

                mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                feedback, angle_right_elbow, angle_left_elbow = pose_estimator.check_form(landmarks)

                cv2.putText(frame, f"Right Elbow: {int(angle_right_elbow)}", 
                            (int(landmarks[13].x * frame.shape[1]), int(landmarks[13].y * frame.shape[0] - 20)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(frame, f"Left Elbow: {int(angle_left_elbow)}", 
                            (int(landmarks[14].x * frame.shape[1]), int(landmarks[14].y * frame.shape[0] - 20)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                for i, msg in enumerate(feedback):
                    cv2.putText(frame, msg, (50, 150 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                for i, msg in enumerate(pose_estimator.feedback_msgs):
                    if "Rep count:" not in msg:
                        cv2.putText(frame, msg, (50, 50 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                pose_estimator.feedback_msgs.clear()

                if not feedback:
                    cv2.putText(frame, "Good form", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


##Connect with Docker Image###
def dockerMongoDB():
    client = MongoClient(host='localhost', port=27017)
    db = client.Fitness_guru
    records = db.register
    links = db.YT_links
    return records, links

records, links = dockerMongoDB()

# open a file and return paragraphs
def parse_file(filename):
    with open(filename, encoding="utf-8-sig") as f:
        paragraphs = []
        buffer = []
        for line in f.readlines():
            line = line.strip()
            if line:
                buffer.append(line)
            elif len(buffer):
                paragraphs.append((" ").join(buffer))
                buffer = []
        if len(buffer):
            paragraphs.append((" ").join(buffer))
        return paragraphs

def save_embeddings(filename, embeddings):
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    with open(f"embeddings/{filename}.json", "w") as f:
        json.dump(embeddings, f)

def load_embeddings(filename):
    if not os.path.exists(f"embeddings/{filename}.json"):
        return False
    with open(f"embeddings/{filename}.json", "r") as f:
        return json.load(f)

def get_embeddings(filename, modelname, chunks):
    if (embeddings := load_embeddings(filename)) is not False:
        return embeddings
    embeddings = [
        ollama.embeddings(model=modelname, prompt=chunk)["embedding"]
        for chunk in chunks
    ]
    save_embeddings(filename, embeddings)
    return embeddings

def find_most_similar(needle, haystack):
    needle_norm = norm(needle)
    similarity_scores = [
        np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack
    ]
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)

@app.route("/", methods=['post', 'get'])
def index():
    message = ''
    if "email" in session:
        return redirect(url_for("chatbot"))
    if request.method == "POST":
        user = request.form.get("fullname")
        email = request.form.get("email")
        password1 = request.form.get("password1")
        password2 = request.form.get("password2")
        user_found = records.find_one({"name": user})
        email_found = records.find_one({"email": email})
        if user_found:
            message = 'There already is a user by that name'
            return render_template('index.html', message=message)
        if email_found:
            message = 'This email already exists in database'
            return render_template('index.html', message=message)
        if password1 != password2:
            message = 'Passwords should match!'
            return render_template('index.html', message=message)
        else:
            hashed = bcrypt.hashpw(password2.encode('utf-8'), bcrypt.gensalt())
            user_input = {'name': user, 'email': email, 'password': hashed}
            records.insert_one(user_input)
            user_data = records.find_one({"email": email})
            new_email = user_data['email']
            return render_template('login.html', email=new_email)
    return render_template('index.html')

@app.route("/login", methods=["POST", "GET"])
def login():
    message = 'Please login to your account'
    if "email" in session:
        return redirect(url_for("chatbot"))
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        email_found = records.find_one({"email": email})
        if email_found:
            email_val = email_found['email']
            passwordcheck = email_found['password']
            if bcrypt.checkpw(password.encode('utf-8'), passwordcheck):
                session["email"] = email_val
                return redirect(url_for('chatbot'))
            else:
                if "email" in session:
                    return redirect(url_for("chatbot"))
                message = 'Wrong password'
                return render_template('login.html', message=message)
        else:
            message = 'Email not found'
            return render_template('login.html', message=message)
    return render_template('login.html', message=message)

@app.route('/logged_in')
def logged_in():
    if "email" in session:
        email = session["email"]
        return render_template('logged_in.html', email=email)
    else:
        return redirect(url_for("login"))

@app.route("/logout", methods=["POST", "GET"])
def logout():
    if "email" in session:
        session.pop("email", None)
        return render_template("signout.html")
    else:
        return render_template('index.html')

@app.route('/input',methods=["POST","GET"])
def input_v():
    return render_template("input_v.html")

# @app.route('/play', methods=['POST', 'GET'])
# def play_video():
#     exercise_name = request.args.get('exercise_name')
#     print(exercise_name)
#     # exercise_name = request.form.get('exercise_name')
#     # print("oahgahw",exercise_name)
#     if exercise_name:
#         link = links.find_one({"exercise_name": exercise_name})
#         print(link)
#         if link:
#             return render_template("Video_player.html", link=link["file_path"])
#     return render_template("VNF.html")

@app.route('/play', methods=['POST', 'GET'])
def play_video():
    exercise_names = request.args.get('exercise_names').split(',')
    current_index = int(request.args.get('current_index', 0))
    
    if current_index < len(exercise_names):
        exercise_name = exercise_names[current_index]
        link = links.find_one({"exercise_name": exercise_name})
        
        if "shoulder press" in exercise_name.lower() or "standing military press" in exercise_name.lower():
            return render_template('videohome.html',link=link["file_path"], exercise_names=exercise_names, current_index=current_index)
        if link:
            return render_template("video_player.html", link=link["file_path"], exercise_names=exercise_names, current_index=current_index)
        else:
            return render_template("VNF.html", exercise_names=exercise_names,current_index=current_index)
    else:
        return redirect(url_for('workout_complete'))

@app.route('/workout_complete')
def workout_complete():
    return render_template('workout_complete.html')


@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot(): 
    query_input = None
    response = None
    response_content = None
    exercise_names = None

    SYSTEM_PROMPT = """If you're unsure, just say that you don't know.
        The text or context that you have been provided is called as the 'fitness database'.
        Context:
    """
    filename = "workouts.txt"
    paragraphs = parse_file(filename)
    embeddings = get_embeddings(filename, "nomic-embed-text", paragraphs)
    if request.method == 'POST':
        query_input = request.form.get('query-input')
        prompt_embedding = ollama.embeddings(model="nomic-embed-text", prompt=query_input)["embedding"]
        most_similar_chunks = find_most_similar(prompt_embedding, embeddings)[:5]

        if query_input:
            try:
                response = ollama.chat(
                model="llama3",
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                        + "/n".join(paragraphs[item[1]] for item in most_similar_chunks),
                    },
                    {"role": "user", "content": query_input},
                ],
            )
                response_content = response["message"]["content"].replace("\n", "<br>")
                exercise_names_response = ollama.chat(
                    model="llama3",
                    messages=[
                        {
                            "role": "system",
                            "content": """Extract just the names of individual exercises, nothing else, just give me the names in an array format.
                            Don't write stuff like "Extracted Exercise Names: Here are the names of individual exercises:" or "Extracted Exercise Names:".
                            from the following response: """
                        },
                        {"role": "user", "content": response["message"]["content"]}
                    ],
                )
                exercise_names = exercise_names_response["message"]["content"]
                
                # Extract the exercise names array from the response content using regular expressions
                exercise_names_match = re.search(r"\[(.*?)\]", exercise_names, re.DOTALL)
                
                if exercise_names_match:
                    exercise_list = re.findall(r'"(.*?)"', exercise_names_match.group(0))
                    print(exercise_list)
                    exercise_names = ",".join(exercise_list)

                output = "No valid exercise found in the database."
            except Exception as e:
                logging.error(f"Error during chatbot invocation: {e}")
                output = "Sorry, an error occurred while processing your request."
    return render_template('chatbot.html', query_input=query_input, output=response_content,exercise_names=exercise_names)

# @app.route('/videohome')
# def videohome():
#     return render_template('videohome.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
  app.run(debug=True, host='0.0.0.0', port=50)