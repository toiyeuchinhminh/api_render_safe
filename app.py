import os
import time
import base64
import csv
from flask import Flask, request, render_template, send_file
from groq import Groq
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Mã hóa ảnh sang base64
def encode_image(file):
    return base64.b64encode(file.read()).decode('utf-8')

# Gửi ảnh đến Groq API
def predict_image(base64_image, question):
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Lỗi: {e}"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("images")
    question = request.form.get("question", "")
    limit = int(request.form.get("limit", 20))

    results = []
    processed = 0

    with open("output.csv", mode="w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Tên ảnh", "Nhãn mẫu", "Kết quả dự đoán AI", "Thời gian xử lý (giây)"])

        for file in files:
            if processed >= limit:
                break

            filename = file.filename
            start_time = time.time()
            base64_img = encode_image(file)
            prediction = predict_image(base64_img, question)
            end_time = time.time()
            duration = round(end_time - start_time, 2)

            label = "normal" if "NORMAL" in filename.upper() else "pneumonia"
            writer.writerow([filename, label, prediction, duration])
            results.append((filename, label, prediction, duration))
            processed += 1

    # Trả kết quả HTML và link tải file
    html = f"<h3>🧠 Câu hỏi đã gửi: <i>{question}</i></h3>"
    html += "<h4>Kết quả dự đoán:</h4><ul>"
    for name, label, pred, duration in results:
        html += f"<li><b>{name}</b> ({label}): {pred} ⏱ {duration} giây</li>"
    html += "</ul><a href='/'>← Quay lại</a> | <a href='/download'>📄 Tải file CSV</a>"

    return html

@app.route("/download")
def download_csv():
    return send_file("output.csv", as_attachment=True, mimetype="text/csv")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render sẽ truyền PORT qua biến môi trường
    app.run(host="0.0.0.0", port=port, debug=True)

