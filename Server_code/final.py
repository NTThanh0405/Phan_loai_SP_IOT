import os
import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from pyzbar.pyzbar import decode
from ultralytics import YOLO
import socket
import threading
import pandas as pd
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================
IMG_SIZE = 700
CAM_INDEX = 0
L_MIN_OVERRIDE = None

YOLO_MODEL_PATH = "E:/XLA/XLA/Server_code/YOLOV8s_Barcode_Detection.pt"

print("🔍 Loading YOLO model...")
model_yolo = YOLO(YOLO_MODEL_PATH)
print("✅ YOLO ready!")

# ============================================================
# TCP SERVER ESP32
# ============================================================
ESP32_CLIENT = None
ESP32_CONNECTED = False
ESP32_BARCODE = None
app = None

# DataFrame to store product information
columns = ['Index', 'Timestamp', 'Product Quality', 'Barcode Status', 'Barcode', 'Image Path']
df = pd.DataFrame(columns=columns)

def esp32_server_thread():
    global ESP32_CLIENT, ESP32_CONNECTED, app

    HOST, PORT = "0.0.0.0", 8888
    print(f"🔌 Server ready at {HOST}:{PORT}")

    while True:
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((HOST, PORT))
            server.listen(1)
        except:
            continue

        print("⏳ Waiting ESP32...")
        client, addr = server.accept()

        ESP32_CLIENT = client
        ESP32_CONNECTED = True

        print("✅ ESP32 connected:", addr)
        if app:
            app.log("ESP32 connected")
            app.set_esp_led(True)

        try:
            client.sendall(b"SERVER_OK\n")
        except:
            pass

        while True:
            try:
                raw = client.recv(1024)
                if not raw:
                    break

                msg = raw.decode().strip()
                print("< ESP32:", msg)

                # ----------- CAPTURE ESP32 BARCODE -------------
                global ESP32_BARCODE
                if "STATE:BAR:" in msg:
                    ESP32_BARCODE = msg.split("STATE:BAR:", 1)[1].strip()
                    if app:
                        app.log(f"[ESP32 BARCODE RECEIVED] {ESP32_BARCODE}")
                # ------------------------------------------------

                if app:
                    app.log(f"<ESP32> {msg}")

                if msg.startswith("STATE:SORT_DONE"):
                    app.pipeline_running = False
                    app.waiting_esp_done = False
                    app.log("✔ SORT DONE → Ready for new cycle")

            except:
                break

        print("⚠ ESP32 disconnected")
        ESP32_CONNECTED = False
        if app:
            app.set_esp_led(False)
            app.log("ESP32 disconnected")

        try:
            client.close()
            server.close()
        except:
            pass


def send_cmd_to_esp32(cmd):
    if ESP32_CONNECTED and ESP32_CLIENT:
        try:
            ESP32_CLIENT.sendall((cmd+"\n").encode())
            print("> Sent:", cmd)
        except:
            print("❌ Send error")
    else:
        print("⚠ ESP32 NOT CONNECTED →", cmd)


# ============================================================
# YOLO BARCODE DETECTION
# ============================================================
def detect_barcode_yolo(image):
    results = model_yolo.predict(image, conf=0.25, imgsz=640, verbose=False)
    out = []
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            out.append((x1, y1, x2 - x1, y2 - y1))
    return out


def try_light_decode(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    out = []
    for bc in decode(gray):
        out.append(bc.data.decode("utf-8"))
    return out


def detect_code(image):
    out = []
    boxes = detect_barcode_yolo(image)
    for (x, y, w, h) in boxes:
        roi = image[y:y + h, x:x + w]
        decoded = try_light_decode(roi)
        if decoded:
            out.append(("BAR", decoded[0], (x, y, w, h)))
        else:
            out.append(("BAR_UNKNOWN", None, (x, y, w, h)))
    return out


# ------------------------------------------------------------
# NEW FUNCTION TO EXTRACT BARCODE FOR EXCEL LOGGING
# ------------------------------------------------------------
def extract_barcode_from_codes(codes):
    for ctype, text, _ in codes:
        if ctype == "BAR" and text is not None:
            return text
    return None


# ============================================================
# ISOLATE BOX
# ============================================================
def isolate_box(image):
    global L_MIN_OVERRIDE

    h, w = image.shape[:2]
    image_roi = image

    hsv = cv2.cvtColor(image_roi, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))

    row_mean = mask_green.mean(axis=1)
    belt = np.where(row_mean > 20)[0]
    if belt.size == 0:
        return None, None

    belt_top, belt_bot = belt[0], belt[-1]

    non_green = cv2.bitwise_not(mask_green)

    L = cv2.cvtColor(image_roi, cv2.COLOR_BGR2LAB)[:, :, 0]
    thr = max(140, min(int(np.percentile(L, 80)), 230))
    if L_MIN_OVERRIDE is not None:
        thr = L_MIN_OVERRIDE

    bright = cv2.inRange(L, thr, 255)
    mask = cv2.bitwise_and(non_green, bright)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((20, 20))

    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    best = None
    best_score = -1
    img_area = w * h

    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh

        if not (img_area * 0.01 < area < img_area * 0.4):
            continue
        if not (0.6 < bw / bh < 1.6):
            continue

        overlap = max(0, min(y + bh, belt_bot) - max(y, belt_top))
        if overlap < bh * 0.5:
            continue

        if area > best_score:
            best_score = area
            best = (x, y, bw, bh)

    return best, mask


# ============================================================
# DAMAGE DETECT
# ============================================================
def detect_damage(image, box, codes):
    if box is None:
        return None

    x, y, w, h = box
    border = int(min(w, h) * 0.12)

    cx1, cy1 = x + border, y + border
    cx2, cy2 = x + w - border, y + h - border

    crop = image[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dark = (gray < gray.mean() - 40).astype(np.uint8)

    mask_bc = np.zeros_like(dark)

    for ctype, text, (bx, by, bw, bh) in codes:
        bx2, by2 = bx + bw, by + bh
        ix1 = max(bx, cx1) - cx1
        iy1 = max(by, cy1) - cy1
        ix2 = min(bx2, cx2) - cx1
        iy2 = min(by2, cy2) - cy1
        if ix1 < ix2 and iy1 < iy2:
            cv2.rectangle(mask_bc, (ix1, iy1), (ix2, iy2), 1, -1)

    dark[mask_bc == 1] = 0

    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    dx, dy, dw, dh = cv2.boundingRect(c)

    if dw * dh < (w * h * 0.05):
        return None

    return cx1 + dx, cy1 + dy, dw, dh


# ============================================================
# PIPELINE
# ============================================================
def analyze_box_pipeline(image):
    box, _ = isolate_box(image)
    if box is None:
        return "EMPTY", None, None, []

    codes = detect_code(image)
    dmg = detect_damage(image, box, codes)

    box_area = box[2] * box[3]

    if box_area < image.shape[0] * image.shape[1] * 0.1:
        return "EMPTY", box, dmg, codes

    if dmg:
        return "BAD", box, dmg, codes

    return "GOOD", box, None, codes


# ============================================================
# BUILD ESP32 MESSAGE
# ============================================================
def build_esp32_message(status, codes):
    bar = None
    for ctype, text, _ in codes:
        if ctype == "BAR":
            bar = text
        elif ctype == "BAR_UNKNOWN":
            bar = "BAR_UNKNOWN"

    if status == "EMPTY":
        return "EMPTY"

    if bar:
        if bar == "BAR_UNKNOWN":
            return status + "+BAR_UNKNOWN"
        return status + f"+BAR:{bar}"

    return status + "+NOBAR"


# ============================================================
# GUI CLASS
# ============================================================
class ImageApp:
    def __init__(self, root):
        global app
        app = self

        self.root = root
        self.root.title("Box Detector – YOLO + ESP32")

        self.calib_active = False
        self.pipeline_running = False
        self.waiting_esp_done = False

        main = tk.Frame(root)
        main.pack()

        left = tk.Frame(main, bd=2, relief="solid")
        left.grid(row=0, column=0)
        self.canvas = tk.Label(left)
        self.canvas.pack()

        right = tk.LabelFrame(main, text="Calibration", padx=10, pady=10)
        right.grid(row=0, column=1, padx=10)

        tk.Button(right, text="CALIB DONE", command=self.on_calib_done, width=20, bg="blue", fg="white").pack(pady=10)
        tk.Button(right, text="Reset System", command=self.reset_system, width=20, bg="red", fg="white").pack(pady=10)

        sliders = tk.Frame(right)
        sliders.pack()

        tk.Label(sliders, text="Brightness").grid(row=0, column=0)
        self.brightness = tk.DoubleVar(value=8)
        tk.Scale(sliders, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.brightness, length=200).grid(row=0, column=1)

        tk.Label(sliders, text="Exposure").grid(row=1, column=0)
        self.exposure = tk.DoubleVar(value=-2)
        tk.Scale(sliders, from_=-13, to=0, orient=tk.HORIZONTAL, variable=self.exposure, length=200).grid(row=1, column=1)

        tk.Label(sliders, text="L_min").grid(row=2, column=0)
        self.lmin = tk.IntVar(value=197)
        tk.Scale(sliders, from_=140, to=230, orient=tk.HORIZONTAL, variable=self.lmin, length=200, command=self.on_lmin_change).grid(row=2, column=1)

        logf = tk.LabelFrame(root, text="Log Output")
        logf.pack(fill=tk.X)
        self.log_text = tk.Text(logf, height=8, bg="black", fg="lime")
        self.log_text.pack(fill=tk.X)

        esp = tk.Frame(root)
        esp.pack(pady=5)
        tk.Label(esp, text="ESP32 Status: ").pack(side=tk.LEFT)
        self.esp_led = tk.Canvas(esp, width=20, height=20, bg="red")
        self.esp_led.pack(side=tk.LEFT)

        self.cap = cv2.VideoCapture(CAM_INDEX)
        self.running_webcam = True
        self.root.after(10, self.update_webcam)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.index = 1

    def reset_system(self):
        self.pipeline_running = False
        self.waiting_esp_done = False
        self.calib_active = False
        self.index = 1
        self.log_text.delete(1.0, tk.END)
        self.log("System has been reset and ready for a new cycle.")
        self.on_calib_done()

    def set_esp_led(self, ok):
        self.esp_led.config(bg="lime" if ok else "red")

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    def on_calib_done(self):
        self.calib_active = True
        send_cmd_to_esp32("CALIB_DONE")
        self.log("✔ CALIB DONE sent")

    def on_lmin_change(self, v):
        global L_MIN_OVERRIDE
        L_MIN_OVERRIDE = int(float(v))

    def update_webcam(self):
        if not self.running_webcam:
            return

        brightness = self.brightness.get()
        exposure = self.exposure.get()

        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)

        ret, frame = self.cap.read()
        if not ret:
            return self.root.after(10, self.update_webcam)

        status, box, dmg, codes = analyze_box_pipeline(frame)
        view = self.draw_result(frame.copy(), status, box, dmg, codes)
        self.display_image(view)

        if not self.calib_active:
            return self.root.after(10, self.update_webcam)

        if self.waiting_esp_done:
            return self.root.after(10, self.update_webcam)

        if self.pipeline_running:
            return self.root.after(10, self.update_webcam)

        if status != "EMPTY":
            self.pipeline_running = True
            self.log("📦 Object detected → waiting 3s...")
            self.root.after(3000, self.capture_final_frame)
            return self.root.after(10, self.update_webcam)

        self.root.after(10, self.update_webcam)

    def capture_final_frame(self):
        if not self.pipeline_running:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        status, box, dmg, codes = analyze_box_pipeline(frame)
        msg = build_esp32_message(status, codes)

        send_cmd_to_esp32(msg)
        self.log(f"➡ Sent final result to ESP32: {msg}")

        # -----------------------------------------
        # SAVE TO EXCEL WITH REAL BARCODE
        # -----------------------------------------
        if status != "EMPTY":
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            barcode_status = "BAR_UNKNOWN" if "BAR_UNKNOWN" in msg else \
                             "BAR" if "BAR" in msg else "NOBAR"

            # Extract actual barcode text
            global ESP32_BARCODE
            # Barcode camera
            cam_barcode = extract_barcode_from_codes(codes)

            # ƯU TIÊN barcode ESP32
            if ESP32_BARCODE:
                barcode_text = ESP32_BARCODE
                barcode_status = "BAR"
            else:
                barcode_text = cam_barcode if cam_barcode else ""
                if "BAR_UNKNOWN" in msg:
                    barcode_status = "BAR_UNKNOWN"
                elif barcode_text:
                    barcode_status = "BAR"
                else:
                    barcode_status = "NOBAR"    

            if not os.path.exists("images"):
                os.makedirs("images")

            img_name = f"product_{self.index}.png"
            img_path = f"images/{img_name}"
            cv2.imwrite(img_path, frame)

            df.loc[self.index] = [
                self.index,
                timestamp,
                "GOOD" if status == "GOOD" else "BAD",
                barcode_status,
                barcode_text if barcode_text else "",
                img_path
            ]

            df.to_excel("product_data.xlsx", index=False)
            self.index += 1

        if status == "EMPTY":
            self.pipeline_running = False
            self.waiting_esp_done = False
            self.root.after(10, self.update_webcam)
        else:
            self.waiting_esp_done = True

    def draw_result(self, img, status, box, dmg, codes):
        if box:
            x, y, w, h = box
            cv2.rectangle(img, (x, y), (x + w, y + h),
                          (0, 255, 0) if status == "GOOD" else (0, 0, 255), 3)

        if dmg:
            x, y, w, h = dmg
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(img, "DAMAGED", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        for ctype, text, (bx, by, bw, bh) in codes:
            color = (0, 255, 0) if ctype == "BAR" else (0, 255, 255)
            label = f"BAR:{text}" if text else "BAR UNKNOWN"
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), color, 2)
            cv2.putText(img, label, (bx, by - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(img, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                    (0, 255, 0) if status == "GOOD" else (0, 0, 255), 3)

        return img

    def display_image(self, img):
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pil.thumbnail((IMG_SIZE, IMG_SIZE))
        self.tk_img = ImageTk.PhotoImage(pil)
        self.canvas.config(image=self.tk_img)

    def on_close(self):
        self.running_webcam = False
        if self.cap:
            self.cap.release()
        self.root.destroy()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    threading.Thread(target=esp32_server_thread, daemon=True).start()

    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
