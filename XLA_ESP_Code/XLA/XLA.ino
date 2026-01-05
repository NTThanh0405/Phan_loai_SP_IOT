#include <WiFi.h>
#include <SoftwareSerial.h>
#include <ESP32Servo.h>

// ---------------- WIFI ----------------
const char* ssid = "POCO 2G";
const char* password = "tu0dentam";
const char* serverIP = "10.71.170.52";
const int serverPort = 8888;

WiFiClient client;

// ---------------- GM65 ----------------
SoftwareSerial gm65(14, 13);  // RX, TX
String gmData = "";
bool waitingGM65 = false;
bool gmDetected = false;
String detectedBarcode = ""; // Store detected barcode data
bool barcodeRead = false;    // Flag to track if barcode has already been read

// ---------------- SERVO ----------------
Servo servo1;   // BAD + NO BAR
Servo servo2;   // GOOD/BAD + BAR

#define SERVO1_PIN 12
#define SERVO2_PIN 15

// ---------------- CONVEYOR ----------------
#define CONVEYOR_PIN 4
#define OB_PIN 2
// FLAGS
bool calibrated = false;
bool busy = false;

// ================================================================
void sendState(String s) {
  if (client.connected()) {
    client.println("STATE:" + s);
  }
}

// ================================================================
void setup() {
  Serial.begin(115200);
  gm65.begin(9600);  // Initialize GM65 Serial

  servo1.attach(SERVO1_PIN);
  servo2.attach(SERVO2_PIN);
  pinMode(OB_PIN, INPUT);

  pinMode(CONVEYOR_PIN, OUTPUT);
  digitalWrite(CONVEYOR_PIN, HIGH);

  // WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(200);
    Serial.print(".");
  }
  Serial.println("Connected to WiFi");

  // Connect to server
  while (!client.connect(serverIP, serverPort)) {
    delay(200);
    Serial.print(".");

  }
  client.println("ESP32_CONNECTED");
  Serial.println("Connected to server");
}

// ================================================================
void loop() {
  // ---------------- READ SERVER ----------------
  if (client.available() && !busy) {
    String cmd = client.readStringUntil('\n');
    cmd.trim();

    Serial.println("From Server: " + cmd);

    if (cmd == "CALIB_DONE") {
      calibrated = true;
      sendState("CALIB_OK");
      return;
    }

    if (!calibrated)
      return;

    busy = true;
    handleCommand(cmd);
  }
}

// ================================================================
void handleCommand(String cmd) {
  if (cmd == "EMPTY") {
    digitalWrite(CONVEYOR_PIN, HIGH);  // Ensure conveyor is stopped
    busy = false;
    return;
  }

  // START conveyor
  digitalWrite(CONVEYOR_PIN, LOW);  // Conveyor starts moving
  sendState("START_SORT");

  bool good = cmd.startsWith("GOOD");
  bool bad = cmd.startsWith("BAD");

  bool hasBarPython = cmd.indexOf("BAR:") > 0;
  bool barUnknown = cmd.indexOf("BAR_UNKNOWN") > 0;
  bool noBarPython = cmd.indexOf("NOBAR") > 0;

  bool hasBarFinal = hasBarPython;

  // --------------------------- FALLBACK WHEN BAR_UNKNOWN ---------------------------
  if (barUnknown) 
  {
    sendState("FALLBACK_GM65");
    while(digitalRead(OB_PIN) == 1)
    {
      delay(1);
    }
    if(digitalRead(OB_PIN) == 0)
    {
      digitalWrite(CONVEYOR_PIN, 1);
      unsigned long startMillis = millis();
      unsigned long timeout = 2500;  // Timeout duration: 1.5 seconds

      gmDetected = false;  // Reset GM65 detection status
      detectedBarcode = ""; // Reset detected barcode data

      // Wait for GM65 response within 1.5 seconds
      while (millis() - startMillis < timeout) 
      {
        // ---------------- READ GM65 ----------------
        if (gm65.available() && !barcodeRead) 
        {
          String gmBarcode = gm65.readString();
          delay(500);
          digitalWrite(CONVEYOR_PIN, 0);
          if (!gmBarcode.isEmpty() && !barcodeRead)  // Check if barcode is read and flag is not set
          {
            gmDetected = true;
            detectedBarcode = gmBarcode;
            sendState("BAR:" + detectedBarcode);  // Send detected barcode to server
            hasBarFinal = true;
            barcodeRead = true;  // Set flag to prevent reading again
            break;
          }
        }
      }
      
      if (!gmDetected) 
      {
        Serial.println("GM65 FAILED");
        sendState("BAR:GM65 FAILED");
        hasBarFinal = false;
      }
    }
  }

  // NOBAR → Don't fallback
  if (noBarPython) {
    digitalWrite(CONVEYOR_PIN, LOW);
    delay(700);
    hasBarFinal = false;
  }

  // ================= CLASSIFICATION ====================
  digitalWrite(CONVEYOR_PIN, LOW);
  if (good && hasBarFinal) {
    sendState("GOOD_BAR");
    delay(4000);
  }
  else if (good && !hasBarFinal) {
    sendState("GOOD_NOBAR");
    delay(1900);
    servo2.write(145);
    delay(400);
    servo2.write(0);
  }
  else if (bad && hasBarFinal) {
    sendState("BAD_BAR");
    delay(2000);
    servo2.write(145);
    delay(1500);
    servo2.write(0);
  }
  else if (bad && !hasBarFinal) {
    sendState("BAD_NOBAR");
    delay(700);
    servo1.write(145);
    delay(1500);
    servo1.write(0);
  }

  // -------------------- After processing, restart conveyor --------------------
  digitalWrite(CONVEYOR_PIN, HIGH);  // Stop conveyor after task is complete
  sendState("SORT_DONE");

  // Reset flags for the next cycle
  busy = false;
  barcodeRead = false;  // Reset barcodeRead flag for next cycle
} 
