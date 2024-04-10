void turn_right(int speed = 125, float degree = 90, float diameter = 4.3, float length = 12.25, float k = 1, int cpr_righ = 1300, int cpr_left = 1300);
void turn_left(int speed = 125, float degree = 90, float diameter = 4.3, float length = 12.25, float k = 1, int cpr_righ = 1300, int cpr_left = 1300);
void driveway_front(int speed = 125, float distance = 30, float kp = 0.1, float diameter = 4.3, int cpr_right = 1300, int cpr_left = 1300);
void driveway_back(int speed = 125, float distance = 30, float kp = 0.1, float diameter = 4.3, int cpr_right = 1300, int cpr_left = 1300);

#include <WiFi.h>
#include <WiFiUdp.h>

const char* ssid = "ITMO-NTO-robot";
const char* password = "Pass-2503ntor";

WiFiUDP udp;
const int udpPort = 5005;
uint8_t buffer[50];

#define right_motor_speed 4
#define right_motor_en 32
const int speed_left = 0;
const int en_left = 2;

#define left_motor_speed 19
#define left_motor_en 18
const int speed_right = 1;
const int en_right = 3;

#define encoder_right_1 12
#define encoder_right_2 14
#define encoder_left_1 5
#define encoder_left_2 15

void setup() {
  Serial.begin(115200);
  delay(1000);

  WiFi.mode(WIFI_STA); //Optional
  WiFi.begin(ssid, password);
  Serial.println("\nConnecting");

  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(100);
  }

  Serial.println("\nConnected to the WiFi network");
  Serial.print("Local ESP32 IP: ");
  Serial.println(WiFi.localIP());

  udp.begin(udpPort);

  pinMode(left_motor_speed, OUTPUT);
  pinMode(left_motor_en, OUTPUT);
  pinMode(right_motor_speed, OUTPUT);
  pinMode(right_motor_en, OUTPUT);

  ledcSetup(speed_left, 500, 8);
  ledcSetup(speed_right, 500, 8);
  ledcSetup(en_left, 500, 8);
  ledcSetup(en_right, 500, 8);

  ledcAttachPin(left_motor_speed, speed_left);
  ledcAttachPin(right_motor_speed, speed_right);
  ledcAttachPin(left_motor_en, en_left);
  ledcAttachPin(right_motor_en, en_right);

  pinMode(encoder_right_1, INPUT);
  pinMode(encoder_right_2, INPUT);
  pinMode(encoder_left_1, INPUT);
  pinMode(encoder_left_2, INPUT);

  attachInterrupt(digitalPinToInterrupt(encoder_right_1), enc_right, CHANGE);
  attachInterrupt(digitalPinToInterrupt(encoder_left_1), enc_left, CHANGE);

}

volatile int stepEnc_right = 0;
volatile int stepEnc_left = 0;

char message[7] = {0, 0, 0, 0, 0, 0, 0};

void loop() {
  udp.parsePacket();
  if (udp.read(buffer, 50) > 0)
  {
    message[0] = (char)buffer[0];
    message[1] = (char)buffer[1];
    message[2] = (char)buffer[2];
    message[3] = (char)buffer[3];
    message[4] = (char)buffer[4];
    message[5] = (char)buffer[5];
    message[6] = (char)buffer[6];
    char charArray[4] = {message[1], message[2], message[3], '\0'};
    int speed_now = atoi(charArray);
    char charArray_dist[4] = {message[4], message[5], message[6], '\0'};
    int distance_now = atoi(charArray_dist);
    if (message[0] == '1') {
      //      Serial.println(123);
      driveway_front(speed_now, distance_now);
    }
    if (message[0] == '2') {
      //      Serial.println(123);
      driveway_back(speed_now, distance_now);
    }
    if (message[0] == '3') {
      //      Serial.println(123);
      turn_right(speed_now, distance_now);
    }
    if (message[0] == '4') {
      turn_left(speed_now, distance_now);
    }
    Serial.print("Server to client: ");
    Serial.print((char *)buffer);
    Serial.print(" ");
    Serial.print(speed_now);
    Serial.print(" ");
    Serial.println(distance_now);
  }

  delay(1000);
}

void driveway_front(int speed, float distance, float kp, float diameter, int cpr_right, int cpr_left) { // distance in sm
  float distance_degree = distance * cpr_right / (3.14 * diameter) ;

  ledcWrite(en_right, 0);
  ledcWrite(en_left, 0);

  stepEnc_right = 0;
  stepEnc_left = 0;
  while (abs(stepEnc_right) < distance_degree) {
    float error = stepEnc_right - stepEnc_left;
    Serial.println(error);
    float speed_left_d = speed - kp * error;
    float speed_right_d = speed + kp * error;

    ledcWrite(speed_right, speed_right_d);
    ledcWrite(speed_left, speed_left_d);
  }
  ledcWrite(speed_right, 255);
  ledcWrite(en_left, 255);
  ledcWrite(en_right, 255);
  ledcWrite(speed_left, 255);
  delay(300);
  ledcWrite(speed_right, 0);
  ledcWrite(en_left, 0);
  ledcWrite(en_right, 0);
  ledcWrite(speed_left, 0);
}

void driveway_back(int speed, float distance, float kp, float diameter, int cpr_right, int cpr_left) { // distance in sm
  float distance_degree = distance * cpr_right / (3.14 * diameter) ;

  ledcWrite(speed_right, 0);
  ledcWrite(speed_left, 0);

  stepEnc_right = 0;
  stepEnc_left = 0;
  while (abs(stepEnc_right) < distance_degree) {
    float error = stepEnc_right - stepEnc_left;
    Serial.println(error);
    float speed_left_d = speed - kp * error;
    float speed_right_d = speed + kp * error;

    ledcWrite(en_right, speed_right_d);
    ledcWrite(en_left, speed_left_d);
  }
  ledcWrite(speed_right, 255);
  ledcWrite(en_left, 255);
  ledcWrite(en_right, 255);
  ledcWrite(speed_left, 255);
  delay(300);
  ledcWrite(speed_right, 0);
  ledcWrite(en_left, 0);
  ledcWrite(en_right, 0);
  ledcWrite(speed_left, 0);
}

void turn_right(int speed, float degree, float diameter, float length, float k, int cpr_right, int cpr_left) { // distance in sm
  ledcWrite(speed_right, 0);
  ledcWrite(en_left, 0);

  stepEnc_right = 0;
  stepEnc_left = 0;
  float turn_cpr = degree * length * cpr_right / (360 * diameter) ;

  while (abs(stepEnc_right) < turn_cpr) {
    float error = (stepEnc_right / cpr_right) - (stepEnc_left / cpr_left);
    float error_ = k * error;
    float speed_left_d = speed - error_;
    float speed_right_d = speed + error_;
    ledcWrite(en_right, speed_right_d);
    ledcWrite(speed_left, speed_left_d);
  }
  ledcWrite(speed_right, 255);
  ledcWrite(en_left, 255);
  ledcWrite(en_right, 255);
  ledcWrite(speed_left, 255);
  delay(300);
  ledcWrite(speed_right, 0);
  ledcWrite(en_left, 0);
  ledcWrite(en_right, 0);
  ledcWrite(speed_left, 0);
}

void turn_left(int speed, float degree, float diameter, float length, float k, int cpr_right, int cpr_left) { // distance in sm
  ledcWrite(en_right, 0);
  ledcWrite(speed_left, 0);

  stepEnc_right = 0;
  stepEnc_left = 0;
  float turn_cpr = degree * length * cpr_right / (360 * diameter) ;

  while (abs(stepEnc_right) < turn_cpr) {
    float error = (stepEnc_right / cpr_right) - (stepEnc_left / cpr_left);
    float error_ = k * error;
    float speed_left_d = speed - error_;
    float speed_right_d = speed + error_;
    ledcWrite(speed_right, speed_right_d);
    ledcWrite(en_left, speed_left_d);
  }
  ledcWrite(speed_right, 255);
  ledcWrite(en_left, 255);
  ledcWrite(en_right, 255);
  ledcWrite(speed_left, 255);
  delay(300);
  ledcWrite(speed_right, 0);
  ledcWrite(en_left, 0);
  ledcWrite(en_right, 0);
  ledcWrite(speed_left, 0);
}


void enc_right() {
  stepEnc_right += digitalRead(encoder_right_2) == digitalRead(encoder_right_1) ? 1 : -1;
}

void enc_left() {
  stepEnc_left += digitalRead(encoder_left_2) == digitalRead(encoder_left_1) ? 1 : -1;
}
