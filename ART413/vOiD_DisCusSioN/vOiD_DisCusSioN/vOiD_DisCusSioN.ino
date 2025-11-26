#include "arduino_secrets.h"

#include <LiquidCrystal.h>

// LCD pins
LiquidCrystal lcd(12, 11, 5, 4, 3, 2); // Pins nÃ©cessaires pour envoyer l'information Ã  MaxMSP (RS (RegisterSelect), EN (Enable), D4 (Data bit 4), D5, D6, D7)

//Ultrason
const int trigPin = 7;
const int echoPin = 6;

// LDR
const int ldrPin = A0;

// Variables
long duration;
int distance;
int lightLevel;

void setup() {
  Serial.begin(115200);
  // LCD
  lcd.begin(16,2);
  lcd.print("BoÃ®te du vide");
  // Ultrason
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
}

void loop() {
    // -- Lecture d'ultrason
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  duration = pulseIn(echoPin, HIGH);
  distance = duration * 0.034 / 2; // cm

  // -- Lecture LDR
  lightLevel = analogRead(ldrPin); // 0 - 1023

  // -- Envoyer Ã  MaxMSP
  Serial.print(distance);
  Serial.print(",");
  Serial.println(lightLevel);

  // -- Afficher LCD
  lcd.setCursor(0, 1);
  lcd.print("D:");
  lcd.print(distance);
  lcd.print("cm ");
  lcd.print("L:");
  lcd.print(lightLevel);

  delay(100);
}
