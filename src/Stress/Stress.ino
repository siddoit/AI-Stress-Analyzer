#include <driver/i2s.h>
#include <math.h>
#include <DHT.h>
#include <Wire.h>
#include <BH1750.h>

// --- PINS (Standard VSPI - Most Reliable) ---
#define I2S_SCK 18
#define I2S_WS  19
#define I2S_SD  21

// --- I2C / LIGHT CONFIG (MOVED PINS!) ---
#define I2C_SDA 32 // Moved off 32 to avoid Mic collision
#define I2C_SCL 33 // Moved off 33 to avoid Mic collision
BH1750 lightMeter;

// --- DHT CONFIG ---
#define DHT_PIN 4
#define DHTTYPE DHT11
DHT dht(DHT_PIN, DHTTYPE);

// --- BUZZER CONFIG ---
#define BUZZER_PIN 22

// --- I2S CONFIG ---
#define I2S_PORT I2S_NUM_0

// --- VARIABLES ---
float db_offset = 10.0; 
unsigned long last_sensor_time = 0;
float last_temp = 0;
float last_hum = 0;
float last_lux = 0;

// --- BUZZER VARIABLES ---
unsigned long buzzer_start_time = 0;
bool is_buzzing = false;
const unsigned long BUZZ_DURATION = 2000; 

void setup() {
  Serial.begin(115200);
  Serial.println("\n--- FINAL CODE: SENSORY OVERLOAD ---");

  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW); 

  // 1. Start DHT
  dht.begin();

  // 2. Start Light Sensor
  pinMode(32, INPUT_PULLUP);
  pinMode(33, INPUT_PULLUP);
  Wire.begin(I2C_SDA, I2C_SCL);
  if (lightMeter.begin(BH1750::CONTINUOUS_HIGH_RES_MODE)) {
    Serial.println("BH1750 Ready.");
  } else {
    Serial.println("BH1750 Error. Check 21/19 wiring.");
  }

  // 3. Start I2S Mic
  const i2s_config_t i2s_config = {
    .mode = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = 16000,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT, 
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_I2S | I2S_COMM_FORMAT_I2S_MSB),
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 64,
    .use_apll = true, 
    .fixed_mclk = 0
  };  

  const i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = -1,
    .data_in_num = I2S_SD
  };

  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_PORT, &pin_config);
  
  Serial.println("System Ready.");
}

void loop() {
  // --- 0. BUZZER LISTENER ---
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim(); 
    
    if (cmd == "BUZZ") {
      is_buzzing = true;
      buzzer_start_time = millis();
      digitalWrite(BUZZER_PIN, HIGH);
    }
  }

  if (is_buzzing && (millis() - buzzer_start_time >= BUZZ_DURATION)) {
    is_buzzing = false;
    digitalWrite(BUZZER_PIN, LOW); 
  }

  // --- 1. READ MICROPHONE (Runs fast) ---
  int32_t samples[64];
  size_t bytes_read = 0;
  i2s_read(I2S_PORT, &samples, sizeof(samples), &bytes_read, portMAX_DELAY);

  float db = 0;
  if (bytes_read > 0) {
    float sum = 0;
    int samples_count = bytes_read / 4; 
    for (int i = 0; i < samples_count; i++) {
      int32_t val = samples[i] >> 14; 
      if (abs(val) > 20) sum += (float)(val * val);
    }
    float rms = sqrt(sum / samples_count);
    if (rms > 0) db = 20 * log10(rms) + db_offset;
  }

  // --- 2. READ DHT & LIGHT (Every 2 Seconds to prevent lag) ---
  if (millis() - last_sensor_time > 2000) {
    float t = dht.readTemperature();
    float h = dht.readHumidity();
    if (!isnan(t) && !isnan(h)) {
      last_temp = t;
      last_hum = h;
    }
    last_lux = lightMeter.readLightLevel();
    last_sensor_time = millis();
  }

  // --- 3. PRINT FOR PYTHON ---
  // Format: Sound: 45 dB | Temp: 28.5 C | Hum: 60 % | Light: 300 lx
  Serial.print("Sound: ");
  Serial.print(db, 0);
  Serial.print(" dB | Temp: ");
  Serial.print(last_temp, 1);
  Serial.print(" C | Hum: ");
  Serial.print(last_hum, 0);
  Serial.print(" % | Light: ");
  Serial.print(last_lux, 0);
  Serial.println(" lx");

  delay(100);
}