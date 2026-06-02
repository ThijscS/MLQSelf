
# Project Proposal: Machine Learning Predictor of Doomscroll Susceptibility ("The Brainrot Tracker")

---

## 1. Introduction & Problem Statement

Compulsive use of algorithmic short-form video applications (e.g., TikTok, Instagram Reels) often stems from psychological states like cognitive fatigue, stress, or boredom. In data science, building a time-series model to predict the exact second a user launches an app results in extreme class imbalance, where the target label is 0 for over 99.9% of the day.

To overcome this, this project frames the problem as an **Early Warning System (EWS)**. Instead of predicting the precise moment of failure, the model uses features aggregated from the current 3-minute time block to predict whether a user will enter a doomscrolling session in the *subsequent* 3-minute window.

---

## 2. Core Research Questions

1. To what degree can low-frequency smartphone kinematic data (stillness, movement intensity) predict an impending drop in user focus?
2. Does the inclusion of immediate system interaction history (app switching, frequency of other app opens) improve the precision of a short-horizon early warning behavioral model?

---

## 3. Data Collection Architecture

The dataset is built using a dual-modal tracking approach across two distinct data layers:

| Data Layer | Primary Sensor | Sampling Frequency | Operational Rationale |
| :--- | :--- | :--- | :--- |
| **Kinematic** | iPhone Accelerometer & Gyroscope | Low Frequency (5 Hz - 10 Hz) | Tracks macroscopic movements of the phone body. Low frequency cuts data overhead while maintaining enough resolution to capture quick pickup gestures. |
| **System & Target** | iOS System Shortcuts | Event-Driven (On Launch) | Automates the extraction of ground-truth timestamps whenever target apps (TikTok, Instagram) or auxiliary apps are opened. |

---

## 4. Target Formulation: The 3-Minute Lookahead Horizon

To build a predictive early warning system, features computed during the current 3-minute interval ($T$) are mapped to a target label derived from the subsequent 3-minute interval ($T+1$):



| Current Window (Time T) | → | Lookahead Window (Time T+1) |
|:---|:---:|:---|
| Extract Features | | Predictive Target: Did TikTok/Instagram Open? (0 or 1) |



- **Class 1 (Susceptible State):** Assigned to the current 3-minute interval ($T$) if a verified target app-open event occurs during the next interval ($T+1$).
- **Class 0 (Baseline State):** Assigned if no target app-open event occurs in the subsequent interval ($T+1$).

---

## 5. Feature Engineering Matrix

Raw data streams are aggregated into 3-minute tabular blocks to generate the following engineered features for XGBoost:

### Kinematic Features

- **`phone_stillness_duration`:** A running counter in seconds measuring how long the phone's composite gyroscope variance has remained below a strict static threshold.
- **`gyro_max_vector`:** The magnitude of the single largest angular velocity spike recorded within the 3-minute window, capturing rapid physical actions like picking up or tossing the device.
- **`gyro_variance`:** The statistical variance of the gyroscope signals over the 3-minute window to quantify user restlessness.

### System & Interaction Features

- **`other_apps_opened_count`:** The total number of non-target applications opened during the 3-minute window, acting as a proxy for attention fragmentation.
- **`app_switch_frequency`:** The rate of transition between different baseline applications within the current block.

### Contextual and Temporal Anchors

- **`time_of_day_sin` / `time_of_day_cos`:** Cyclical trigonometric transformations of the 24-hour clock to map circadian vulnerabilities without creating a hard boundary discontinuity at midnight.
- **`time_since_last_scroll`:** A continuous timer mapping the time elapsed since the last recorded target app closure, defining the behavioral refractory period of dopamine cravings.

---

## 6. Step-by-Step Data Collection Protocol

### Step 1: Setting Up Automated Target & System Logging on iOS

To log app-open events into a clean dataset without manual tracking:

1. Open the native **Shortcuts** app on your iPhone and navigate to the **Automation** tab.
2. **For Target Apps:** Click the **+** icon and select **App**. Choose **Instagram** and **TikTok**, check **Is Opened**, and select **Run Immediately**.
3. In the action workspace, select **Add Action** and search for **Text**. Type the template: `Current Date, Target_Open`.
4. Search for the **Append to Text File** action. Set it to append the output text into a file named `brainrot_tracker.csv` inside your Shortcuts folder on iCloud Drive.
5. **For Auxiliary Apps:** Create identical automations for other high-frequency apps (e.g., Messages, Mail, Safari) that write to the same CSV file with the format `Current Date, Aux_Open` to populate the attention fragmentation features.

### Step 2: Recording Background Phone Motion Data

1. Download a sensor logging application that supports persistent background recording threads (e.g., **Sensor Logger**).
2. Configure the app settings to log the **Gyroscope** and **Accelerometer** at a hardware sampling frequency of **5 Hz** or **10 Hz**.
3. Enable **Background Recording** within the application settings, granting "Always Allow" location permissions if prompted (this forces iOS to keep the sensor polling thread active when the screen is locked or another app is active).
4. Manually start the logging session at the beginning of a designated study or work interval, and export the resulting continuous CSV at the end of the session to align with the Shortcuts event logs via timestamps during Python preprocessing.

