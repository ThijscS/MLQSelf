# Project Proposal: Machine Learning Predictor of Doomscroll Susceptibility ("The Brainrot Tracker")

---

## 1. Introduction & Problem Statement

Compulsive use of algorithmic short-form video applications (e.g., TikTok, Instagram Reels) often stems from psychological states like cognitive fatigue, stress, or boredom. In data science, building a time-series model to predict the exact second a user launches an app results in extreme class imbalance, where the target label is 0 for over 99.9% of the day.

To overcome this, this project frames the problem as an **Early Warning System (EWS)**. Instead of predicting the precise moment of failure, the model predicts a user's **Susceptibility Window**. This identifies the behavioral, systemic, and physiological incubation states that occur in the 15 minutes leading up to a doomscrolling session.

---

## 2. Core Research Questions

1. Can physiological signals (heart rate metrics) acquired via ear-based photoplethysmography (PPG) detect the onset of cognitive down-regulation or boredom?
2. To what degree does adding high-frequency smartphone kinematic data (stillness) and system-level interaction data (unlock counts, session lengths) improve the precision of an early warning behavior model?

---

## 3. Data Collection Architecture

The dataset is built using a multi-modal tracking approach across three distinct data layers:

| Data Layer | Primary Sensor | Sampling Frequency | Operational Rationale |
| :--- | :--- | :--- | :--- |
| **Physiological** | AirPods Pro 3 (Ear PPG Sensor) | Continuous (~1 Hz) | Monitors autonomic nervous system transitions. Decreasing heart rate and flattened cardiac variance serve as proxies for diminished cognitive engagement and creeping boredom. |
| **Kinematic** | iPhone Accelerometer & Gyroscope | High Frequency (50 Hz) | Tracks micro-movements of the physical body and phone placement. Prolonged physical stillness indicates a state of sedentary stagnation, raising the baseline probability of phone retrieval. |
| **System & Target** | iOS System Shortcuts | Event-Driven (On Launch) | Automates the extraction of ground-truth timestamps whenever target applications are opened, as well as tracking basic device interactions (unlocks). |

---

## 4. Target Formulation: The 15-Minute Susceptibility Horizon

To create a balanced and mathematically viable classification target, the continuous time-series data is processed using a rolling 15-minute lookahead window:

```
                           [   15-Minute Warning Window   ]
  |------------------------|------------------------------|-------------------> Time
Normal Data State (Class 0)   Susceptible State (Class 1)   Event: TikTok Opened
```

- **Class 1 (Susceptible State):** All data samples falling within the 15-minute window immediately preceding a verified app-open event are assigned a target label of `1`.
- **Class 0 (Baseline State):** All other operational data points gathered during productive work, study, or resting periods are assigned a target label of `0`.

---

## 5. Feature Engineering Matrix

Raw data streams will be transformed into the following engineered features to optimize model performance:

### Physiological Features

- **`hr_delta_5m`:** The mathematical difference between the current heart rate reading and the trailing 5-minute rolling average. Negative values signify physical down-regulation.
- **`hr_variance_5m`:** The rolling variance of the heart rate over a 5-minute window. A low variance indicates a rigid, non-fluctuating pulse pattern, which heavily correlates with cognitive fatigue.

### Kinematic Features

- **`phone_stillness_duration`:** A running counter in seconds measuring how long the phone's composite accelerometer variance has remained below a strict static threshold.

### System & Interaction Features (The Digital Footprint)

- **`unlock_count_15m`:** The total number of times the phone was unlocked in the trailing 15-minute window. A high frequency of "micro-checks" indicates a fragmented attention span and rising dopamine cravings.
- **`screen_on_duration`:** A running counter tracking how long the screen has currently been active. If the screen has been on for 45 minutes of productive work, susceptibility to a distraction switch increases due to fatigue.

### Contextual and Temporal Anchors

- **`time_of_day_sin` / `time_of_day_cos`:** Cyclical trigonometric transformations of the 24-hour clock to map circadian vulnerabilities without creating a hard boundary discontinuity at midnight.
- **`time_since_last_scroll`:** A continuous timer mapping the time elapsed since the last recorded app closure, defining the behavioral refractory period of dopamine cravings.

---

## 6. Step-by-Step Data Collection Protocol

### Step 1: Setting Up Automated Target & System Logging on iOS

To log app-open events and system unlocks into a clean dataset without manual tracking:

1. Open the native **Shortcuts** app on your iPhone and navigate to the **Automation** tab.
2. **For App Opens (The Target):** Click the **+** icon and select **App**. Choose Instagram and TikTok, check **Is Opened**, and select **Run Immediately**.
3. In the action workspace, select **Add Action** and search for **Text**. Type the following template: `Current Date, Instagram_Open` (ensure the date variable is adjusted to display the exact timestamp down to the second).
4. Search for the **Append to Text File** action. Set it to append the output text into a file named `brainrot_tracker.csv` inside your Shortcuts folder on iCloud Drive.
5. **For System Unlocks (The Feature):** Create a second Automation. Instead of "App", select **Device → is Unlocked**. Follow the exact same steps, but append the text `Current Date, Phone_Unlocked` to the same CSV file.

### Step 2: Recording Continuous Heart Rate Data

1. Put on your AirPods Pro 3 and confirm a secure fit to maintain stable ear-canal PPG contact.
2. Open the **Strava** or native **Apple Fitness** app on your phone.
3. Start an indoor workout recording (e.g., "Indoor Workout" or "Yoga") right as you begin your study or work session. Allow this to run silently in the background to bypass standard background timeouts.
4. Maintain a stationary workspace posture to ensure your physical movements do not inject kinetic noise into the cardiovascular signal.

### Step 3: Recording High-Frequency Phone Motion Data

1. Position your iPhone consistently throughout the experiment (either flat on the desk beside you or securely inside your front pants pocket).
2. Open an app such as **Phyphox** or a sensor logger to capture raw internal Accelerometer and Gyroscope data at 50 Hz.
3. Keep the logger running concurrently during your structured work intervals.