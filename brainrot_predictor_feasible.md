
# Project Proposal: Machine Learning Predictor of Doomscroll Susceptibility ("The Brainrot Tracker")

---

## 1. Introduction & Problem Statement

Compulsive use of algorithmic short-form video applications (e.g., TikTok, Instagram Reels) often stems from psychological states like cognitive fatigue, stress, or boredom. In data science, building a time-series model to predict the exact second a user launches an app results in extreme class imbalance, where the target label is 0 for over 99.9% of the day.

To overcome this, this project frames the problem as an **Early Warning System (EWS)**. Instead of predicting the precise moment of failure, the model uses features aggregated from the current 3-minute time block to predict whether a user will enter a doomscrolling session in the *subsequent* 3-minute window.

---

## 2. Core Research Questions

1. **Predictability:** Can we predict brainrot time intervals? That is, using smartphone kinematic data (stillness, movement intensity) and system-interaction history aggregated over the current window, can we predict whether the user will enter a doomscrolling session in the subsequent window?
2. **Transferability:** Does a learned algorithm transfer across people? If a model is trained on one user's data, does its predictive performance hold when applied to a different user who was not seen during training?

> Note: RQ2 directly satisfies the assignment's evaluation requirement that generalization be measured on *another person* (or a new recording of the same person), rather than by randomly sampling time points from a single dataset.

---

## 3. Data Collection Architecture

The dataset is collected by **all three project members**, each recording **continuously and in parallel** on their own iPhone from the start of the project until the final deadline. Three participants enable the cross-person (leave-one-person-out) evaluation that RQ2 requires. The dataset is built using a dual-modal tracking approach across two distinct data layers:

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

## 4b. Evaluation Strategy

Per the assignment, the evaluation must measure **generalization**, never random time-point sampling within one continuous dataset (adjacent 3-minute blocks are autocorrelated, so a random split leaks information).

- **RQ1 (Predictability):** Train and test on the *same* primary user, but split by **recording session / day**. Earlier recording sessions form the training set; held-out later sessions form the test set. This prevents temporal leakage between neighbouring windows.
- **RQ2 (Transferability):** Use a **leave-one-person-out** scheme across the three participants. Train on two users and test on the held-out third, whose data never appears in training. Performance is then compared against that user's own within-person baseline (a model trained and tested on them alone) to quantify the transfer gap.
- **Metrics:** Because Class 1 (susceptible) blocks are rare, report **PR-AUC, precision, recall, and F1** rather than raw accuracy, plus a confusion matrix. Establish a majority-class baseline for reference.

---

## 5. Feature Engineering Matrix

Raw data streams are aggregated into 3-minute tabular blocks to generate the following engineered features for XGBoost:

### Kinematic Features

- **`phone_stillness_duration`:** A running counter in seconds measuring how long the phone's composite gyroscope variance has remained below a strict static threshold.
- **`gyro_max_vector`:** The magnitude of the single largest angular velocity spike recorded within the 3-minute window, capturing rapid physical actions like picking up or tossing the device.
- **`gyro_variance`:** The statistical variance of the gyroscope signals over the 3-minute window to quantify user restlessness.

### System & Interaction Features

- **`other_apps_opened_count`:** The total number of non-target applications opened during the 3-minute window, acting as a proxy for attention fragmentation. Auxiliary opens are tagged into four categories — **Search** (LLM assistants such as Gemini and Claude, and web browsers), **Social** (e.g., WhatsApp, Snapchat), **Entertainment** (e.g., Netflix, YouTube), and **Other** (anything else) — yielding per-category open counts as finer-grained features.
- **`app_switch_frequency`:** The rate of transition between different app categories within the current block.

### Contextual and Temporal Anchors

- **`time_of_day_sin` / `time_of_day_cos`:** Cyclical trigonometric transformations of the 24-hour clock to map circadian vulnerabilities without creating a hard boundary discontinuity at midnight.
- **`time_since_last_target_open`:** A continuous timer mapping the time elapsed since the last recorded target app-open event (TikTok/Instagram). Because iOS Shortcuts can only log app *open* events ("Is Opened") and cannot reliably detect app *close*, this feature is anchored on opens rather than closures. It captures the behavioral refractory period between doomscrolling bouts.

---

## 6. Step-by-Step Data Collection Protocol

### Step 1: Setting Up Automated Target & System Logging on iOS

To log app-open events into a clean dataset without manual tracking:

1. Open the native **Shortcuts** app on your iPhone and navigate to the **Automation** tab.
2. **For Target Apps:** Click the **+** icon and select **App**. Choose **Instagram** and **TikTok**, check **Is Opened**, and select **Run Immediately**.
3. In the action workspace, select **Add Action** and search for **Text**. Type the template: `Current Date, Target_Open`.
4. Search for the **Append to Text File** action. Set it to append the output text into a file named `brainrot_tracker.csv` inside your Shortcuts folder on iCloud Drive.
5. **For Auxiliary Apps:** Create one automation **per category** (four total), each multi-selecting the apps that belong to it, writing to the same CSV file with a category-specific tag: `Current Date, Search_Open` (LLM assistants such as Gemini/Claude and browsers), `Current Date, Social_Open` (e.g., WhatsApp, Snapchat), `Current Date, Entertainment_Open` (e.g., Netflix, YouTube), and `Current Date, Other_Open` (anything else). The Phone app is deliberately excluded, as calling behaviour is not expected to correlate with scrolling onset.

### Step 2: Recording Background Phone Motion Data

1. Download a sensor logging application that supports persistent background recording threads (e.g., **Sensor Logger**).
2. Configure the app settings to log the **Gyroscope** and **Accelerometer** at a hardware sampling frequency of **5 Hz** or **10 Hz**.
3. Enable **Background Recording** within the application settings, granting "Always Allow" location permissions if prompted (this forces iOS to keep the sensor polling thread active when the screen is locked or another app is active).
4. Record **continuously** for the full duration of the study (not just designated work intervals), periodically exporting the resulting CSV to align with the Shortcuts event logs via timestamps during Python preprocessing.

