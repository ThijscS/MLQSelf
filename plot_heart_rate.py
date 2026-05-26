import xml.etree.ElementTree as ET
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

XML_PATH = "data/apple_health_export/export_cda.xml"
TARGET_DATE = "20260516"
NS = "urn:hl7-org:v3"
T = lambda tag: f"{{{NS}}}{tag}"

tree = ET.parse(XML_PATH)
root = tree.getroot()

timestamps = []
bpm_values = []

for obs in root.iter(T("observation")):
    code_el = obs.find(T("code"))
    if code_el is None or code_el.get("code") != "8867-4":
        continue

    low_el = obs.find(f".//{T('effectiveTime')}/{T('low')}")
    val_el = obs.find(T("value"))

    if low_el is None or val_el is None:
        continue

    ts_str = low_el.get("value", "")
    if not ts_str.startswith(TARGET_DATE):
        continue

    try:
        dt = datetime.strptime(ts_str[:14], "%Y%m%d%H%M%S")
    except ValueError:
        continue

    bpm = float(val_el.get("value"))
    timestamps.append(dt)
    bpm_values.append(bpm)

print(f"Found {len(timestamps)} heart rate readings on May 16, 2026")
if not timestamps:
    print("No data found.")
    exit()

print(f"Time range: {min(timestamps).strftime('%H:%M:%S')} – {max(timestamps).strftime('%H:%M:%S')}")
print(f"HR range:   {min(bpm_values):.0f} – {max(bpm_values):.0f} bpm")
print(f"Mean HR:    {sum(bpm_values)/len(bpm_values):.1f} bpm")

pairs = sorted(zip(timestamps, bpm_values))
timestamps, bpm_values = zip(*pairs)

fig, ax = plt.subplots(figsize=(13, 5))

ax.plot(timestamps, bpm_values, color="#e74c3c", linewidth=1.2, alpha=0.95)
ax.fill_between(timestamps, bpm_values, alpha=0.12, color="#e74c3c")

ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
plt.xticks(rotation=45)

ax.set_xlabel("Time (May 16, 2026)")
ax.set_ylabel("Heart Rate (bpm)")
ax.set_title(f"Heart Rate – Run on May 16, 2026  |  {len(timestamps)} readings  |  avg {sum(bpm_values)/len(bpm_values):.0f} bpm")
ax.grid(True, alpha=0.3)

for bpm_thresh, color, label in [
    (120, "#f1c40f", "Zone 2 (120 bpm)"),
    (150, "#e67e22", "Zone 3 (150 bpm)"),
    (170, "#c0392b", "Zone 4 (170 bpm)"),
]:
    ax.axhline(bpm_thresh, color=color, linestyle="--", linewidth=0.9, alpha=0.7, label=label)

ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig("heart_rate_may16.png", dpi=150)
print("Saved → heart_rate_may16.png")
plt.show()
