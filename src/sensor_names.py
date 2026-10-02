"""Human-readable names/categories for the C-MAPSS sensors, taken from
NASA's own documentation for this dataset. Used only to make feature
importance easier to read (e.g. "Temperature" instead of "s2_rmean_5").

Note: this dataset has no vibration sensor -- the 21 sensors are all
temperature, pressure, speed, and flow-ratio measurements from the
simulated engine's thermodynamic model.
"""

SENSOR_INFO = {
    "s1":  ("Fan inlet temperature", "Temperature"),
    "s2":  ("LPC outlet temperature", "Temperature"),
    "s3":  ("HPC outlet temperature", "Temperature"),
    "s4":  ("LPT outlet temperature", "Temperature"),
    "s5":  ("Fan inlet pressure", "Pressure"),
    "s6":  ("Bypass-duct pressure", "Pressure"),
    "s7":  ("HPC outlet pressure", "Pressure"),
    "s8":  ("Physical fan speed", "Speed"),
    "s9":  ("Physical core speed", "Speed"),
    "s10": ("Engine pressure ratio", "Pressure"),
    "s11": ("HPC outlet static pressure", "Pressure"),
    "s12": ("Fuel flow ratio", "Fuel Flow"),
    "s13": ("Corrected fan speed", "Speed"),
    "s14": ("Corrected core speed", "Speed"),
    "s15": ("Bypass ratio", "Other"),
    "s16": ("Burner fuel-air ratio", "Fuel Flow"),
    "s17": ("Bleed enthalpy", "Other"),
    "s18": ("Demanded fan speed", "Speed"),
    "s19": ("Demanded corrected fan speed", "Speed"),
    "s20": ("HPT coolant bleed", "Other"),
    "s21": ("LPT coolant bleed", "Other"),
}


def base_sensor(feature_name: str) -> str | None:
    """Given an engineered feature column like 's4_rmean_20' or
    's12_slope_20', return the base sensor id 's4'/'s12'. Returns None
    for non-sensor columns (e.g. op_setting_1)."""
    for sensor_id in SENSOR_INFO:
        if feature_name == sensor_id or feature_name.startswith(sensor_id + "_"):
            return sensor_id
    return None


def category_of(feature_name: str) -> str:
    sensor_id = base_sensor(feature_name)
    if sensor_id is None:
        return "Other"
    return SENSOR_INFO[sensor_id][1]
