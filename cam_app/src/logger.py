import datetime

def log_plate(plate_text, filename="plates.log"):
    timestamp = datetime.datetime.now().isoformat()
    with open(filename, "a") as f:
        f.write(f"{timestamp}, {plate_text}\n")
