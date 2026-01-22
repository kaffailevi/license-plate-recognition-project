import time

class PlateTracker:
    def __init__(self, cooldown=3.0):
        self.last_seen = {}
        self.cooldown = cooldown

    def should_log(self, plate_text, score):
        now = time.time()
        last = self.last_seen.get(plate_text, 0)

        if now - last >= self.cooldown and score > 0.85:
            self.last_seen[plate_text] = now
            return True
        return False
