import time
import pyautogui


def log_movement():
    while True:
        x, y = pyautogui.position()
        timestamp = round(time.monotonic(), 2)

        print(f"x: {x}, y: {y}, time: {timestamp}")
        pyautogui.time.sleep(0.2)


if __name__ == "__main__":
    log_movement()
