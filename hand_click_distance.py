import mediapipe as md
import cv2
import math
import numpy as np
from screeninfo import get_monitors
from pynput.mouse import Controller, Button
import time

class HandDistance():
    def __init__(self, 
        PATH_TO_OUTPUT_VIDEO: str='output_video_hand_click.mp4',
        MIN_DETECTION_CONFIDENCE: float=0.7,
        MIN_TRACKING_CONFIDENCE: float=0.6,
        MAX_NUM_HANDS: int=1,
        FRAME_WIDTH: int=1200,
        FRAME_HEIGHT: int=700,
        CIRCLE_RADIUS: int=4,
        MONITOR: int=0,
        THICKNESS: int=3,
        THICKNESS_TEXT: int=2,
        DISTANCE_FOR_CLICK: int=80,
        CLICK_COUNT: int=1,
        CLICK_COOLDOWN: float=1.0,
        MOUSE_BUTTON: str='left'
        ):
        BUTTON_DICT = {
            'left': Button.left,
            'right': Button.right,
            'middle': Button.middle,
            'x1': Button.x1,
            'x2': Button.x2
        }
        if MIN_DETECTION_CONFIDENCE > 1:
            raise ValueError('MIN_DETECTION_CONFIDENCE must be less than 1')
        if MIN_TRACKING_CONFIDENCE > 1:
            raise ValueError('MIN_TRACKING_CONFIDENCE must be less than 1')
        if MAX_NUM_HANDS > 20:
            raise ValueError('MAX_NUM_HANDS must be less than 20')
        if CIRCLE_RADIUS > 15:
            raise ValueError('CIRCLE_RADIUS must be less than 15')
        
        self.MONITOR = MONITOR
        SRC_WIDTH, SRC_HEIGHT = self.get_monitor_size()
        if FRAME_WIDTH > SRC_WIDTH or FRAME_HEIGHT > SRC_HEIGHT:
            raise ValueError('FRAME_WIDTH or FRAME_HEIGHT cant be that big')
        if THICKNESS > 10:
            raise ValueError('THICKNESS must be less than 10')
        self.THICKNESS_TEXT = THICKNESS_TEXT
        if self.THICKNESS_TEXT > 10:
            raise ValueError('THICKNESS_TEXT must be less than 10')
        self.DISTANCE_FOR_CLICK = round(DISTANCE_FOR_CLICK / 10) * 10
        if self.DISTANCE_FOR_CLICK > 130:
            raise ValueError('DISTANCE_FOR_CLICK must be less than 130')
        self.CLICK_COUNT = CLICK_COUNT
        if self.CLICK_COUNT > 3:
            raise ValueError('CLICK_COUNT must be less than 3')
        self.cooldown = CLICK_COOLDOWN
        if self.cooldown < 1:
            raise ValueError('CLICK_COOLDOWN must be more than 1')
        if MOUSE_BUTTON in BUTTON_DICT:
            self.BUTTON = BUTTON_DICT[MOUSE_BUTTON]
        else:
            raise ValueError('MOUSE_BUTTON must be only: left, right, middle, x1, x2')
                
        self.THICKNESS_HANDS = THICKNESS
        self.hands = md.solutions.hands
        self.drawing = md.solutions.drawing_utils
        self.hands_points = self.hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )
        self.drawing_canvas = None
        
        self.CLICK_CONTROLLE = Controller()
        self.level = 'WALK'
        self.last_click_time = time.time()
        
        self.CIRCLE_RADIUS = CIRCLE_RADIUS
        self.CAP = cv2.VideoCapture(self.MONITOR)
        self.VIDEO_PATH = PATH_TO_OUTPUT_VIDEO
        self.CAP.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.CAP.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.FOURCC = cv2.VideoWriter_fourcc(*'mp4v')
        self.WRITER = cv2.VideoWriter(self.VIDEO_PATH, self.FOURCC, 20.0, (FRAME_WIDTH, FRAME_HEIGHT))
        
    def get_hand_and_distance(self, frame) -> any:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands_points.process(rgb)
        
        if results.multi_hand_landmarks:
            for landmark in results.multi_hand_landmarks:
                self.drawing.draw_landmarks(
                    frame,
                    landmark,
                    self.hands.HAND_CONNECTIONS,  
                    self.drawing.DrawingSpec(color=(0, 0, 255), thickness=self.THICKNESS_HANDS, circle_radius=self.CIRCLE_RADIUS),
                    self.drawing.DrawingSpec(color=(255, 255, 255), thickness=self.THICKNESS_HANDS)
                )
                thumb_tip = landmark.landmark[self.hands.HandLandmark.THUMB_TIP]
                index_tip = landmark.landmark[self.hands.HandLandmark.INDEX_FINGER_TIP]
                
                h, w, _ = frame.shape
                thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)


                return thumb_x, thumb_y, index_x, index_y, landmark

        return None, None, None, None, None
    
    def get_monitor_size(self) -> tuple[int, int]:
        monitor = get_monitors()
        total_monitors = len(get_monitors())
        
        if self.MONITOR > total_monitors:
            raise ValueError(f'MONITORS must be less than {total_monitors}')
        
        main_monitor = monitor[self.MONITOR]
        WIDTH = main_monitor.width
        HEIGHT = main_monitor.height
        
        return WIDTH, HEIGHT
    
    def is_click(self, distance) -> bool:
        current_time = time.time()
        if current_time - self.last_click_time >= self.cooldown: 
            if self.DISTANCE_FOR_CLICK < distance:
                self.level = 'JUMP'
                self.last_click_time = current_time
                self.CLICK_CONTROLLE.click(self.BUTTON, self.CLICK_COUNT)
                return True
            else:
                self.level = 'WALK'
                return True
        else:
            return False
    
    def calc_time_by_distance(self, thumb_x, thumb_y, index_x, index_y):
        distance = math.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2) / 2
        
        mid_x = (thumb_x + index_x) // 2
        mid_y = (thumb_y + index_y) // 2

        return distance, mid_x, mid_y
    
    def run(self) -> any:
        while self.CAP.isOpened():
            ret, frame = self.CAP.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            if self.drawing_canvas is None:
                h, w = frame.shape[:2]
                self.drawing_canvas = np.zeros((h, w, 3), dtype=np.uint8)
            
            thumb_x, thumb_y, index_x, index_y, landmark = self.get_hand_and_distance(frame)

            if thumb_x is not None:
                distance, mid_x, mid_y = self.calc_time_by_distance(thumb_x, thumb_y, index_x, index_y)
                distance = round(distance / 10) * 10
                
                cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 0), 2)
                cv2.circle(frame, (thumb_x, thumb_y), self.CIRCLE_RADIUS, (0, 255, 0), -1)
                cv2.circle(frame, (index_x, index_y), self.CIRCLE_RADIUS, (0, 255, 0), -1)
                cv2.circle(frame, (mid_x, mid_y), self.CIRCLE_RADIUS, (255, 255, 0), -1)

                self.is_click(distance)
                cv2.putText(frame, f'{self.level} -> {distance}', 
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), self.THICKNESS_TEXT)

                self.drawing.draw_landmarks(frame, landmark, self.hands.HAND_CONNECTIONS)

            output_frame = cv2.addWeighted(frame, 0.9, self.drawing_canvas, 1.0, 0)
            self.WRITER.write(output_frame)
            cv2.imshow('Hand Tracking', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or cv2.getWindowProperty("Hand Tracking", cv2.WND_PROP_VISIBLE) < 1:
                break 
            
        self.CAP.release()
        self.WRITER.release()
        cv2.destroyAllWindows()
            
if __name__ == '__main__':
    hand = HandDistance()
    hand.run()