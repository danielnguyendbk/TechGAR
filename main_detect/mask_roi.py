import argparse
import json
import cv2
import numpy as np
from pathlib import Path

class MaskROIBuilder:
    def __init__(self, window_name, image, initial_points=None, target_points=4):
        self.window_name = window_name
        self.image = image
        self.display = image.copy()
        self.points = initial_points or []
        self.target_points = target_points
        self.fixed_points = len(self.points)
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < self.target_points:
                self.points.append((x, y))
                self._draw()

    def _draw(self):
        self.display = self.image.copy()
        for i, pt in enumerate(self.points):
            color = (255, 0, 0) if i < self.fixed_points else (0, 0, 255)
            cv2.circle(self.display, pt, 4, color, -1)
            cv2.putText(self.display, f"P{i+1}", (pt[0]+5, pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        if len(self.points) > 1:
            for i in range(len(self.points) - 1):
                color = (255, 255, 0) if (i == 0 and self.fixed_points >= 2) else (0, 255, 0)
                cv2.line(self.display, self.points[i], self.points[i+1], color, 2)
                mid_x = (self.points[i][0] + self.points[i+1][0]) // 2
                mid_y = (self.points[i][1] + self.points[i+1][1]) // 2
                cv2.putText(self.display, f"Edge {i+1}", (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
        if len(self.points) == self.target_points:
            cv2.line(self.display, self.points[-1], self.points[0], (0, 255, 0), 2)
            mid_x = (self.points[-1][0] + self.points[0][0]) // 2
            mid_y = (self.points[-1][1] + self.points[0][1]) // 2
            cv2.putText(self.display, f"Edge {self.target_points}", (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            overlay = self.display.copy()
            pts = np.array(self.points, np.int32)
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.addWeighted(overlay, 0.3, self.display, 0.7, 0, self.display)

        cv2.imshow(self.window_name, self.display)

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self._draw()
        
        print(f"[{self.window_name}] Click to add points (need {self.target_points - self.fixed_points} more).")
        print(f"Press 'r' to reset (preserves fixed points). Press 'Enter' when done.")
        
        while True:
            key = cv2.waitKey(50) & 0xFF
            if key == 13 and len(self.points) == self.target_points:  # Enter
                break
            elif key == ord('r'):
                self.points = self.points[:self.fixed_points]
                self._draw()
                
        cv2.destroyWindow(self.window_name)
        return self.points

def get_frame(url):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video stream {url}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read frame from {url}")
    return frame

def load_transform(calibration_path):
    with open(calibration_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return np.array(data["camera_transforms"]["cam2"], dtype=np.float32)

def run_app(args):
    print("Reading frames from cameras...")
    cam1_frame = get_frame(args.cam1_url)
    cam2_frame = get_frame(args.cam2_url)
    
    # Cam 1: 4 points
    builder1 = MaskROIBuilder("CAM 1", cam1_frame, target_points=4)
    points1 = builder1.run()
    
    print("\n--- Handoff Configuration ---")
    edge1 = -1
    while edge1 < 1 or edge1 > 4:
        try:
            edge1 = int(input("Enter handoff Edge number for CAM 1 (1-4): "))
        except ValueError:
            pass
            
    # Project edge points to cam2
    p1_idx = edge1 - 1
    p2_idx = (edge1) % 4
    # Wait, the outward normal in cam1 points outward. If we project p2 -> p1 (reversed order), 
    # it becomes p1 -> p2 in cam2, so its outward normal in cam2 points properly away from cam1?
    # To keep polygon clockwise/consistent, let's reverse the order of points for cam2
    pt1 = points1[p2_idx] # reverse order so it's a shared edge with consistent winding
    pt2 = points1[p1_idx]
    
    transform_cam2_to_cam1 = load_transform(args.calibration)
    # To project cam1 -> cam2, we need inverse
    transform_cam1_to_cam2 = np.linalg.inv(transform_cam2_to_cam1)
    
    pts_cam1 = np.array([[[float(pt1[0]), float(pt1[1])], [float(pt2[0]), float(pt2[1])]]], dtype=np.float32)
    pts_cam2 = cv2.perspectiveTransform(pts_cam1, transform_cam1_to_cam2)[0]
    
    proj_pt1 = (int(round(pts_cam2[0][0])), int(round(pts_cam2[0][1])))
    proj_pt2 = (int(round(pts_cam2[1][0])), int(round(pts_cam2[1][1])))
    
    # Cam 2: starts with 2 projected points, user clicks 2 more
    print("\nProjected edge onto CAM 2 (Cyan Line). Please click 2 more points to complete the CAM 2 mask.")
    builder2 = MaskROIBuilder("CAM 2", cam2_frame, initial_points=[proj_pt1, proj_pt2], target_points=4)
    points2 = builder2.run()
    
    # In Cam 2, the handoff edge is exactly the first 2 points we injected!
    # So edge2 is always 1 (from P1 to P2)
    edge2 = 1

    # Save to JSON
    cam1_data = {
        "polygon": [{"x": p[0], "y": p[1]} for p in points1],
        "handoff_edge": edge1,
        "handoff_target": "cam2",
        "image_size": [cam1_frame.shape[1], cam1_frame.shape[0]]
    }
    
    cam2_data = {
        "polygon": [{"x": p[0], "y": p[1]} for p in points2],
        "handoff_edge": edge2,
        "handoff_target": "cam1",
        "image_size": [cam2_frame.shape[1], cam2_frame.shape[0]]
    }
    
    Path(args.save_mask_cam1).parent.mkdir(parents=True, exist_ok=True)
    with open(args.save_mask_cam1, "w", encoding="utf-8") as f:
        json.dump(cam1_data, f, indent=2)
        
    Path(args.save_mask_cam2).parent.mkdir(parents=True, exist_ok=True)
    with open(args.save_mask_cam2, "w", encoding="utf-8") as f:
        json.dump(cam2_data, f, indent=2)
        
    print(f"\nMasks saved to {args.save_mask_cam1} and {args.save_mask_cam2}")

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam1-url", required=True)
    parser.add_argument("--cam2-url", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--save-mask-cam1", required=True)
    parser.add_argument("--save-mask-cam2", required=True)
    return parser

if __name__ == "__main__":
    run_app(make_parser().parse_args())
