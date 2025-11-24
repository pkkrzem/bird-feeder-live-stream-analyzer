import cv2
import sys
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass
import threading
import time
import queue
import os
import csv
from datetime import datetime
from pathlib import Path
import json

# NEW: For iNaturalist model
try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification, EfficientNetImageProcessor, EfficientNetForImageClassification
    import torch
    from PIL import Image
    TRANSFORMERS_AVAILABLE = True
    print("✓ Transformers library loaded for bird identification model")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers library not found. Install with: pip install transformers torch pillow")
    print("   Bird identification will use placeholder mode.")

# yt-dlp -g YOUTUBE_LIVE_URL
# yt-dlp -f "bestvideo[height<=720]+bestaudio/best[height<=720]" -g YOUTUBE_LIVE_URL


# --- Configuration ---
@dataclass
class AppConfig:
    MODEL_PATH_TENSORRT: str = 'yolov8s.engine'
    MODEL_PATH_PYTORCH: str = 'yolov8s.pt'
    CONFIDENCE_THRESHOLD: float = 0.20  # Lowered for small birds
    IOU_THRESHOLD: float = 0.5
    INFERENCE_SIZE: int = 640  # Increased back to 640 for better small object detection
    DEBUG: bool = False
    MIN_BBOX_AREA: int = 400  # Reduced from 800 to catch smaller birds
    MAX_BBOX_AREA: int = 150000
    MAX_ASPECT_RATIO: float = 4.5
    IOU_MATCHING_THRESHOLD: float = 0.3  # Increased for better tracking
    DISPLAY_FPS: int = 30  # Target FPS for the display window
    PROCESS_TARGET_FPS: int = 2  # Reduced due to higher inference size
    WINDOW_NAME: str = "Live Bird Feeder Cam"
    CONFIRM_COLOR: tuple = (0, 255, 0)
    TRACKER_COLOR: tuple = (0, 165, 255)

    # Improved timing
    CAPTURE_FPS_LIMIT: int = 15  # Increased from 10 for smoother video
    DISPLAY_FRAME_TIME: float = 1.0 / 30  # 30 FPS display target

    # New: Multi-scale detection settings
    USE_MULTI_SCALE: bool = True  # Enable multi-scale detection for small birds
    SCALES: list = None  # Will be set in __post_init__
    SMALL_BIRD_CONFIDENCE: float = 0.15  # Even lower threshold for small detections
    ENHANCE_CONTRAST: bool = True  # Enhance image for better small bird detection

    ID_TRIGGER_HITS: int = 10
    ID_HIGH_CONFIDENCE: float = 0.7
    ID_STABLE_TIME: float = 3.0
    ID_MIN_SPECIES_CONFIDENCE: float = 0.6
    ID_MAX_ATTEMPTS: int = 3
    ID_COOLDOWN_MINUTES: int = 5
    PHOTO_SAVE_PATH: str = "bird_photos"
    LOG_FILE_PATH: str = "bird_log.csv"
    ENABLE_BIRD_ID: bool = True

    def __post_init__(self):
        if self.SCALES is None:
            self.SCALES = [640, 832]  # Multiple inference sizes


# --- Helper Functions ---
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


def xywh_to_xyxy(xywh):
    x, y, w, h = xywh
    return (x, y, x + w, y + h)


def calculate_frame_quality(frame_crop, bbox_area):
    """Calculate quality score for a bird photo crop"""
    if frame_crop is None or frame_crop.size == 0:
        return 0.0

    # Convert to grayscale for analysis
    gray = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY)

    # 1. Sharpness (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(laplacian_var / 1000.0, 1.0)  # Normalize

    # 2. Contrast (standard deviation)
    contrast_score = min(gray.std() / 50.0, 1.0)  # Normalize

    # 3. Brightness (avoid too dark or too bright)
    brightness = gray.mean()
    brightness_score = 1.0 - abs(brightness - 128) / 128.0

    # 4. Size score (larger birds get higher scores)
    size_score = min(bbox_area / 10000.0, 1.0)  # Normalize to reasonable bird size

    # Combined score
    quality_score = (sharpness_score * 0.4 +
                     contrast_score * 0.3 +
                     brightness_score * 0.2 +
                     size_score * 0.1)

    return quality_score


# --- NEW: Bird Identification Classes ---
class BirdIdentifier:
    """Handles bird species identification using iNaturalist model"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.model_loaded = False
        self.recent_identifications = {}  # For cooldown tracking
        self.processor = None
        self.model = None
        self.device = None

        # Create photo directory
        Path(self.config.PHOTO_SAVE_PATH).mkdir(exist_ok=True)

        # Initialize model
        self._load_model()

    def _load_model(self):
        """Load bird-specific identification model"""
        if not TRANSFORMERS_AVAILABLE:
            print("Using placeholder bird identification (install transformers for real model)")
            self._load_placeholder_model()
            return

        try:
            print("Loading bird-specific EfficientNet model...")

            # Use the bird-specific EfficientNet model (99.1% accuracy on 525 species)
            model_name = "dennisjooo/Birds-Classifier-EfficientNetB2"

            print(f"Downloading model: {model_name}")
            print("This may take a few minutes on first run...")

            # Load processor and model
            self.processor = EfficientNetImageProcessor.from_pretrained(model_name)
            self.model = EfficientNetForImageClassification.from_pretrained(model_name)

            # Set device (GPU if available, otherwise CPU)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()

            print(f"✓ Bird classification model loaded on {self.device}")
            print(f"✓ Model can identify {len(self.model.config.id2label)} bird species")

            self.model_loaded = True

        except Exception as e:
            print(f"Failed to load bird identification model: {e}")
            print("Falling back to placeholder model...")
            self._load_placeholder_model()

    def _load_placeholder_model(self):
        """Load placeholder model for demonstration"""
        print("Bird identification model: Using placeholder (limited accuracy)")
        self.model_loaded = True

        # Common North American backyard birds
        self.demo_species = [
            "American Robin", "Blue Jay", "Northern Cardinal", "House Sparrow",
            "European Starling", "Mourning Dove", "Red-winged Blackbird",
            "House Finch", "American Goldfinch", "Black-capped Chickadee",
            "White-breasted Nuthatch", "Downy Woodpecker", "Carolina Wren",
            "Tufted Titmouse", "Song Sparrow", "Dark-eyed Junco",
            "Cedar Waxwing", "Ruby-throated Hummingbird", "Barn Swallow",
            "Rock Pigeon", "Common Grackle", "Baltimore Oriole"
        ]

    def _preprocess_image(self, image_crop):
        """Preprocess image for bird identification model"""
        if image_crop is None or image_crop.size == 0:
            return None

        try:
            # Convert BGR (OpenCV) to RGB (PIL)
            image_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            if TRANSFORMERS_AVAILABLE and self.processor:
                # Use the model's processor for proper preprocessing
                inputs = self.processor(pil_image, return_tensors="pt")
                return inputs.to(self.device)
            else:
                # Fallback preprocessing for placeholder
                return cv2.resize(image_crop, (224, 224))

        except Exception as e:
            if self.config.DEBUG:
                print(f"DEBUG: Error preprocessing image: {e}")
            return None

    def identify_species(self, image_crop, track_id):
        """Identify bird species from image crop using bird-specific model"""
        if not self.model_loaded or image_crop is None:
            return None, 0.0

        if not TRANSFORMERS_AVAILABLE:
            return self._identify_species_placeholder(image_crop, track_id)

        try:
            # Preprocess image
            inputs = self._preprocess_image(image_crop)
            if inputs is None:
                return None, 0.0

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

                # Get top prediction
                top_prediction = torch.topk(predictions, k=1)
                predicted_class_idx = top_prediction.indices[0][0].item()
                confidence = top_prediction.values[0][0].item()

                # Get species name from model's label mapping
                species_name = self.model.config.id2label[predicted_class_idx]

                # Clean up species name (remove codes, format nicely)
                species_name = self._clean_species_name(species_name)

                if self.config.DEBUG:
                    print(
                        f"DEBUG: Bird model identified Track {track_id} as '{species_name}' (confidence: {confidence:.3f})")

                return species_name, confidence

        except Exception as e:
            if self.config.DEBUG:
                print(f"DEBUG: Bird species identification failed: {e}")
            return None, 0.0

    def _clean_species_name(self, raw_name):
        """Clean up species name from model output"""
        if not raw_name:
            return "Unknown Bird"

        # Remove common prefixes/suffixes and format nicely
        cleaned = raw_name.replace("_", " ").replace("-", " ")

        # Capitalize each word
        cleaned = " ".join(word.capitalize() for word in cleaned.split())

        # Handle common formats
        if cleaned.startswith("Bird "):
            cleaned = cleaned[5:]  # Remove "Bird " prefix

        return cleaned

    def _identify_species_placeholder(self, image_crop, track_id):
        """Placeholder identification for demo purposes"""
        import random

        # Simulate processing time
        time.sleep(0.1)

        # Simulate identification with varying confidence based on image quality
        quality = calculate_frame_quality(image_crop, image_crop.shape[0] * image_crop.shape[1])

        # Higher quality images get higher confidence
        base_confidence = 0.3 + (quality * 0.6)
        confidence = random.uniform(base_confidence * 0.8, base_confidence * 1.2)
        confidence = min(0.95, max(0.1, confidence))

        # Choose species based on confidence (more common birds for higher confidence)
        if confidence > 0.8:
            species = random.choice(self.demo_species[:8])  # Common birds
        elif confidence > 0.6:
            species = random.choice(self.demo_species[:15])  # Moderately common
        else:
            species = random.choice(self.demo_species)  # Any bird

        if self.config.DEBUG:
            print(f"DEBUG: Placeholder identified Track {track_id} as {species} (confidence: {confidence:.3f})")

        return species, confidence

    def should_identify(self, track_id, bbox_center):
        """Check if we should identify this bird (cooldown logic)"""
        current_time = time.time()

        # Clean old entries
        cutoff_time = current_time - (self.config.ID_COOLDOWN_MINUTES * 60)
        self.recent_identifications = {
            k: v for k, v in self.recent_identifications.items()
            if v['time'] > cutoff_time
        }

        # Check if we recently identified a bird in this area
        cx, cy = bbox_center
        for entry in self.recent_identifications.values():
            prev_cx, prev_cy = entry['location']
            distance = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5
            if distance < 100:  # Within 100 pixels
                return False

        return True

    def record_identification(self, track_id, species, bbox_center):
        """Record identification for cooldown tracking"""
        self.recent_identifications[track_id] = {
            'species': species,
            'time': time.time(),
            'location': bbox_center
        }


class BirdLogger:
    """Handles logging of bird identifications"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.log_file = self.config.LOG_FILE_PATH

        # Create log file with headers if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'track_id', 'species', 'confidence',
                    'detection_count', 'track_duration', 'photo_path',
                    'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'bbox_area'
                ])

    def log_identification(self, track_id, species, confidence, track_info, photo_path):
        """Log a bird identification"""
        timestamp = datetime.now().isoformat()

        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, track_id, species, confidence,
                track_info['hits'], track_info['duration'], photo_path,
                track_info['bbox'][0], track_info['bbox'][1],
                track_info['bbox'][2], track_info['bbox'][3],
                track_info['bbox_area']
            ])

        if self.config.DEBUG:
            print(f"DEBUG: Logged identification - Track {track_id}: {species} ({confidence:.2f})")

    def save_photo(self, track_id, image_crop):
        """Save bird photo"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bird_{track_id}_{timestamp}.jpg"
        photo_path = os.path.join(self.config.PHOTO_SAVE_PATH, filename)

        cv2.imwrite(photo_path, image_crop)
        return photo_path


# --- Core Classes ---
class BirdDetector:
    def __init__(self, config: AppConfig):
        self.config = config
        self.model, self.is_tensorrt = self._load_model()
        self.bird_class_id = self._get_bird_class_id()

        # Create CLAHE for contrast enhancement
        if self.config.ENHANCE_CONTRAST:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _load_model(self):
        try:
            model = YOLO(self.config.MODEL_PATH_TENSORRT)
            print("TensorRT engine loaded.")
            return model, True
        except Exception:
            print(f"TensorRT not found. Falling back to: {self.config.MODEL_PATH_PYTORCH}")
            model = YOLO(self.config.MODEL_PATH_PYTORCH)
            return model, False

    def _get_bird_class_id(self):
        for i, name in self.model.names.items():
            if name == 'bird':
                print(f"'bird' class ID: {i}")
                return i
        raise ValueError("'bird' class not found.")

    def _enhance_frame(self, frame):
        """Enhance frame to improve small bird detection"""
        if not self.config.ENHANCE_CONTRAST:
            return frame

        # Option 1: CLAHE + Sharpening (current method)
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        l = self.clahe.apply(l)

        # Merge channels and convert back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # Slight sharpening for small details
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel * 0.1)
        enhanced = cv2.addWeighted(frame, 0.7, enhanced, 0.3, 0)

        # Option 2: Alternative brightness/contrast boost (uncomment to try)
        # enhanced = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)  # Brightness boost

        # Option 3: Combine both methods (uncomment to try)
        # enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=5)  # Extra brightness

        return enhanced

    def _apply_filters(self, box_coords, confidence=None):
        x1, y1, x2, y2 = box_coords
        w, h = x2 - x1, y2 - y1
        area = w * h

        # Adaptive area filtering based on confidence
        min_area = self.config.MIN_BBOX_AREA
        if confidence and confidence < self.config.SMALL_BIRD_CONFIDENCE + 0.1:
            min_area = self.config.MIN_BBOX_AREA // 2  # Allow smaller birds if low confidence

        if not (min_area < area < self.config.MAX_BBOX_AREA):
            if self.config.DEBUG:
                print(f"DEBUG: Rejecting box due to area: {area} (min: {min_area})")
                return False
        if h == 0 or w == 0:
            return False
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio > self.config.MAX_ASPECT_RATIO:
            if self.config.DEBUG:
                print(f"DEBUG: Rejecting box due to aspect ratio: {aspect_ratio:.2f}")
                return False
        return True

    def _detect_single_scale(self, frame, inference_size, confidence_threshold):
        """Run detection at a single scale"""
        height, width, _ = frame.shape
        scale = inference_size / max(height, width)
        resized_frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

        use_half = not self.is_tensorrt
        results = self.model.predict(source=resized_frame, conf=confidence_threshold,
                                     iou=self.config.IOU_THRESHOLD, classes=[self.bird_class_id],
                                     half=use_half, verbose=False)

        detections = []
        for box in results[0].boxes:
            xyxy_resized = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = [int(coord / scale) for coord in xyxy_resized]
            confidence = box.conf[0].item()

            if self._apply_filters((x1, y1, x2, y2), confidence):
                detections.append({
                    "box": (x1, y1, x2, y2),
                    "confidence": confidence,
                    "scale": inference_size
                })

        return detections

    def _merge_multi_scale_detections(self, all_detections):
        """Merge detections from multiple scales using NMS"""
        if not all_detections:
            return []

        # Convert to format for NMS
        boxes = []
        scores = []

        for detection in all_detections:
            x1, y1, x2, y2 = detection["box"]
            boxes.append([x1, y1, x2, y2])
            scores.append(detection["confidence"])

        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes, scores,
                                   score_threshold=self.config.SMALL_BIRD_CONFIDENCE,
                                   nms_threshold=0.4)

        if len(indices) == 0:
            return []

        # Return merged detections
        merged_detections = []
        for i in indices.flatten():
            merged_detections.append(all_detections[i])

        return merged_detections

    def detect(self, frame):
        """Enhanced detection with multi-scale support"""
        # Enhance frame for better small bird detection
        enhanced_frame = self._enhance_frame(frame)

        if not self.config.USE_MULTI_SCALE:
            # Single scale detection (original method)
            return self._detect_single_scale(enhanced_frame,
                                             self.config.INFERENCE_SIZE,
                                             self.config.CONFIDENCE_THRESHOLD)

        # Multi-scale detection
        all_detections = []

        # Run detection at multiple scales
        for i, scale in enumerate(self.config.SCALES):
            # Use lower confidence for smaller scales (better for small birds)
            conf_threshold = (self.config.SMALL_BIRD_CONFIDENCE if scale > 640
                              else self.config.CONFIDENCE_THRESHOLD)

            if self.config.DEBUG and i == 0:
                print(f"DEBUG: Running multi-scale detection: scales={self.config.SCALES}")

            scale_detections = self._detect_single_scale(enhanced_frame, scale, conf_threshold)
            all_detections.extend(scale_detections)

        # Merge detections from all scales
        return self._merge_multi_scale_detections(all_detections)


class ImprovedTrack:
    """Improved tracker with bird identification capabilities"""

    def __init__(self, track_id, detection):
        self.track_id = track_id
        self.misses = 0
        self.hits = 1  # Track consecutive hits
        self.was_just_confirmed = True
        self.last_detection_time = time.time()
        self.creation_time = time.time()

        # NEW: Identification tracking
        self.identified = False
        self.identification_attempts = 0
        self.species = None
        self.species_confidence = 0.0
        self.best_frames = []  # Store best quality frames for ID
        self.stable_start_time = None  # When bird became stable

        # Enhanced Kalman filter for erratic bird movement
        # State: [x, y, width, height, vx, vy, vw, vh]
        self.kf = cv2.KalmanFilter(8, 4)

        # Measurement matrix (we observe x, y, w, h)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], np.float32)

        # Transition matrix (constant velocity model with damping)
        dt = 1.0  # time step
        damping = 0.7  # velocity damping factor for erratic bird movement
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, dt, 0, 0, 0],
            [0, 1, 0, 0, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, 0, dt, 0],
            [0, 0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 0, damping, 0, 0, 0],
            [0, 0, 0, 0, 0, damping, 0, 0],
            [0, 0, 0, 0, 0, 0, damping, 0],
            [0, 0, 0, 0, 0, 0, 0, damping]
        ], np.float32)

        # Process noise covariance (higher for bird movement uncertainty)
        process_noise = np.eye(8, dtype=np.float32)
        process_noise[0:2, 0:2] *= 5.0  # position uncertainty
        process_noise[2:4, 2:4] *= 2.0  # size uncertainty
        process_noise[4:6, 4:6] *= 10.0  # velocity uncertainty (birds change direction quickly)
        process_noise[6:8, 6:8] *= 5.0  # size velocity uncertainty
        self.kf.processNoiseCov = process_noise

        # Measurement noise covariance (how much we trust detections)
        measurement_noise = np.eye(4, dtype=np.float32)
        measurement_noise[0:2, 0:2] *= 8.0  # position measurement noise
        measurement_noise[2:4, 2:4] *= 4.0  # size measurement noise
        self.kf.measurementNoiseCov = measurement_noise

        # Error covariance
        cv2.setIdentity(self.kf.errorCovPost, 1.0)

        # Initialize state
        x1, y1, x2, y2 = detection["box"]
        w, h = x2 - x1, y2 - y1
        center_x, center_y = x1 + w / 2, y1 + h / 2
        self.kf.statePost = np.array([center_x, center_y, w, h, 0, 0, 0, 0], np.float32)

    def predict(self):
        """Predict next state"""
        prediction = self.kf.predict()
        self.was_just_confirmed = False
        self.misses += 1

        # Adaptive process noise based on time since last detection
        time_since_detection = time.time() - self.last_detection_time
        if time_since_detection > 1.0:  # If no detection for 1 second, increase uncertainty
            self.kf.processNoiseCov *= 1.2

        cx, cy, w, h = prediction[0], prediction[1], prediction[2], prediction[3]

        # Ensure reasonable bounding box
        w = max(10, abs(w))
        h = max(10, abs(h))

        return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))

    def update(self, detection):
        """Update with new detection"""
        self.misses = 0
        self.hits += 1
        self.was_just_confirmed = True
        self.last_detection_time = time.time()

        # Store confidence for display
        self.last_confidence = detection.get("confidence", 0.5)

        x1, y1, x2, y2 = detection["box"]
        w, h = x2 - x1, y2 - y1
        center_x, center_y = x1 + w / 2, y1 + h / 2
        measurement = np.array([center_x, center_y, w, h], np.float32)

        # Adaptive measurement noise based on confidence
        confidence = detection.get("confidence", 0.5)
        noise_factor = 1.5 - confidence  # Lower confidence = higher noise
        temp_measurement_noise = self.kf.measurementNoiseCov.copy()
        temp_measurement_noise *= noise_factor
        self.kf.measurementNoiseCov = temp_measurement_noise

        self.kf.correct(measurement)

        # Reset process noise if we got a good detection
        if confidence > 0.7:
            process_noise = np.eye(8, dtype=np.float32)
            process_noise[0:2, 0:2] *= 5.0
            process_noise[2:4, 2:4] *= 2.0
            process_noise[4:6, 4:6] *= 10.0
            process_noise[6:8, 6:8] *= 5.0
            self.kf.processNoiseCov = process_noise

    def store_frame_for_id(self, frame, bbox, config):
        """Store high-quality frames for later identification"""
        if self.identified or len(self.best_frames) >= config.ID_MAX_ATTEMPTS + 2:
            return  # Already identified or have enough frames

        x1, y1, x2, y2 = bbox

        # Extract bird crop with some padding
        padding = 20
        h, w = frame.shape[:2]
        x1_pad = max(0, x1 - padding)
        y1_pad = max(0, y1 - padding)
        x2_pad = min(w, x2 + padding)
        y2_pad = min(h, y2 + padding)

        crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
        if crop.size == 0:
            return

        # Calculate quality score
        bbox_area = (x2 - x1) * (y2 - y1)
        quality = calculate_frame_quality(crop, bbox_area)
        confidence = getattr(self, 'last_confidence', 0.5)

        frame_info = {
            'crop': crop.copy(),
            'quality': quality,
            'confidence': confidence,
            'bbox': (x1, y1, x2, y2),
            'bbox_area': bbox_area,
            'timestamp': time.time()
        }

        # Keep only the best frames (sorted by quality)
        self.best_frames.append(frame_info)
        self.best_frames.sort(key=lambda x: x['quality'] * x['confidence'], reverse=True)

        # Keep only top N frames
        max_frames = config.ID_MAX_ATTEMPTS + 2
        self.best_frames = self.best_frames[:max_frames]

    def should_attempt_identification(self, config):
        """Check if we should attempt bird identification"""
        if self.identified or self.identification_attempts >= config.ID_MAX_ATTEMPTS:
            return False

        # Check hit count
        if self.hits < config.ID_TRIGGER_HITS:
            return False

        # Check if we have high confidence detection
        if hasattr(self, 'last_confidence') and self.last_confidence >= config.ID_HIGH_CONFIDENCE:
            return True

        # Check if bird has been stable for required time
        current_time = time.time()
        if self.is_stable():
            if self.stable_start_time is None:
                self.stable_start_time = current_time
            elif (current_time - self.stable_start_time) >= config.ID_STABLE_TIME:
                return True
        else:
            self.stable_start_time = None

        return False

    def is_stable(self):
        """Check if bird movement is stable (low velocity)"""
        state = self.kf.statePost
        vx, vy = state[4], state[5]
        velocity = (vx ** 2 + vy ** 2) ** 0.5
        return velocity < 5.0  # Pixels per frame

    def get_best_frame_for_id(self):
        """Get the best quality frame for identification"""
        if not self.best_frames:
            return None
        return self.best_frames[0]['crop']

    def get_track_info(self):
        """Get track information for logging"""
        duration = time.time() - self.creation_time
        bbox = self.bbox
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        return {
            'hits': self.hits,
            'duration': duration,
            'bbox': bbox,
            'bbox_area': bbox_area
        }

    @property
    def bbox(self):
        """Get current bounding box"""
        state = self.kf.statePost
        cx, cy, w, h = state[0], state[1], state[2], state[3]

        # Ensure reasonable bounding box
        w = max(10, abs(w))
        h = max(10, abs(h))

        return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))

    @property
    def bbox_center(self):
        """Get bounding box center"""
        bbox = self.bbox
        return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

    @property
    def is_confirmed(self):
        """Track is confirmed if it has enough hits"""
        return self.hits >= 3


class FinalObjectTracker:
    def __init__(self, config: AppConfig):
        self.config = config
        self.tracks = {}
        self.next_track_id = 0
        self.max_misses = 20  # Adjusted for 3 FPS processing

    def update(self, detections):
        # Predict all tracks
        predicted_boxes = {}
        for track_id, track in self.tracks.items():
            predicted_boxes[track_id] = track.predict()

        if detections:
            track_indices = list(self.tracks.keys())
            detection_indices = list(range(len(detections)))
            matched_track_ids = set()
            matched_detection_indices = set()

            if track_indices and detection_indices:
                # Compute IOU matrix
                iou_matrix = np.zeros((len(track_indices), len(detection_indices)))
                for i, track_id in enumerate(track_indices):
                    for j, det_idx in enumerate(detection_indices):
                        iou_matrix[i, j] = calculate_iou(predicted_boxes[track_id], detections[det_idx]["box"])

                # Hungarian-like assignment (greedy)
                for _ in range(min(len(track_indices), len(detection_indices))):
                    if np.all(iou_matrix < self.config.IOU_MATCHING_THRESHOLD):
                        break

                    track_idx, det_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                    if iou_matrix[track_idx, det_idx] < self.config.IOU_MATCHING_THRESHOLD:
                        break

                    track_id = track_indices[track_idx]
                    detection_index = detection_indices[det_idx]
                    self.tracks[track_id].update(detections[detection_index])
                    matched_track_ids.add(track_id)
                    matched_detection_indices.add(detection_index)

                    # Mark row and column as used
                    iou_matrix[track_idx, :] = -1
                    iou_matrix[:, det_idx] = -1

            # Create new tracks for unmatched detections
            for det_idx in set(range(len(detections))) - matched_detection_indices:
                new_track = ImprovedTrack(self.next_track_id, detections[det_idx])
                self.tracks[self.next_track_id] = new_track
                self.next_track_id += 1
                if self.config.DEBUG:
                    print(f"DEBUG: Created new track {self.next_track_id - 1}")

        # Remove tracks with too many misses
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track.misses > self.max_misses:
                tracks_to_remove.append(track_id)
                if self.config.DEBUG:
                    print(f"DEBUG: Removing track {track_id} after {track.misses} misses")

        for track_id in tracks_to_remove:
            del self.tracks[track_id]

        # Return only confirmed tracks for display
        return [track for track in self.tracks.values() if track.is_confirmed]


# --- IMPROVED Single Thread Approach with Bird ID ---
class StreamProcessor:
    """Improved single-threaded processor with bird identification"""

    def __init__(self, config: AppConfig, stream_url: str):
        self.config = config
        self.stream_url = stream_url
        self.detector = BirdDetector(config)
        self.tracker = FinalObjectTracker(config)

        # NEW: Bird identification components
        if self.config.ENABLE_BIRD_ID:
            self.bird_identifier = BirdIdentifier(config)
            self.bird_logger = BirdLogger(config)
            print("Bird identification system initialized")
        else:
            self.bird_identifier = None
            self.bird_logger = None

        # Timing control
        self.last_detection_time = 0
        self.detection_interval = 1.0 / self.config.PROCESS_TARGET_FPS

        # Statistics
        self.frames_read = 0
        self.frames_processed = 0
        self.frames_skipped = 0
        self.birds_identified = 0

    def process_bird_identification(self, frame, visible_tracks):
        """Process bird identification for eligible tracks"""
        if not self.config.ENABLE_BIRD_ID or not self.bird_identifier:
            return

        for track in visible_tracks:
            # Store frame for potential identification
            track.store_frame_for_id(frame, track.bbox, self.config)

            # Check if we should attempt identification
            if track.should_attempt_identification(self.config):
                self.attempt_identification(track)

    def attempt_identification(self, track):
        """Attempt to identify a bird species"""
        if not self.bird_identifier.should_identify(track.track_id, track.bbox_center):
            if self.config.DEBUG:
                print(f"DEBUG: Skipping identification for Track {track.track_id} (cooldown)")
            return

        # Get best frame for identification
        best_crop = track.get_best_frame_for_id()
        if best_crop is None:
            if self.config.DEBUG:
                print(f"DEBUG: No suitable frame for Track {track.track_id} identification")
            return

        track.identification_attempts += 1

        # Attempt identification
        species, confidence = self.bird_identifier.identify_species(best_crop, track.track_id)

        if species and confidence >= self.config.ID_MIN_SPECIES_CONFIDENCE:
            # Successful identification
            track.identified = True
            track.species = species
            track.species_confidence = confidence

            # Save photo and log identification
            photo_path = self.bird_logger.save_photo(track.track_id, best_crop)
            track_info = track.get_track_info()
            self.bird_logger.log_identification(
                track.track_id, species, confidence, track_info, photo_path
            )

            # Record for cooldown
            self.bird_identifier.record_identification(
                track.track_id, species, track.bbox_center
            )

            self.birds_identified += 1
            print(f"🐦 IDENTIFIED: Track {track.track_id} -> {species} ({confidence:.2f})")

        elif self.config.DEBUG:
            print(f"DEBUG: Low confidence identification for Track {track.track_id}: {species} ({confidence:.2f})")

    def run(self):
        print("Opening stream with bird identification support...")
        cap = cv2.VideoCapture(self.stream_url)

        # Optimize capture settings
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Small buffer for smoother playback
        cap.set(cv2.CAP_PROP_FPS, self.config.CAPTURE_FPS_LIMIT)

        if not cap.isOpened():
            print("Could not open stream.")
            return False

        print("Stream opened successfully! Press 'q' to quit.")
        cv2.namedWindow(self.config.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.config.WINDOW_NAME, 1280, 720)

        # For display timing
        last_display_time = time.time()
        frame_count = 0

        while True:
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret:
                print("End of stream.")
                break

            self.frames_read += 1
            frame_count += 1

            # Clear buffer if it's getting full (prevents lag)
            buffer_size = cap.get(cv2.CAP_PROP_BUFFERSIZE)
            if buffer_size > 1:
                # Read and discard extra frames
                for _ in range(int(buffer_size) - 1):
                    ret_skip, _ = cap.read()
                    if ret_skip:
                        self.frames_skipped += 1

            detections = []
            current_time = time.time()

            # Only run detection at target FPS
            if (current_time - self.last_detection_time) >= self.detection_interval:
                self.last_detection_time = current_time
                if self.config.DEBUG and self.frames_processed % 10 == 0:
                    print(f"DEBUG: Running detection on frame {self.frames_read}")
                detections = self.detector.detect(frame)
                self.frames_processed += 1

            # Always update tracker (handles missing detections gracefully)
            visible_tracks = self.tracker.update(detections)

            # NEW: Process bird identification
            self.process_bird_identification(frame, visible_tracks)

            # Draw results with species information
            annotated_frame = frame.copy()
            for track in visible_tracks:
                x1, y1, x2, y2 = track.bbox

                # Color coding based on identification status
                if track.identified:
                    color = (0, 255, 0)  # Green for identified birds
                    thickness = 3
                elif track.hits >= 5:
                    color = (0, 255, 255)  # Yellow for stable tracks
                    thickness = 3
                elif track.was_just_confirmed:
                    color = (255, 255, 0)  # Cyan for new detections
                    thickness = 2
                else:
                    color = self.config.TRACKER_COLOR  # Orange for tracking
                    thickness = 2

                # Prepare label with species info
                if track.identified and track.species:
                    label = f"{track.species} ({track.species_confidence:.2f})"
                else:
                    confidence_info = ""
                    if hasattr(track, 'last_confidence'):
                        confidence_info = f" ({track.last_confidence:.2f})"
                    label = f"Bird {track.track_id} ({track.hits}){confidence_info}"

                # Draw bounding box and label
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Add ID status indicator
                if track.identified:
                    cv2.putText(annotated_frame, "✓ ID", (x2 - 30, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                elif track.should_attempt_identification(self.config):
                    cv2.putText(annotated_frame, "?", (x2 - 15, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            # Add identification statistics to display
            if self.config.ENABLE_BIRD_ID:
                stats_text = f"Birds Identified: {self.birds_identified}"
                cv2.putText(annotated_frame, stats_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show frame
            cv2.imshow(self.config.WINDOW_NAME, annotated_frame)

            # Debug output every 10 seconds
            if self.config.DEBUG and (current_time - last_display_time) > 10:
                last_display_time = current_time
                fps = frame_count / 10.0
                frame_count = 0
                print(f"DEBUG: Display FPS: {fps:.1f}, Frames read: {self.frames_read}, "
                      f"processed: {self.frames_processed}, skipped: {self.frames_skipped}, "
                      f"birds identified: {self.birds_identified}")

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Control frame rate for smooth playback
            loop_time = time.time() - loop_start
            target_loop_time = self.config.DISPLAY_FRAME_TIME
            if loop_time < target_loop_time:
                time.sleep(target_loop_time - loop_time)

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print(f"Stopped. Total frames read: {self.frames_read}, processed: {self.frames_processed}, "
              f"skipped: {self.frames_skipped}, birds identified: {self.birds_identified}")

        if self.config.ENABLE_BIRD_ID:
            print(f"Bird photos saved to: {self.config.PHOTO_SAVE_PATH}")
            print(f"Identification log saved to: {self.config.LOG_FILE_PATH}")

        return True


# --- FIXED Simple Threading Approach ---
class SimpleFrameProcessor:
    """Fixed simpler two-thread approach"""

    def __init__(self, config: AppConfig, stream_url: str):
        self.config = config
        self.stream_url = stream_url
        self.detector = BirdDetector(config)
        self.tracker = FinalObjectTracker(config)

        # Shared state
        self.current_frame = None
        self.annotated_frame = None
        self.frame_lock = threading.Lock()
        self.stop_event = threading.Event()

        # Statistics
        self.frames_captured = 0
        self.frames_processed = 0

    def capture_thread(self):
        """Lightweight capture thread"""
        print("Capture thread started.")
        cap = cv2.VideoCapture(self.stream_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print("Capture thread: Could not open stream.")
            self.stop_event.set()
            return

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Capture thread: End of stream.")
                self.stop_event.set()
                break

            self.frames_captured += 1

            # Update shared frame (always try to update)
            with self.frame_lock:
                self.current_frame = frame.copy()

        cap.release()
        print(f"Capture thread stopped. Captured {self.frames_captured} frames.")

    def processing_thread(self):
        """Processing thread"""
        print("Processing thread started.")
        last_process_time = 0
        process_interval = 1.0 / self.config.PROCESS_TARGET_FPS

        while not self.stop_event.is_set():
            current_time = time.time()

            # Get latest frame
            with self.frame_lock:
                if self.current_frame is None:
                    time.sleep(0.01)
                    continue
                frame = self.current_frame.copy()

            detections = []

            # Only run detection at target FPS
            if (current_time - last_process_time) >= process_interval:
                last_process_time = current_time
                detections = self.detector.detect(frame)
                self.frames_processed += 1
                if self.config.DEBUG and self.frames_processed % 5 == 0:
                    print(f"DEBUG: Processed frame {self.frames_processed}")

            # Update tracker
            visible_tracks = self.tracker.update(detections)

            # Draw results
            annotated_frame = frame.copy()
            for track in visible_tracks:
                x1, y1, x2, y2 = track.bbox

                if track.hits >= 5:
                    color = self.config.CONFIRM_COLOR
                    thickness = 3
                elif track.was_just_confirmed:
                    color = (0, 255, 255)  # Yellow for new detections
                    thickness = 2
                else:
                    color = self.config.TRACKER_COLOR
                    thickness = 2

                label = f"Bird {track.track_id} ({track.hits})"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Update shared annotated frame
            with self.frame_lock:
                self.annotated_frame = annotated_frame.copy()

            time.sleep(0.01)  # Small sleep to prevent 100% CPU

        print(f"Processing thread stopped. Processed {self.frames_processed} frames.")

    def run(self):
        """Main run method"""
        # Start threads
        capture_thread = threading.Thread(target=self.capture_thread, daemon=True)
        processing_thread = threading.Thread(target=self.processing_thread, daemon=True)

        capture_thread.start()
        time.sleep(0.5)  # Give capture thread time to start
        processing_thread.start()

        # Wait for first frame with timeout
        print("Waiting for first frame...")
        timeout = time.time() + 10  # 10 second timeout
        while self.annotated_frame is None and not self.stop_event.is_set() and time.time() < timeout:
            time.sleep(0.1)

        if self.stop_event.is_set():
            print("Failed to start - capture thread error.")
            return False

        if self.annotated_frame is None:
            print("Timeout waiting for first frame.")
            self.stop_event.set()
            return False

        print("Stream opened successfully! Press 'q' to quit.")
        cv2.namedWindow(self.config.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.config.WINDOW_NAME, 1280, 720)

        # Display loop
        last_display_time = time.time()
        frame_count = 0

        while not self.stop_event.is_set():
            loop_start = time.time()

            with self.frame_lock:
                if self.annotated_frame is not None:
                    display_frame = self.annotated_frame.copy()
                else:
                    time.sleep(0.01)
                    continue

            cv2.imshow(self.config.WINDOW_NAME, display_frame)
            frame_count += 1

            # Debug output
            current_time = time.time()
            if self.config.DEBUG and (current_time - last_display_time) > 10:
                fps = frame_count / 10.0
                frame_count = 0
                last_display_time = current_time
                print(
                    f"DEBUG: Display FPS: {fps:.1f}, Captured: {self.frames_captured}, Processed: {self.frames_processed}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_event.set()

            # Control display timing
            loop_time = time.time() - loop_start
            if loop_time < self.config.DISPLAY_FRAME_TIME:
                time.sleep(self.config.DISPLAY_FRAME_TIME - loop_time)

        # Cleanup
        self.stop_event.set()
        capture_thread.join(timeout=2)
        processing_thread.join(timeout=2)
        cv2.destroyAllWindows()
        return True


# --- Main App ---
class BirdWatcherApp:
    def __init__(self, config: AppConfig):
        self.config = config

    def run(self):
        stream_url = input("Paste the direct M3U8 stream URL obtained from yt-dlp here: ")
        if not stream_url.strip():
            return

        print("\nChoose processing method:")
        print("1. Single-threaded (recommended for smooth tracking)")
        print("2. Simple two-threaded (fixed version)")
        choice = input("Enter choice (1 or 2) [1]: ").strip() or "1"

        if choice == "1":
            processor = StreamProcessor(self.config, stream_url)
            processor.run()
        else:
            processor = SimpleFrameProcessor(self.config, stream_url)
            processor.run()


if __name__ == "__main__":
    print("🐦 Bird Feeder Cam - Detection & Identification System")
    print("=" * 55)

    # Check dependencies
    if not TRANSFORMERS_AVAILABLE:
        print("\n⚠️  OPTIONAL DEPENDENCIES MISSING:")
        print("For real bird identification, install:")
        print("  pip install transformers torch pillow")
        print("  (System will work with placeholder identification)")
        print()

    app_config = AppConfig()
    app_config.DEBUG = True  # Enable debug by default

    # Ask about bird identification
    print("1. Bird Identification:")
    if TRANSFORMERS_AVAILABLE:
        print("   ✓ EfficientNet bird model available (525 species, 99.1% accuracy)")
    else:
        print("   ⚠️  Placeholder model (install transformers for real identification)")

    enable_id = input("Enable bird identification? (y/n) [y]: ").strip().lower()
    if enable_id in ['n', 'no']:
        app_config.ENABLE_BIRD_ID = False
        print("✓ Bird identification disabled - detection only mode")
    else:
        app_config.ENABLE_BIRD_ID = True
        if TRANSFORMERS_AVAILABLE:
            print("✓ EfficientNet bird identification enabled (525 global species)")
        else:
            print("✓ Placeholder bird identification enabled")

        # ID configuration options
        print("\n   Identification Trigger Settings:")
        trigger_mode = input("   Use default settings? (y/n) [y]: ").strip().lower()
        if trigger_mode in ['n', 'no']:
            try:
                hits = int(input(
                    f"   Minimum detections before ID attempt [{app_config.ID_TRIGGER_HITS}]: ") or app_config.ID_TRIGGER_HITS)
                confidence = float(input(
                    f"   High confidence threshold [{app_config.ID_HIGH_CONFIDENCE}]: ") or app_config.ID_HIGH_CONFIDENCE)
                stable_time = float(input(
                    f"   Stable time requirement (seconds) [{app_config.ID_STABLE_TIME}]: ") or app_config.ID_STABLE_TIME)

                app_config.ID_TRIGGER_HITS = hits
                app_config.ID_HIGH_CONFIDENCE = confidence
                app_config.ID_STABLE_TIME = stable_time
                print("✓ Custom identification settings applied")
            except ValueError:
                print("✓ Using default identification settings")

    # Ask about small bird detection mode
    print("\n2. Small Bird Detection:")
    print("   Standard: Faster processing, good for larger birds")
    print("   Enhanced: Slower but better for small/distant birds")

    detection_mode = input("Choose detection mode (standard/enhanced) [standard]: ").strip().lower()

    if detection_mode in ['enhanced', 'e']:
        print("✓ Enhanced small bird detection enabled")
        app_config.USE_MULTI_SCALE = True
        app_config.ENHANCE_CONTRAST = True
        app_config.CONFIDENCE_THRESHOLD = 0.20
        app_config.SMALL_BIRD_CONFIDENCE = 0.15
        app_config.MIN_BBOX_AREA = 400
        app_config.PROCESS_TARGET_FPS = 2 if not app_config.ENABLE_BIRD_ID else 1
    else:
        print("✓ Standard detection mode enabled")
        app_config.USE_MULTI_SCALE = False
        app_config.ENHANCE_CONTRAST = False
        app_config.PROCESS_TARGET_FPS = 3 if not app_config.ENABLE_BIRD_ID else 2

    # Show final configuration
    print("\n" + "=" * 55)
    print("Final Configuration:")
    print(f"• Bird Identification: {'Enabled' if app_config.ENABLE_BIRD_ID else 'Disabled'}")
    if app_config.ENABLE_BIRD_ID:
        if TRANSFORMERS_AVAILABLE:
            print("• ID Model: EfficientNet-B2 (525 species, 99.1% accuracy)")
        else:
            print("• ID Model: Placeholder (demo mode)")
    print(f"• Detection Mode: {'Enhanced' if app_config.USE_MULTI_SCALE else 'Standard'}")
    print(f"• Processing Rate: {app_config.PROCESS_TARGET_FPS} FPS")
    if app_config.ENABLE_BIRD_ID:
        print(
            f"• ID Trigger: {app_config.ID_TRIGGER_HITS} hits OR {app_config.ID_HIGH_CONFIDENCE} confidence OR {app_config.ID_STABLE_TIME}s stable")
        print(f"• Photos will be saved to: {app_config.PHOTO_SAVE_PATH}/")
        print(f"• Log will be saved to: {app_config.LOG_FILE_PATH}")
    print("=" * 55)

    if app_config.ENABLE_BIRD_ID and not TRANSFORMERS_AVAILABLE:
        print("\n💡 TIP: For better bird identification accuracy, install:")
        print("   pip install transformers torch pillow")
        print("   Then restart the application.")

    # Create necessary directories
    if app_config.ENABLE_BIRD_ID:
        Path(app_config.PHOTO_SAVE_PATH).mkdir(exist_ok=True)

    app = BirdWatcherApp(app_config)
    app.run()