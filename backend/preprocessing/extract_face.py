import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import math

from muton.config import env_path

# =========================
# 설정 (경로 확인!)
# =========================
VIDEO_ROOT = env_path("MUTON_VIDEO_ROOT", "data/video")
OUTPUT_ROOT = env_path("MUTON_FRAME_ROOT", "data/frames")

TARGET_FPS = 5  # 초당 5프레임 검사
PADDING_RATIO = 0.6
ROTATE_CLOCKWISE = True  # 영상이 누워있으면 True로 설정

# =========================
# MediaPipe FaceMesh (민감도 극대화!)
# =========================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.1  # ★ 0.5에서 0.1로 낮춰서 아주 작은 단서도 잡도록!
)

# 이미지를 회전시키는 함수
def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

# 영상이 누워있을 때 세우는 함수
def rotate_if_needed(frame):
    if ROTATE_CLOCKWISE:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame

# 종합 감정 점수 계산 (입 벌어짐 + 미간 찌푸림 + 눈 크기)
def calculate_emotion_score(landmarks):
    # 1. 입 벌어짐
    lip_top = landmarks[13]
    lip_bottom = landmarks[14]
    mouth_h = np.linalg.norm(np.array([lip_top.x, lip_top.y]) - np.array([lip_bottom.x, lip_bottom.y]))
    
    # 2. 미간 (좁을수록 점수 높음)
    brow_left = landmarks[107]
    brow_right = landmarks[336]
    brow_dist = np.linalg.norm(np.array([brow_left.x, brow_left.y]) - np.array([brow_right.x, brow_right.y]))
    brow_score = 1.0 / (brow_dist + 0.01)
    
    # 3. 눈 벌어짐
    l_eye_top = landmarks[159]
    l_eye_btm = landmarks[145]
    eye_h = np.linalg.norm(np.array([l_eye_top.x, l_eye_top.y]) - np.array([l_eye_btm.x, l_eye_btm.y]))
    
    total_score = (mouth_h * 10.0) + (brow_score * 0.5) + (eye_h * 5.0)
    return total_score

# 얼굴 수평 맞추기 (정렬)
def align_face(frame, landmarks):
    h, w = frame.shape[:2]
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    lx, ly = int(left_eye.x * w), int(left_eye.y * h)
    rx, ry = int(right_eye.x * w), int(right_eye.y * h)
    dy = ry - ly
    dx = rx - lx
    angle = math.degrees(math.atan2(dy, dx))
    center = ((lx + rx) // 2, (ly + ry) // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_CUBIC)

# 얼굴만 잘라내기 (크롭)
def crop_face(frame, landmarks):
    h, w = frame.shape[:2]
    xs = [l.x for l in landmarks]
    ys = [l.y for l in landmarks]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = int((x_min + x_max) / 2 * w)
    cy = int((y_min + y_max) / 2 * h)
    bw = int((x_max - x_min) * w)
    bh = int((y_max - y_min) * h)
    pad = int(max(bw, bh) * PADDING_RATIO)
    x1 = max(0, cx - bw // 2 - pad)
    y1 = max(0, cy - bh // 2 - pad)
    x2 = min(w, cx + bw // 2 + pad)
    y2 = min(h, cy + bh // 2 + pad)
    face = frame[y1:y2, x1:x2]
    return face

# ★★★ [핵심 기능] 회전하며 끈질기게 얼굴 찾기 ★★★
def detect_face_robust(frame):
    # 1. 원본 이미지에서 시도
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    if result.multi_face_landmarks:
        return frame, result.multi_face_landmarks[0].landmark
    
    # 2. 실패하면? 돌려보자 (-30도, +30도, -45도, +45도)
    angles = [-30, 30, -45, 45]
    for angle in angles:
        rotated_frame = rotate_image(frame, angle)
        rgb_rot = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb_rot)
        
        if result.multi_face_landmarks:
            # 찾았다! (회전된 프레임과 랜드마크 반환)
            return rotated_frame, result.multi_face_landmarks[0].landmark

    return None, None

# 비디오 처리 함수
def process_video(video_path: Path, out_dir: Path, video_stem: str):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    interval = max(1, int(round(fps / TARGET_FPS)))
    
    out_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    best_score = -1.0
    best_face_img = None
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        if idx % interval == 0:
            # 1. 기본 회전 (세로 모드 보정)
            frame = rotate_if_needed(frame)

            # 2. ★ 강제 탐색 (기울어진 얼굴도 놓치지 않음!)
            found_frame, landmarks = detect_face_robust(frame)
            
            if found_frame is not None and landmarks is not None:
                # 3. 점수 계산
                score = calculate_emotion_score(landmarks)
                
                # 4. 최고의 한 장 뽑기
                if score > best_score:
                    # 회전된 상태(found_frame)를 기준으로 정렬 및 크롭
                    aligned = align_face(found_frame, landmarks)
                    face = crop_face(aligned, landmarks)
                    
                    if face is not None and face.shape[0] > 20:
                        best_score = score
                        best_face_img = face 

        idx += 1

    cap.release()

    # 5. 결과 저장
    if best_face_img is not None:
        out_filename = f"{video_stem}.jpeg"
        out_path = out_dir / out_filename
        cv2.imwrite(str(out_path), best_face_img)
        print(f"[SAVE] {out_filename} (Score: {best_score:.2f})")
    else:
        print(f"[FAIL] {video_path.name} - 얼굴 아예 못 찾음")

# 메인 함수
def main():
    for folder_path in VIDEO_ROOT.iterdir():
        if not folder_path.is_dir(): continue
        folder_name = folder_path.name
        final_out_dir = OUTPUT_ROOT / folder_name
        
        print(f"\n=== Processing {folder_name} ===")
        for mp4 in sorted(folder_path.glob("*.mp4")):
            process_video(mp4, final_out_dir, mp4.stem)

if __name__ == "__main__":
    main()
