# detect_jenga.py
import cv2
from ultralytics import YOLO

def main():
    # 1) YOLOv8n 모델 로드 (처음 실행 시 자동 다운로드)
    model = YOLO("yolov8n.pt")

    # 2) 테스트용 이미지 경로 (여러 장이면 반복 처리하도록 확장 가능)
    image_path = "jenga_test.jpg"  # 여기에 본인이 찍은 이미지 경로 입력

    # 3) 이미지 읽기
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

    # 4) 모델 추론
    #    conf=0.5: 50% 이상 신뢰도만 사용 (상황 보면서 조정 가능)
    results = model(img, conf=0.5)[0]

    # 5) 결과에서 bbox(x1, y1, x2, y2) 가져오기
    boxes = results.boxes.xyxy  # (N, 4) tensorr
    scores = results.boxes.conf # (N,)
    clses  = results.boxes.cls  # (N,)

    centers = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.tolist()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        score = float(scores[i])
        cls_id = int(clses[i])

        centers.append((cx, cy))

        # 6) 시각화를 위해 bbox와 중심점 그리기
        #    (지금은 클래스 구분 없이 모든 박스를 "Jenga"라고 가정)
        cv2.rectangle(
            img,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0), 2
        )
        cv2.circle(img, (int(cx), int(cy)), 4, (0, 0, 255), -1)
        cv2.putText(
            img,
            f"id:{i} ({int(cx)},{int(cy)})",
            (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    print("검출된 객체 개수:", len(centers))
    for i, (cx, cy) in enumerate(centers):
        print(f"[{i}] center = ({cx:.1f}, {cy:.1f})")

    # 7) 결과 이미지 출력
    cv2.imshow("Jenga detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
