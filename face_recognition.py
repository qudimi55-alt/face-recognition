import cv2
import time
# مكتبة YOLO
from ultralytics import YOLO

import winsound


def play_system_beep(duration_seconds=1):
    """
    تنشيط صفارة نظام وتردد عالي لنظام ويندوز
    """
    frequency = 2500  # التردد بالـ Hz (صوت حاد للتنبيه)
    duration_ms = int(duration_seconds * 1000)  # تحويل إلى ميلي ثانية
    
    winsound.Beep(frequency, duration_ms)
# تحميل نموذج YOLOv8 المدرب مسبقاً على كشف الكائنات (بما فيها الأشخاص والهواتف)
model = YOLO('yolov8n.pt') 

cap = cv2.VideoCapture(0)

# لتخزين توقيت بداية حمل الهاتف لكل شخص
phone_usage_start_times = {} 

# الفئات في نموذج YOLO: 0 للشخص، 67 للهاتف
PERSON_CLASS_ID = 0
PHONE_CLASS_ID = 67

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # إجراء التعرف على الكائنات في الإطار الحالي
    results = model(frame, stream=True)

    detected_phones = []
    detected_persons = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # إذا تم اكتشاف شخص
            if cls == PERSON_CLASS_ID and conf > 0.5:
                detected_persons.append(box)
                
            # إذا تم اكتشاف هاتف
            elif cls == PHONE_CLASS_ID and conf > 0.5:
                detected_phones.append(box)
                
    # المنطق الأساسي: هل يوجد هاتف ضمن منطقة شخص؟
    # هذا جزء مبسط جداً، في الواقع نحتاج تتبع كل شخص
    for person_box in detected_persons:
        px1, py1, px2, py2 = map(int, person_box.xyxy[0])
        
        person_has_phone = False
        for phone_box in detected_phones:
            phx1, phy1, phx2, phy2 = map(int, phone_box.xyxy[0])
            
            # التحقق مما إذا كان مربع الهاتف يتقاطع مع مربع الشخص
            if phx1 > px1 and phx2 < px2 and phy1 > py1 and phy2 < py2:
                person_has_phone = True
                break
                
        # إذا تم اكتشاف استخدام الهاتف لدى هذا الشخص
        if person_has_phone:
            person_id = "Employee_0" # في المرحلة التالية سنستبدل هذا بالاسم الحقيقي
            
            if person_id not in phone_usage_start_times:
                # تسجيل وقت بداية حمل الهاتف
                phone_usage_start_times[person_id] = time.time()
                print(f"[{time.strftime('%H:%M:%S')}] إنذار: تم رصد موظف يحمل هاتفاً!")
            else:
                # حساب مدة حمل الهاتف
                duration = time.time() - phone_usage_start_times[person_id]
                cv2.putText(frame, f"في الهاتف: {duration:.1f} ثانية", (px1, py1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
            # رسم مربع أحمر حول الشخص المخالف
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
            play_system_beep(1.5)
        else:
            # إذا لم يكن يحمل هاتفاً، احذف توقيت البداية إذا وجد
            if "Employee_0" in phone_usage_start_times:
                del phone_usage_start_times["Employee_0"]
            # رسم مربع أخضر حول الشخص الطبيعي
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
            print("تعديل من main")            
    cv2.imshow('نظام مراقبة الموظفين', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()