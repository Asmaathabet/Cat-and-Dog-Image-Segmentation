from ultralytics import YOLO

model = YOLO('yolo11n-seg-custom.pt')

# model.predict(source='D:/2026/Developing/ml_projects/instance-segmentation-yollo11/2.jpg',
#                show=False,
#                save=True, 
#                conf =0.5, 
#                line_width=2, 
#                show_labels=True,
#                show_conf=True, 
#                save_txt=False,
#                device='cpu',
#                classes=[0, 1])


model.predict(source='D:/2026/Developing/ml_projects/instance-segmentation-yollo11/cats-and-dogs.mp4',
               show=False,
               save=True, 
               conf =0.5, 
               line_width=2, 
               show_labels=True,
               show_conf=True, 
               save_txt=False,
               device='cpu',
               classes=[0, 1])

