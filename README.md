road_object_classification/
├── dataset/
│   ├── daytime/
│   │   ├── car/
│   │   ├── bus/
│   │   ├── truck/
│   │   ├── motorcycle/
│   │   └── human/
│   ├── nighttime/
│   │   ├── car/
│   │   ├── bus/
│   │   ├── truck/
│   │   ├── motorcycle/
│   │   └── human/
│   └── combined/              ← (Optional, merged via script)
├── scripts/
│   ├── train_mobilenetv3.py   ← Training script (in progress)
│   ├── preprocess_dataset.py  ← (Optional dataset merger)
│   └── utils.py               ← (Optional helpers)
├── models/
│   └── mobilenetv3_roadobject.pt
├── outputs/
│   ├── logs/
│   └── plots/                 ← Training metrics
├── inference/
│   └── infer_single_image.py  ← Future: for S22 inference testing
├── README.md
└── requirements.txt
