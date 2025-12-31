# coding=utf-8
# ===================== 【基准版】毫米波+实例增量+FasterRCNN+Naive策略 无任何报错 ✅ =====================
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
import torchvision.models.detection as detection_models
from PIL import Image
from torch.utils.data import Dataset, Subset
from torch import optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

# ---------------------- Avalanche 核心导入 ----------------------
from avalanche.benchmarks.utils import make_detection_dataset
from avalanche.benchmarks.utils.detection_dataset import detection_collate_fn
from avalanche.benchmarks.scenarios.deprecated.generators import ni_benchmark
from avalanche.training.supervised.naive_object_detection import ObjectDetectionTemplate
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import loss_metrics, timing_metrics
from avalanche.evaluation.metrics.detection import DetectionMetrics

# ===================== 自定义毫米波数据集类 =====================
class MillimeterWaveDetDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.transform = transform
        self.img_paths = sorted([p for p in self.img_dir.glob("*.png")])
        self.ann_paths = [self.ann_dir / (p.stem + ".xml") for p in self.img_paths]
        assert len(self.img_paths) == len(self.ann_paths), "图片和标注数量不匹配！"
        self.targets = [0] * len(self.img_paths)  # 解决Avalanche数据集校验报错

    def parse_xml(self, ann_path):
        tree = ET.parse(ann_path)
        root = tree.getroot()
        boxes = []
        obj_node = root.find("object")
        if obj_node is not None:
            for sub_others in obj_node.findall(".//others/others"):
                bbox = sub_others.find("bndbox")
                xmin = float(bbox.find("xmin").text)
                ymin = float(bbox.find("ymin").text)
                xmax = float(bbox.find("xmax").text)
                ymax = float(bbox.find("ymax").text)
                boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.empty((0, 4))
        labels = torch.ones(len(boxes), dtype=torch.int64)  # 1类目标 + 背景 = num_classes=2
        target = {
            "boxes": boxes, "labels": labels,
            "image_id": torch.tensor([hash(ann_path.stem)]),
            "area": (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]) if len(boxes)>0 else torch.empty(0),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64)
        }
        return target

    def __len__(self): return len(self.img_paths)
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        target = self.parse_xml(self.ann_paths[idx])
        if self.transform is not None:
            transformed = self.transform(image=np.array(img), bboxes=target["boxes"].numpy(), labels=target["labels"].numpy())
            img = transformed["image"]
            target["boxes"] = torch.tensor(transformed["bboxes"], dtype=torch.float32)
            target["labels"] = torch.tensor(transformed["labels"], dtype=torch.int64)
        return img, target

# ===================== 主函数 =====================
def main():
    # ---------------------- 固定配置参数 ----------------------
    IMG_DIR = "/media/data/qkl/cl/images/"
    ANN_DIR = "/media/data/qkl/cl/Annotations/"
    NUM_TASKS = 3        # 训练集切分3个增量任务
    BATCH_SIZE = 20
    NUM_CLASSES = 2      # 不可改！1类毫米波目标 + 1背景
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    TEST_RATIO = 0.2     # 20%测试集 80%训练集
    SEED = 42
    TRAIN_EPOCHS_PER_TASK = 10
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ---------------------- 数据增强区分 ----------------------
    train_transform = A.Compose([
        A.Resize(height=800, width=1024),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]))
    
    test_transform = A.Compose([
        A.Resize(height=800, width=1024),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]))

    # ---------------------- 训练/测试集划分 ----------------------
    full_dataset = MillimeterWaveDetDataset(IMG_DIR, ANN_DIR, transform=None)
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    split = int(np.floor(TEST_RATIO * dataset_size))
    train_indices, test_indices = indices[split:], indices[:split]

    train_dataset = MillimeterWaveDetDataset(IMG_DIR, ANN_DIR, transform=train_transform)
    train_dataset = Subset(train_dataset, train_indices)
    train_dataset.targets = [0]*len(train_dataset)

    test_dataset = MillimeterWaveDetDataset(IMG_DIR, ANN_DIR, transform=test_transform)
    test_dataset = Subset(test_dataset, test_indices)
    test_dataset.targets = [0]*len(test_dataset)

    # ---------------------- 数据集封装 ----------------------
    avalanche_train = make_detection_dataset(dataset=train_dataset, collate_fn=detection_collate_fn)
    avalanche_test = make_detection_dataset(dataset=test_dataset, collate_fn=detection_collate_fn)
    
    print(f"✅ 数据集划分完成 | 总数据量: {dataset_size} | 训练集: {len(avalanche_train)} | 测试集: {len(avalanche_test)}")

    # ---------------------- 增量基准构建 ----------------------
    benchmark = ni_benchmark(
        train_dataset=avalanche_train, test_dataset=avalanche_test,
        n_experiences=NUM_TASKS, shuffle=True, seed=SEED
    )

    # ---------------------- 模型+优化器+策略 ----------------------
    model = detection_models.fasterrcnn_resnet50_fpn(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    optimizer = optim.SGD(
        model.parameters(), lr=0.005, momentum=0.9,
        weight_decay=0.0005, nesterov=True
    )

    evaluator = EvaluationPlugin(
        loss_metrics(epoch=True, experience=True),  
        timing_metrics(epoch=True),
        DetectionMetrics(iou_types=["bbox"], default_to_coco=True, summarize_to_stdout=True),
        strict_checks=False,
        loggers=[InteractiveLogger()]
        )

    strategy = ObjectDetectionTemplate(
        model=model, optimizer=optimizer,
        train_mb_size=BATCH_SIZE, eval_mb_size=BATCH_SIZE, train_epochs=TRAIN_EPOCHS_PER_TASK, eval_every=1,
        device=DEVICE, evaluator=evaluator
    )

    # ---------------------- 训练+评估 ----------------------
    print("\n===== 开始增量训练 (Naive策略 无抗遗忘) =====")
    for task_id, exp in enumerate(benchmark.train_stream):
        print(f"\n--- 训练 增量任务 {task_id+1}/{NUM_TASKS} ---")
        strategy.train(exp)
        print(f"\n--- 用完整测试集评估 任务{task_id+1}训练效果 ---")
        strategy.eval(benchmark.test_stream)

    print("\n✅ 所有增量任务训练完成！")

if __name__ == "__main__":
    main()