# eval.py
import os
import hydra
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from tqdm import tqdm
from omegaconf import DictConfig
from torchvision import datasets, transforms

# 学習時に利用しているものをインポート (例: conv_net, fix_seed 等)
from src.models.cnn import conv_net
from src.utils.utils import fix_seed

@hydra.main(version_base=None, config_path="configs", config_name="config")
def evaluate(args: DictConfig):
    """
    CIFAR-10の学習済みモデルを評価するためのスクリプト。
    torchvision.datasets.CIFAR10(train=False) を使って
    テストセットを取得し、AccuracyとLossを計算します。
    """
    # 乱数シードの固定
    fix_seed(args.seed)

    # 学習済みモデルのパス (Hydra から指定できるようにする)
    # 例: python eval.py ckpt_path="outputs/2023-01-01/0/model_best.pt"
    ckpt_path = args.get("ckpt_path", "model_best.pt")

    # 使用デバイス (config.yaml で device: "cuda" などを指定)
    device = args.device

    # -----------------------------------------
    #  1) テストデータセットを用意 (torchvision)
    # -----------------------------------------
    # 必要に応じて学習時と同じ変換を指定
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # 学習時と同じ正規化などを入れる場合
        # transforms.Normalize((0.4914, 0.4822, 0.4465),
        #                      (0.2023, 0.1994, 0.2010))
    ])

    # 公式のCIFAR10テストセット(ラベルあり)
    test_dataset = datasets.CIFAR10(
        root="data",     # データを保存するディレクトリ
        train=False,     # テスト用データ
        download=True,   # まだ無ければダウンロード
        transform=transform_test
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # -----------------------------------------
    #  2) モデルを定義 & 学習済み重みをロード
    # -----------------------------------------
    model = conv_net.to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # 評価用のLoss関数 & Accuracyメトリクス
    loss_fn = nn.CrossEntropyLoss()
    accuracy = Accuracy(task="multiclass", num_classes=10).to(device)

    # -----------------------------------------
    #  3) 推論ループで評価
    # -----------------------------------------
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            accuracy.update(outputs, labels)

    # 平均LossとAccuracyを計算
    avg_loss = total_loss / total_samples
    acc = accuracy.compute().item()

    print(f"[Test] Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")

if __name__ == "__main__":
    evaluate()

