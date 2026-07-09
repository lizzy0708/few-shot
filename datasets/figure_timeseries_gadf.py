#원본 시계열 데이터 -> 이미지로 변환이 어떻게 되는지 그림 (학회에 넣을 그림)

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pyts.image import GramianAngularField
from PIL import Image

WINDOW_SIZE = 1024
IMAGE_SIZE = 128   # 저장 크기 (make_gadf.py 기준)
MODEL_SIZE = 224   # 모델 입력 크기 (hust_image.py resize 기준)

files = {
    "Normal (N800)":  ("../HUST bearing dataset/N800.mat",  0),
    "Ball Fault (B700)": ("../HUST bearing dataset/B700.mat", 0),
    "Outer Fault (O800)": ("../HUST bearing dataset/O800.mat", 0),
}

gaf = GramianAngularField(image_size=IMAGE_SIZE, method='difference')

fig = plt.figure(figsize=(14, 4 * len(files)))
outer = gridspec.GridSpec(len(files), 1, figure=fig, hspace=0.5)

for row, (title, (path, start)) in enumerate(files.items()):
    mat = sio.loadmat(path)
    signal = mat["data"].squeeze()

    segment = signal[start: start + WINDOW_SIZE].astype(np.float64)
    segment_norm = (segment - segment.min()) / (segment.max() - segment.min() + 1e-8)
    segment_norm = 2 * segment_norm - 1

    gadf_img = gaf.fit_transform(segment_norm.reshape(1, -1))[0]

    inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[row],
                                             width_ratios=[5, 0.5, 5], wspace=0.05)

    # 왼쪽: 시계열
    ax_ts = fig.add_subplot(inner[0])
    ax_ts.plot(np.arange(WINDOW_SIZE), segment, color="#2196F3", linewidth=0.8)
    ax_ts.set_title(f"{title}\nRaw Signal (1024 samples)", fontsize=10)
    ax_ts.set_xlabel("Sample")
    ax_ts.set_ylabel("Amplitude")
    ax_ts.spines["top"].set_visible(False)
    ax_ts.spines["right"].set_visible(False)

    # 가운데: 화살표
    ax_arrow = fig.add_subplot(inner[1])
    ax_arrow.annotate("", xy=(0.9, 0.5), xytext=(0.1, 0.5),
                      xycoords="axes fraction",
                      arrowprops=dict(arrowstyle="->", lw=2, color="gray"))
    ax_arrow.text(0.5, 0.35, "GADF", ha="center", va="center",
                  transform=ax_arrow.transAxes, fontsize=9, color="gray")
    ax_arrow.axis("off")

    # 오른쪽: GADF 이미지 → 224×224로 리사이즈 (모델 실제 입력과 동일)
    ax_img = fig.add_subplot(inner[2])
    img_uint8 = (gadf_img * 255).astype(np.uint8)
    img_resized = np.array(Image.fromarray(img_uint8).resize((MODEL_SIZE, MODEL_SIZE), Image.BILINEAR))
    ax_img.imshow(img_resized, cmap="gray", vmin=0, vmax=255)
    ax_img.set_title("GADF Image (224×224)", fontsize=10)
    ax_img.set_xticks([])
    ax_img.set_yticks([])

fig.suptitle("Time Series → GADF Image Transformation\n(HUST Bearing Dataset)",
             fontsize=13, fontweight="bold", y=1.01)

plt.savefig("gadf_transform_preview.png", dpi=180, bbox_inches="tight", transparent=True)
plt.show()
print("저장 완료: gadf_transform_preview.png")
