"""
processed_gadf_fine_4096/ 의 fine-grained 도메인을 RPM 그룹별로 병합.

결과: processed_gadf_coarse_4096/
  500/ ← 500 + 502 + 504 샘플 (심볼릭 링크)
  600/ ← 600 + 602 + 604
  700/ ← 700 + 702 + 704
  800/ ← 800 + 802 + 804

원본 파일명 충돌 방지: 502 파일은 '502_원본명', 504는 '504_원본명' prefix 붙임.
"""
import os
import sys

SRC = "processed_gadf_fine_4096"
DST = "processed_gadf_coarse_4096"

GROUPS = {
    "500": ["500", "502", "504"],
    "600": ["600", "602", "604"],
    "700": ["700", "702", "704"],
    "800": ["800", "802", "804"],
}
SPLITS = ["normal", "anomaly"]


def make_coarse():
    src_root = os.path.abspath(SRC)
    dst_root = os.path.abspath(DST)

    total_links = 0
    for coarse_name, fine_domains in GROUPS.items():
        for split in SPLITS:
            dst_dir = os.path.join(dst_root, coarse_name, split)
            os.makedirs(dst_dir, exist_ok=True)

            for fine in fine_domains:
                src_dir = os.path.join(src_root, fine, split)
                if not os.path.isdir(src_dir):
                    print(f"  [skip] {src_dir} not found")
                    continue

                for fname in sorted(os.listdir(src_dir)):
                    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                        continue

                    # 파일명 충돌 방지: 기본 도메인(x00) 이외에는 도메인 번호 prefix
                    if fine == coarse_name:
                        dst_fname = fname
                    else:
                        dst_fname = f"{fine}_{fname}"

                    src_path = os.path.join(src_dir, fname)
                    dst_path = os.path.join(dst_dir, dst_fname)

                    if os.path.lexists(dst_path):
                        os.remove(dst_path)
                    os.symlink(src_path, dst_path)
                    total_links += 1

    print(f"Done. Created {total_links} symlinks in {dst_root}")
    for coarse_name in GROUPS:
        for split in SPLITS:
            d = os.path.join(dst_root, coarse_name, split)
            n = len(os.listdir(d)) if os.path.isdir(d) else 0
            print(f"  {coarse_name}/{split}: {n} files")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    make_coarse()
