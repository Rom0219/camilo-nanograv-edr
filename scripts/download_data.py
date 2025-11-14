# scripts/download_data.py
import os
import urllib.request

BASE_URL = "https://www.gw-openscience.org/archive/links/GW150914_4KHZ_R1/"

FILES = {
    "H-H1_LOSC_4_V1-1126259446-32.hdf5",
    "L-L1_LOSC_4_V1-1126259446-32.hdf5",
}

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def download_file(filename: str) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    url = BASE_URL + filename
    out_path = os.path.join(DATA_DIR, filename)

    if os.path.exists(out_path):
        print(f"[OK] {filename} ya existe en data/, se omite descarga.")
        return

    print(f"Descargando {url} -> {out_path}")
    urllib.request.urlretrieve(url, out_path)
    print(f"[OK] Guardado {out_path}")


def main() -> None:
    print("Descarga de datos GW150914 (H1 y L1) en la carpeta data/")
    for fname in FILES:
        download_file(fname)


if __name__ == "__main__":
    main()
