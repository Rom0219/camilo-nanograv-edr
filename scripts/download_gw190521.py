# scripts/download_gw190521.py
"""
Descarga datos de strain H1 y L1 alrededor de GW190521 desde GWOSC
y los guarda en data/ en formato HDF5 de 32 s, 4 kHz.

Requiere tener instalado el paquete gwosc:
  conda install -y gwosc
o
  pip install gwosc
"""

import os
from gwosc import datasets
from gwosc.locate import get_event_urls

EVENT_NAME = "GW190521"
SAMPLE_RATE = 4096  # Hz (4 kHz)


def download_event_data(event: str, detector: str, out_dir: str) -> str:
    """
    Descarga el archivo HDF5 de 32 s y 4 kHz para un evento y detector dados,
    si no existe ya en out_dir. Devuelve la ruta local.
    """
    os.makedirs(out_dir, exist_ok=True)

    gps = datasets.event_gps(event)
    # Pedimos URLs para ±16 s alrededor del evento
    urls = get_event_urls(
        event=event,
        detector=detector,
        gps=gps,
        duration=32,
        sample_rate=SAMPLE_RATE,
        format="hdf5",
    )

    if not urls:
        raise RuntimeError(f"No se encontraron URLs GWOSC para {event} ({detector}).")

    url = urls[0]
    filename = os.path.basename(url)
    out_path = os.path.join(out_dir, filename)

    if os.path.exists(out_path):
        print(f"[{detector}] Ya existe {out_path}, no se descarga de nuevo.")
        return out_path

    print(f"[{detector}] Descargando {url}")
    import urllib.request

    with urllib.request.urlopen(url) as resp, open(out_path, "wb") as f:
        f.write(resp.read())

    print(f"[{detector}] Guardado en {out_path}")
    return out_path


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(repo_root, "data")

    for det in ["H1", "L1"]:
        try:
            path = download_event_data(EVENT_NAME, det, data_dir)
            print(f"{det}: listo -> {path}")
        except Exception as e:
            print(f"Error descargando datos para {det}: {e}")


if __name__ == "__main__":
    main()
