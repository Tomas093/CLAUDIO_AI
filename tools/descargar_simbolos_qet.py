"""
descargar_simbolos_qet.py — Descargador autónomo y concurrente de símbolos QElectroTech.
Crawl concurrente + Descarga concurrente con reporte en tiempo real.
"""

import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue, Empty
from urllib.parse import urljoin, unquote
import urllib.request

BASE_URL = "https://download.qelectrotech.org/qet/elements/10_electric/"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "Base de simbolols" / "elementos_electricos"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
}

def fetch_html(url: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=8) as response:
                return response.read().decode("utf-8", errors="ignore")
        except Exception:
            if attempt == retries - 1:
                return ""
            time.sleep(0.5)
    return ""

def download_file(url: str, dest_path: Path) -> bool:
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return True
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=10) as response:
            dest_path.write_bytes(response.read())
            return True
    except Exception:
        return False

def main():
    print(f"==================================================", flush=True)
    print(f"Descargador de Simbologia QElectroTech a Local", flush=True)
    print(f"Origen:  {BASE_URL}", flush=True)
    print(f"Destino: {OUTPUT_DIR}", flush=True)
    print(f"==================================================", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dir_queue = Queue()
    dir_queue.put(BASE_URL)
    
    visited_dirs = {BASE_URL}
    total_downloaded = 0
    total_discovered = 0

    download_pool = ThreadPoolExecutor(max_workers=16)
    futures = []

    print("[*] Iniciando descubrimiento y descarga concurrente...", flush=True)
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=10) as crawl_pool:
        while True:
            # Obtener lote de directorios para explorar
            batch = []
            while len(batch) < 10:
                try:
                    batch.append(dir_queue.get_nowait())
                except Empty:
                    break

            if not batch and not futures:
                # Si no hay más directorios y no hay descargas pendientes, terminamos
                break

            if batch:
                crawl_futures = {crawl_pool.submit(fetch_html, d_url): d_url for d_url in batch}
                for cf in as_completed(crawl_futures):
                    cur_url = crawl_futures[cf]
                    html = cf.result()
                    if not html:
                        continue

                    links = re.findall(r'href=[\x22\x27]([^\x22\x27#]+)[\x22\x27]', html)
                    for link in links:
                        link_str = link.strip()
                        if not link_str or link_str.startswith("?") or ".." in link_str:
                            continue
                        if link_str.endswith(".css") or link_str.endswith(".js") or link_str.endswith(".png"):
                            continue

                        full_url = urljoin(cur_url, link_str)
                        if not full_url.startswith(BASE_URL):
                            continue

                        if link_str.endswith(".elmt"):
                            rel_path = unquote(full_url[len(BASE_URL):]).replace("/", os.sep)
                            dest_file = OUTPUT_DIR / rel_path
                            fut = download_pool.submit(download_file, full_url, dest_file)
                            futures.append(fut)
                            total_discovered += 1
                        elif (link_str.endswith("/index.html") or link_str.endswith("/")) and full_url not in visited_dirs:
                            visited_dirs.add(full_url)
                            dir_queue.put(full_url)

            # Recolectar resultados de descargas terminadas
            still_running = []
            for f in futures:
                if f.done():
                    if f.result():
                        total_downloaded += 1
                else:
                    still_running.append(f)
            futures = still_running

            elapsed = int(time.time() - start_time)
            print(f"\rDirectorios: {len(visited_dirs)} | Descubiertos: {total_discovered} | Descargados: {total_downloaded} | Tiempo: {elapsed}s", end="", flush=True)
            time.sleep(0.1)

    # Esperar descargas restantes
    for f in as_completed(futures):
        if f.result():
            total_downloaded += 1

    print(f"\n\n[OK] Finalizado exitosamente.")
    print(f"Total de carpetas: {len(visited_dirs)}")
    print(f"Total de simbolos guardados en local: {total_downloaded}")
    print(f"Ruta local: {OUTPUT_DIR}", flush=True)

if __name__ == "__main__":
    main()
