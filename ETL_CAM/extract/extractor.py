import os
import re
import time
import logging
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(SCRIPT_DIR / "extractor_cam.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class ExtractorCAM:
    def __init__(self):
        self.base_dir = PROJECT_DIR / "Datos_CAM"
        self.output_dir = SCRIPT_DIR / "data"
        self.output_file = self.output_dir / "cam_extraido.csv"

        self.tiempo_inicio = time.time()
        self.registros_extraidos = 0
        self.archivos_procesados = 0
        self.archivos_fallidos = 0

    def _obtener_anio(self, nombre_archivo: str):
        match = re.search(r"(20\d{2}|19\d{2})", nombre_archivo)
        return int(match.group(1)) if match else None

    def _normalizar_columnas(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = (
            df.columns
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
            .str.replace("°", "", regex=False)
            .str.replace("(", "", regex=False)
            .str.replace(")", "", regex=False)
            .str.replace("/", "_", regex=False)
            .str.replace(",", "", regex=False)
        )
        return df

    def _leer_csv(self, archivo: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(archivo, sep=";", decimal=",", encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(archivo, sep=";", decimal=",", encoding="latin-1")

    def _leer_excel(self, archivo: Path) -> pd.DataFrame:
        try:
            return pd.read_excel(archivo, engine="openpyxl")
        except ImportError as e:
            raise ImportError(
                "No se puede leer Excel porque falta openpyxl. "
                "Instalalo con: pip install openpyxl"
            ) from e

    def _leer_archivo(self, archivo: Path) -> pd.DataFrame:
        if archivo.suffix.lower() == ".csv":
            df = self._leer_csv(archivo)
        elif archivo.suffix.lower() in [".xlsx", ".xls"]:
            df = self._leer_excel(archivo)
        else:
            return pd.DataFrame()

        df = self._normalizar_columnas(df)
        return df

    def extraer(self) -> bool:
        if not self.base_dir.exists():
            logger.error(f"No existe la carpeta {self.base_dir}")
            return False

        registros = []

        logger.info(f"Extrayendo datos desde {self.base_dir}")

        for carpeta_estacion in self.base_dir.iterdir():
            if not carpeta_estacion.is_dir():
                continue

            estacion = carpeta_estacion.name

            for archivo in carpeta_estacion.iterdir():
                if archivo.suffix.lower() not in [".csv", ".xlsx", ".xls"]:
                    continue

                try:
                    logger.info(f"Procesando {archivo}")

                    df = self._leer_archivo(archivo)

                    if df.empty:
                        logger.warning(f"Archivo vacio omitido: {archivo}")
                        continue

                    df["estacion"] = estacion
                    df["anio"] = self._obtener_anio(archivo.name)
                    df["archivo_origen"] = archivo.name

                    registros.append(df)

                    self.archivos_procesados += 1
                    self.registros_extraidos += len(df)

                except Exception as e:
                    self.archivos_fallidos += 1
                    logger.error(f"Error procesando {archivo}: {e}")

        if not registros:
            logger.error("No se extrajo ningun registro")
            return False

        df_final = pd.concat(registros, ignore_index=True, sort=False)

        self.output_dir.mkdir(exist_ok=True)
        df_final.to_csv(self.output_file, index=False, sep=";", encoding="utf-8-sig")

        tiempo = round(time.time() - self.tiempo_inicio, 2)

        logger.info("Extraccion completada")
        logger.info(f"Archivo generado: {self.output_file}")
        logger.info(f"Registros extraidos: {self.registros_extraidos}")
        logger.info(f"Archivos procesados: {self.archivos_procesados}")
        logger.info(f"Archivos fallidos: {self.archivos_fallidos}")
        logger.info(f"Tiempo ejecucion: {tiempo}s")

        return True


if __name__ == "__main__":
    extractor = ExtractorCAM()
    exito = extractor.extraer()
    exit(0 if exito else 1)
