import logging
import time
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]

INPUT_FILE = PROJECT_DIR / "ETL_CAM" / "extract" / "data" / "cam_extraido.csv"
OUTPUT_DIR = SCRIPT_DIR / "data"
OUTPUT_FILE = OUTPUT_DIR / "cam_transformado.csv"
LOG_FILE = SCRIPT_DIR / "transformador_cam.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class TransformadorCAM:
    def __init__(self):
        self.input_file = INPUT_FILE
        self.output_dir = OUTPUT_DIR
        self.output_file = OUTPUT_FILE
        self.tiempo_inicio = time.time()
        self.registros_entrada = 0
        self.registros_salida = 0
        self.duplicados_eliminados = 0

    def _normalizar_nombres_columnas(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = (
            df.columns
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
            .str.replace("°", "", regex=False)
            .str.replace("Â°", "", regex=False)
            .str.replace("(", "", regex=False)
            .str.replace(")", "", regex=False)
            .str.replace("/", "_", regex=False)
            .str.replace(",", "", regex=False)
        )
        return df

    def _coalescer_columnas(self, df: pd.DataFrame, columnas_origen: list[str], columna_destino: str) -> pd.DataFrame:
        existentes = [col for col in columnas_origen if col in df.columns]

        if not existentes:
            df[columna_destino] = pd.NA
            return df

        df[columna_destino] = df[existentes].bfill(axis=1).iloc[:, 0]
        return df

    def _convertir_fecha(self, df: pd.DataFrame) -> pd.DataFrame:
        self._coalescer_columnas(df, ["fecha_hora", "fecha"], "fecha_hora")
        fechas_originales = df["fecha_hora"].astype("string").str.strip()
        df["fecha_hora"] = pd.to_datetime(
            fechas_originales,
            errors="coerce",
            dayfirst=True,
            format="mixed",
        )

        fechas_invalidas = df["fecha_hora"].isna() & fechas_originales.notna() & (fechas_originales != "")
        if fechas_invalidas.any():
            logger.warning(f"Fechas no interpretadas: {int(fechas_invalidas.sum())}")

        return df

    def _convertir_numericos(self, df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
        for columna in columnas:
            if columna not in df.columns:
                df[columna] = pd.NA

            df[columna] = (
                df[columna]
                .astype("string")
                .str.strip()
                .str.replace(",", ".", regex=False)
                .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
            )
            df[columna] = pd.to_numeric(df[columna], errors="coerce")

        return df

    def _limpiar_textos(self, df: pd.DataFrame) -> pd.DataFrame:
        for columna in ["estacion", "archivo_origen"]:
            if columna not in df.columns:
                df[columna] = pd.NA

            df[columna] = df[columna].astype("string").str.strip()

        return df

    def _validar_rangos(self, df: pd.DataFrame) -> pd.DataFrame:
        reglas = {
            "temperatura_c": (-10, 60),
            "humedad_pct": (0, 100),
            "radiacion_solar_w_m2": (0, 1600),
            "lluvia_mm": (0, 500),
            "direccion_viento_grados": (0, 360),
            "velocidad_viento_m_s": (0, 80),
            "presion_atmosferica_hpa": (500, 1100),
            "voltaje_bateria_vdc": (0, 30),
            "nivel_m": (-10, 100),
            "velocidad_agua_m_s": (0, 20),
        }

        for columna, (minimo, maximo) in reglas.items():
            if columna not in df.columns:
                continue

            fuera_rango = df[columna].notna() & ~df[columna].between(minimo, maximo)
            df.loc[fuera_rango, columna] = pd.NA

        return df

    def _crear_columnas_calidad(self, df: pd.DataFrame, columnas_medicion: list[str]) -> pd.DataFrame:
        df["fecha_valida"] = df["fecha_hora"].notna()
        df["tiene_medicion"] = df[columnas_medicion].notna().any(axis=1)
        df["registro_valido"] = df["fecha_valida"] & df["estacion"].notna() & df["tiene_medicion"]

        df["cantidad_variables_disponibles"] = df[columnas_medicion].notna().sum(axis=1)
        df["cantidad_variables_faltantes"] = df[columnas_medicion].isna().sum(axis=1)

        return df

    def transformar(self) -> bool:
        if not self.input_file.exists():
            logger.error(f"No existe el archivo de entrada: {self.input_file}")
            return False

        logger.info(f"Cargando datos desde {self.input_file}")
        df = pd.read_csv(self.input_file, sep=";", encoding="utf-8-sig", low_memory=False)
        self.registros_entrada = len(df)

        logger.info(f"Registros de entrada: {self.registros_entrada}")

        df = self._normalizar_nombres_columnas(df)
        df = self._convertir_fecha(df)
        df = self._limpiar_textos(df)

        self._coalescer_columnas(df, ["humedad_%", "humedad_relativa_%"], "humedad_pct")
        self._coalescer_columnas(df, ["presion_atmosferica_hpa", "pres_atmosferica_hpa"], "presion_atmosferica_hpa")
        self._coalescer_columnas(df, ["radiacion_solar_w_m2", "rad_u_w_m2"], "radiacion_solar_w_m2")
        self._coalescer_columnas(df, ["direccion_viento_", "direccion_viento_m_s"], "direccion_viento_grados")

        columnas_medicion = [
            "nivel_m",
            "temperatura_c",
            "humedad_pct",
            "radiacion_solar_w_m2",
            "lluvia_mm",
            "direccion_viento_grados",
            "velocidad_viento_m_s",
            "presion_atmosferica_hpa",
            "voltaje_bateria_vdc",
            "velocidad_agua_m_s",
        ]

        df = self._convertir_numericos(df, columnas_medicion + ["anio"])
        df = self._validar_rangos(df)

        columnas_finales = [
            "fecha_hora",
            "estacion",
            "anio",
            "archivo_origen",
            *columnas_medicion,
        ]

        df = df[columnas_finales]
        df = df.dropna(subset=["fecha_hora", "estacion"])

        antes_duplicados = len(df)
        df = df.sort_values(["estacion", "fecha_hora", "archivo_origen"])
        df = df.groupby(["estacion", "fecha_hora"], as_index=False).first()
        self.duplicados_eliminados = antes_duplicados - len(df)

        df = self._crear_columnas_calidad(df, columnas_medicion)
        self.registros_salida = len(df)

        self.output_dir.mkdir(exist_ok=True)
        df.to_csv(self.output_file, index=False, sep=";", encoding="utf-8-sig")

        tiempo = round(time.time() - self.tiempo_inicio, 2)

        logger.info("Transformacion completada")
        logger.info(f"Archivo generado: {self.output_file}")
        logger.info(f"Registros entrada: {self.registros_entrada}")
        logger.info(f"Registros salida: {self.registros_salida}")
        logger.info(f"Duplicados eliminados: {self.duplicados_eliminados}")
        logger.info(f"Tiempo ejecucion: {tiempo}s")

        return True


if __name__ == "__main__":
    transformador = TransformadorCAM()
    exito = transformador.transformar()
    exit(0 if exito else 1)
