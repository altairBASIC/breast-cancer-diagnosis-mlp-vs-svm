import streamlit as st

# Descripciones base de cada característica
FEATURE_DESCRIPTIONS = {
    "radius": "radio del tumor (distancia promedio del centro al borde).",
    "texture": "variación de la intensidad de la imagen; qué tan rugosa se ve la zona.",
    "perimeter": "perímetro del contorno del tumor.",
    "area": "área de la región del tumor en la imagen.",
    "smoothness": "qué tan suaves o irregulares son las variaciones del radio.",
    "compactness": "qué tan compacto es el tumor (relación perímetro² / área).",
    "concavity": "qué tan profundas son las zonas cóncavas del contorno.",
    "concave points": "cantidad de puntos cóncavos en el contorno.",
    "symmetry": "simetría de la forma del tumor.",
    "fractal dimension": "complejidad del borde (aproximación de dimensión fractal).",
}


def build_tooltip(feature: str) -> str:
    """Genera el texto del tooltip según si es media, error estándar o peor valor."""
    base = feature
    prefix = ""

    if feature.startswith("mean "):
        prefix = "Valor medio de "
        base = feature.replace("mean ", "")
    elif feature.endswith(" error"):
        prefix = "Error estándar de "
        base = feature.replace(" error", "")
    elif feature.startswith("worst "):
        prefix = "Peor valor de "
        base = feature.replace("worst ", "")

    desc = FEATURE_DESCRIPTIONS.get(base, base)
    return prefix + desc.capitalize()
