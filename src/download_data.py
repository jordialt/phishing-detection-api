import pandas as pd
import requests
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def download_from_openphish():
    url = "https://openphish.com/feed.txt"
    response = requests.get(url)
    if response.status_code == 200:
        urls = response.text.strip().split("\n")
        df = pd.DataFrame(urls, columns=["url"])
        df["label"] = "phishing"
        return df
    else:
        print("Error al descargar desde OpenPhish")
        return pd.DataFrame()

def download_from_kaggle(dataset_name):
    api = KaggleApi()
    api.authenticate()
    # Descargar el dataset completo como ZIP
    api.dataset_download_files(dataset_name, path="./data/", unzip=False)

    # Buscar el archivo ZIP descargado (puede variar el nombre)
    import os

    data_dir = "./data"
    zip_files = [f for f in os.listdir(data_dir) if f.endswith('.zip')]
    if not zip_files:
        print(f"No se encontró archivo .zip en {data_dir} después de la descarga")
        return pd.DataFrame()

    zip_path = os.path.join(data_dir, zip_files[0])
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
    except zipfile.BadZipFile:
        print(f"El archivo {zip_path} no es un ZIP válido")
        return pd.DataFrame()

    # Intentar encontrar un CSV dentro de la carpeta data
    csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not csv_files:
        # Revisar subdirectorios
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.endswith('.csv'):
                    csv_files.append(os.path.join(root, f))

    if not csv_files:
        print("No se encontró ningún archivo CSV tras la extracción del ZIP")
        return pd.DataFrame()

    # Leer el primer CSV encontrado (asumimos que contiene las URLs)
    csv_path = csv_files[0]
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error leyendo {csv_path}: {e}")
        return pd.DataFrame()

    return df

def get_legitimate_urls():
    # Lista de URLs legítimas de ejemplo (más amplia)
    urls = [
        # Motores de búsqueda
        "https://www.google.com",
        "https://www.bing.com",
        "https://www.duckduckgo.com",
        "https://www.yahoo.com",
        "https://www.startpage.com",
        "https://www.ecosia.org",
        # Redes sociales
        "https://www.facebook.com",
        "https://www.twitter.com",
        "https://www.instagram.com",
        "https://www.linkedin.com",
        "https://www.snapchat.com",
        "https://www.tiktok.com",
        "https://www.pinterest.com",
        # Desarrollo
        "https://www.github.com",
        "https://www.stackoverflow.com",
        "https://www.gitlab.com",
        "https://www.bitbucket.org",
        "https://developer.mozilla.org",
        "https://docs.python.org",
        "https://www.w3schools.com",
        "https://stackoverflow.com",
        # Noticias
        "https://www.bbc.com",
        "https://www.cnn.com",
        "https://www.nytimes.com",
        "https://www.reuters.com",
        "https://www.aljazeera.com",
        "https://www.theguardian.com",
        "https://www.wsj.com",
        "https://www.economist.com",
        "https://www.bloomberg.com",
        # Compras
        "https://www.amazon.com",
        "https://www.ebay.com",
        "https://www.aliexpress.com",
        "https://www.walmart.com",
        "https://www.target.com",
        "https://www.costco.com",
        "https://www.bestbuy.com",
        "https://www.etsy.com",
        # Tecnología
        "https://www.techcrunch.com",
        "https://www.theverge.com",
        "https://arstechnica.com",
        "https://www.wired.com",
        "https://www.tomshardware.com",
        "https://www.gsmarena.com",
        "https://www.cnet.com",
        "https://www.zdnet.com",
        # Educación
        "https://www.khanacademy.org",
        "https://www.coursera.org",
        "https://www.edx.org",
        "https://www.udemy.com",
        "https://www.scholar.google.com",
        "https://www.academia.edu",
        "https://www.researchgate.net",
        # Wikipedia
        "https://www.wikipedia.org",
        "https://es.wikipedia.org",
        "https://en.wikipedia.org",
        "https://fr.wikipedia.org",
        "https://de.wikipedia.org",
        "https://ja.wikipedia.org",
        # Gobierno (ejemplos genéricos)
        "https://www.un.org",
        "https://www.europa.eu",
        "https://www.gov.uk",
        "https://www.usa.gov",
        "https://www.whitehouse.gov",
        # Bancos (nombres genéricos, no usar reales)
        "https://www.bankofamerica.com",
        "https://www.chase.com",
        "https://www.wellsfargo.com",
        "https://www.citibank.com",
        "https://www.hsbc.com",
        "https://www.barclays.com",
        # Otros
        "https://www.apple.com",
        "https://www.microsoft.com",
        "https://www.adobe.com",
        "https://www.netflix.com",
        "https://www.spotify.com",
        "https://www.youtube.com",
        "https://www.vimeo.com",
        "https://www.twitch.tv",
        "https://www.discord.com",
        "https://www.slack.com",
        "https://www.dropbox.com",
        "https://drive.google.com",
        "https://www.office.com",
        # Salud
        "https://www.who.int",
        "https://www.cdc.gov",
        "https://www.mayoclinic.org",
        # Viajes
        "https://www.booking.com",
        "https://www.airbnb.com",
        "https://www.tripadvisor.com",
        "https://www.expedia.com",
        # Deportes
        "https://www.espn.com",
        "https://www.bbc.com/sport",
        "https://www.goal.com",
        "https://www.fifa.com",
        # Juegos
        "https://store.steampowered.com",
        "https://www.xbox.com",
        "https://www.playstation.com",
        # Otros
        "https://www.imdb.com",
        "https://www.rottentomatoes.com",
        "https://www.allrecipes.com",
        "https://www.weather.com",
        "https://www.accuweather.com",
        # Agrega tantas como necesites para equilibrar
    ]
    df = pd.DataFrame(urls, columns=["url"])
    df["label"] = "legitimate"
    return df

def combine_datasets():
    # Descargar de OpenPhish
    df_openphish = download_from_openphish()

    # Descargar de Kaggle
    # Nuevo dataset indicado por el usuario
    dataset_name = "taruntiwarihp/phishing-site-urls"
    df_kaggle = download_from_kaggle(dataset_name)

    # Combinar datasets
    df_combined = pd.concat([df_openphish, df_kaggle], ignore_index=True)

    # Añadir URLs legítimas
    df_legit = get_legitimate_urls()
    df_combined = pd.concat([df_combined, df_legit], ignore_index=True)

    # Eliminar duplicados
    df_combined.drop_duplicates(subset=["url"], keep="first", inplace=True)

    # Guardar
    df_combined.to_csv("./data/raw_data.csv", index=False)
    print("Datos descargados y guardados en ./data/raw_data.csv")
    return df_combined

if __name__ == "__main__":
    df = combine_datasets()