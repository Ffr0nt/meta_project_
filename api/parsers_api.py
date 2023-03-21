import requests
import json
from bs4 import BeautifulSoup as bs

headers = {
    'Accept-Encoding': 'gzip, deflate, sdch',
    'Accept-Language': 'en-US,en;q=0.8',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Cache-Control': 'max-age=0',
    'Connection': 'keep-alive',
}


class HTTP_error(Exception):
    def __init__(self, status, link):
        self.status = status
        self.link = link


def parse_IGN(URL: str,
              save_path: str,
              json_save: bool = False
              ):
    """
    Gets information about review about game on a page (name of game,
    headline of article, text of review) and returns it as dict. Works only with IGN!
    IGN main page - https://www.ign.com/
    :param URL: str
        page url
    :param  save_path: str
        Path to save data on json format
    :return: None or dict
        if dict - full of info (keys: name, name_review, ref, text)
    """

    # get HTTP page by GET request
    r = requests.get(URL, headers=headers)

    # HTTP error handler (f.e. for 404 request status)
    if not r.ok:
        raise HTTP_error(r.status_code, r.url)  # that going to be handled outside

    # HTML code scrubbing start
    soup = bs(r.text, "html.parser")
    game_header = soup.find(class_="display-title jsx-4038437347")

    games_rewievs = soup.find(class_="article-page")
    games_verdict = soup.find(class_="jsx-3103488995 article-section")
    # HTML code scrubbing end

    # Dictionary with data forming
    game_full_rewiev = games_rewievs.text + games_verdict.text
    data = {'name_review': game_header.text,
            'ref': URL,
            'date': 0,
            'text': game_full_rewiev}

    # Save dictionary as json or return
    if json_save:
        with open(save_path, 'w') as f:
            json.dump(data)
        return
    else:
        return data
