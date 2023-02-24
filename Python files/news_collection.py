from GoogleNews import GoogleNews
import pandas as pd
import json
import requests
from bs4 import BeautifulSoup


gnews = GoogleNews(lang='en', region='India', period='7d')

def get_news_from_google(text):
    df = pd.read_excel('India_news_15_08.xlsx')
    #df = pd.read_pickle('India_news.pkl')
    #lst = []
    #df = pd.DataFrame()
    #gnews.search(text)  # Searching the topic
    #for i in range(1, 20):  # Storing the news to a list
    #    lst.append(gnews.page_at(i))
    #    print(gnews.page_at(i))
    #for i in range(19):  # Storing the news to a dataframe from the list.
    #    df = df.append(lst[i], ignore_index=True)
    df = df.dropna()
    #df.to_excel("Canada_08_15.xlsx")
    df['news'] = df['link'].apply(lambda x: extract_news(x))  # Function call to webscrape news from the article.
    # Return News from Dataframe which are not null
    return df[df['news'].notnull()]


def extract_news(link):
    agent = {
        "User-Agent": 'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.115 Safari/537.36'}
    r = requests.get(link, headers=agent)  # Reading the HTML Page
    # print(r.content)

    soup = BeautifulSoup(r.content, 'html5lib')  # Loading the HTMl to a beautifulSoup object
    # print(soup.prettify())

    try:
        if 'ndtv' not in link:
            table = soup.find_all('script', type='application/ld+json', text=True)  # Extracting the Script tags
            a = []  # List to store script tag contents

            # Iterating through Scripts tags and extracting article Body
            for i in range(len(table)):
                a.append((json.loads(table[i].contents[0])))

            # Iterating through the list of dictionaries to extract the article Body Key value.
            for i in a:
                if type(i) == list:
                    for j in i:
                        if 'articleBody' in j.keys():
                            return str(i['articleBody'])
                else:
                    if 'articleBody' in i.keys():
                        return str(i['articleBody'])
            else:
                return None
        else:
            table = soup.find_all('p', text=True)
            a = []
            for i in range(len(table)):
                a.append(table[i].contents[0])
            return str(a)
    except:
        return None
