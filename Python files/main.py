from flask import Flask, jsonify, request
import os
import news_collection as nc
from Text_Preprocessing import get_summarized_article
from Clustering import cluster

app = Flask(__name__)  # Declaring the flask APP
app.config['JSON_SORT_KEYS'] = False
basedir = os.path.abspath(os.path.dirname(__file__))
print(basedir)


def news_summarization(text):
    lst = []
    print(2)
    df = nc.get_news_from_google(text)
    print(3)
    print(df['news'])
    for i, j, k in zip(df['title'], df['media'], df['news']):
        if len(k.split(' ')) > 100:
            print(k)
            lst.append([{"Title": i, "Source": j, 'Article': get_summarized_article(k)}])
    print(4)
    topics = cluster(df)
    return lst, topics


@app.route('/news')
def get_news():  # put application's code here
    print(1)
    text = request.args.get('text')
    print(5)
    lst,topics = news_summarization(text)
    print(lst)
    print(6)
    return jsonify(Top_5_Headlines= topics,Summarized_News=lst)


if __name__ == '__main__':
    app.run(debug=True)