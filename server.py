from pprint import pprint

from flask import request
from flask import Response
from flask import Flask
import json

from server.rank_descriptors import score_descriptors, score_descriptors2

app = Flask(__name__)


@app.route("/ranking", methods=['GET', 'POST'])
def handle_ranking():
    if request.method == 'POST':
        req_json = request.get_json()
        # pprint(req_json)
        results = score_descriptors(req_json)
        resp = Response(results, status=200, mimetype='application/json')
        return resp
    else:
        return "The request is not valid."


@app.route("/exception", methods=['GET', 'POST'])
def handle_exception():
    if request.method == 'POST':
        req_json = request.get_json()
        print(req_json)

        resp = Response(repr('OK'), status=200, mimetype='application/json')
        return resp
    else:
        return "The request is not valid."


if __name__ == "__main__":
    app.run(debug=True)
