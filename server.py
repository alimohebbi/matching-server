from pprint import pprint

from flask import request
from flask import Response
from flask import Flask
import json

from server.rank_descriptors import rank_descriptors

app = Flask(__name__)


@app.route("/ranking", methods=['GET', 'POST'])
def handle_ranking():
    if request.method == 'POST':
        req_json = request.get_json()
        results = rank_descriptors(req_json)
        json_results = json.dumps(results)
        resp = Response(json_results, status=200, mimetype='application/json')
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