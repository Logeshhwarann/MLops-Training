from flask import Flask, render_template, request

from utils import get_default_values, get_form_options, predict_severity

app = Flask(__name__)

options = get_form_options()
defaults = get_default_values()


@app.route("/", methods=["GET", "POST"])
def index():
    values = defaults.copy()
    result = None

    if request.method == "POST":
        for key in values:
            values[key] = request.form.get(key, values[key])
        result = predict_severity(values)

    return render_template(
        "index.html",
        options=options,
        values=values,
        result=result
    )


if __name__ == "__main__":
    app.run(debug=True)
