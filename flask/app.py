
# import the Flask class from the flask module
from flask import Flask, render_template, redirect, url_for, request
import model_predict

# create the application object
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0



# link main function to root
@app.route('/', methods=['GET', 'POST'])
def predict():
    pred_result = None
    if request.method == 'POST':
        print(request.form['comment_text'])
        text = request.form['comment_text']
        print(type(text))
        print(text)
        if model_predict.clean_input_str(text) == '':
            print('empty string')
            return redirect('/')
        else:
            
            pred_result = model_predict.get_pred_exp(text)
            print(type(pred_result))

            pred_result = pred_result.replace('var pp = new lime.PredictProba(pp_svg, ["0", "1"]', 
                                'var pp = new lime.PredictProba(pp_svg, ["Not Removed", "Removed"]')
            pred_result = pred_result.replace('var exp = new lime.Explanation(["0", "1"]);',
                                              'var exp = new lime.Explanation(["Not Removed", "Removed"]);')

            f = open('static/output.html', 'w', encoding='utf8')
            f.write(pred_result)
            f.flush()
            f.close()

        return pred_result
    return render_template('predict.html', pred_result = pred_result)

# start the server with the 'run()' method
if __name__ == '__main__':
    app.run(debug=True)
