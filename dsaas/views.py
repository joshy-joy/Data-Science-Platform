"""
Routes and views for the flask application.
git add .
git commit -m "First commit"
git push origin master

Auther Arulanandam Kajavatan
"""

from datetime import datetime
from flask import render_template
from dsaas import app
from flask import Flask, redirect, url_for, session, escape , request
from werkzeug import secure_filename
import pandas as pd
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import io
import urllib
from io import StringIO
import base64
import json
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import seaborn as sns

#import g if you need to hold some memory
#from flask import g
# https://www.ablebits.com/office-addins-blog/2016/05/11/make-histogram-excel/
# https://towardsdatascience.com/https-towardsdatascience-com-algorithmic-trading-using-sentiment-analysis-on-news-articles-83db77966704 
"""
def get_messages():
    messages = getattr(g, '_messages', None)
    if messages is None:
        g._messages = []  # to store messages you may use a dictionary

    return g._messages

def add_message(message):
    messages = get_messages()
    messages.append(message)
    setattr(g, '_messages', messages)
    return messages
 """

@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
    )

@app.route('/contact')
def contact():
    """Renders the contact page."""
    return render_template(
        'contact.html',
        title='Contact',
        year=datetime.now().year,
        message='Your contact page.'
    )

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template(
        'about.html',
        title='About',
        year=datetime.now().year,
        message='Your application description page.'
    )

# Route for handling the login page logic
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            error = 'Invalid Credentials. Please try again.'
        else:
            session['username'] = request.form['username']
            return redirect(url_for('visualization'))
    return render_template('login.html', title = 'Login', year=datetime.now().year ,error=error)

# Route for handling the signup page logic
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = None
    if request.method == 'POST':
        return render_template('thankyou.html', title = 'Thank you', year=datetime.now().year ,error=error,
                              message='Please check your email! we will send you the confirmation about your account setup')
    return render_template('login.html', title = 'Login', year=datetime.now().year ,error=error)
    

@app.route('/services')
def services():
    """Renders the services page."""
    if 'username' in session:    
        username = session['username']
        return render_template('services.html',title='Services',year=datetime.now().year, message='Your application description page.')
    return render_template('index.html',title='Home Page',year=datetime.now().year)

@app.route('/logout')
def logout():
   # remove the username from the session if it is there
   session.pop('username', None)
   return redirect(url_for('home'))


@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      tempfile_path = tempfile.NamedTemporaryFile().name
      f.save(tempfile_path)
      sheet = pd.read_csv(tempfile_path)
      return_url = request.referrer
      redirecturl = return_url.split('/')
      if redirecturl[len(redirecturl)-1] == 'bubblechart':
          str_height = request.form['height']
          if (str_height == ''):
              str_height = 9
          str_weight = request.form['width']
          if (str_weight == ''):
              str_weight = 12
          str_xrotation = request.form['xticks']
          if (str_xrotation == ''):
              str_xrotation = 0
          str_yrotation = request.form['yticks']
          if (str_yrotation == ''):
              str_yrotation = 0
          str_title = request.form['title']
          if (str_title == ''):
              str_title = 'Bubble Chart'
          xlabel = request.form['xlabel']
          ylabel = request.form['ylabel']

          graph1_url = buildbubblechart(sheet,xrotate = int(str_xrotation),yrotate = int(str_yrotation), width = int(str_weight) ,height = int(str_height),heading = str_title, xLabel = xlabel, yLabel =ylabel ); 
          return render_template('result.html', title='Bubble Chart', year=datetime.now().year, graph1=graph1_url)
      
      #boxplot
      if redirecturl[len(redirecturl)-1] == 'boxplot':
          str_height = request.form['height']
          if (str_height == ''):
              str_height = 9
          str_weight = request.form['width']
          if (str_weight == ''):
              str_weight = 12
          str_xrotation = request.form['xticks']
          if (str_xrotation == ''):
              str_xrotation = 0
          str_yrotation = request.form['yticks']
          if (str_yrotation == ''):
              str_yrotation = 0
          str_title = request.form['title']
          if (str_title == ''):
              str_title = 'Box Plot'
          xlabel = request.form['xlabel']
          ylabel = request.form['ylabel']
          graph1_url = buildboxplot(sheet,xrotate = int(str_xrotation),yrotate = int(str_yrotation), width = int(str_weight) ,height = int(str_height),heading = str_title, xLabel = xlabel, yLabel =ylabel ); 
          return render_template('result.html', title='Box Plot', year=datetime.now().year, graph1=graph1_url)
      
      #Connectedscatterplot
      if redirecturl[len(redirecturl)-1] == 'Connectedscatterplot':
          str_height = request.form['height']
          if (str_height == ''):
              str_height = 9
          str_weight = request.form['width']
          if (str_weight == ''):
              str_weight = 12
          str_xrotation = request.form['xticks']
          if (str_xrotation == ''):
              str_xrotation = 0
          str_yrotation = request.form['yticks']
          if (str_yrotation == ''):
              str_yrotation = 0
          str_title = request.form['title']
          if (str_title == ''):
              str_title = 'Box Plot'
          xlabel = request.form['xlabel']
          ylabel = request.form['ylabel']
          graph1_url = buildConnectedscatterplot(sheet,xrotate = int(str_xrotation),yrotate = int(str_yrotation), width = int(str_weight) ,height = int(str_height),heading = str_title, xLabel = xlabel, yLabel =ylabel ); 
          return render_template('result.html', title='Connected Scatter Chart', year=datetime.now().year, graph1=graph1_url)
      
      #linechart
      if redirecturl[len(redirecturl)-1] == 'linechart':
          str_height = request.form['height']
          if (str_height == ''):
              str_height = 9
          str_weight = request.form['width']
          if (str_weight == ''):
              str_weight = 12
          str_xrotation = request.form['xticks']
          if (str_xrotation == ''):
              str_xrotation = 0
          str_yrotation = request.form['yticks']
          if (str_yrotation == ''):
              str_yrotation = 0
          str_title = request.form['title']
          if (str_title == ''):
              str_title = 'Line Chart'
          xlabel = request.form['xlabel']
          ylabel = request.form['ylabel']
          graph1_url = buildLineChart(sheet,xrotate = int(str_xrotation),yrotate = int(str_yrotation), width = int(str_weight) ,height = int(str_height),heading = str_title, xLabel = xlabel, yLabel =ylabel ); 
          return render_template('result.html', title='Line Chart', year=datetime.now().year, graph1=graph1_url)
      
      #highlightedlinechart
      if redirecturl[len(redirecturl)-1] == 'highlightedlinechart':
          str_height = request.form['height']
          if (str_height == ''):
              str_height = 9
          str_weight = request.form['width']
          if (str_weight == ''):
              str_weight = 12
          str_xrotation = request.form['xticks']
          if (str_xrotation == ''):
              str_xrotation = 0
          str_yrotation = request.form['yticks']
          if (str_yrotation == ''):
              str_yrotation = 0
          str_title = request.form['title']
          if (str_title == ''):
              str_title = 'Highlighted Line Chart'
          xlabel = request.form['xlabel']
          ylabel = request.form['ylabel']
          graph1_url = buildhighlightedlinechart(sheet,xrotate = int(str_xrotation),yrotate = int(str_yrotation), width = int(str_weight) ,height = int(str_height),heading = str_title, xLabel = xlabel, yLabel =ylabel ); 
          return render_template('result.html', title='Highlighted Line Chart', year=datetime.now().year, graph1=graph1_url)
      
      #piechart
      if redirecturl[len(redirecturl)-1] == 'piechart':
          str_height = request.form['height']
          if (str_height == ''):
              str_height = 9
          str_weight = request.form['width']
          if (str_weight == ''):
              str_weight = 12          
          str_title = request.form['title']
          if (str_title == ''):
              str_title = 'Pie Chart'
          graph1_url = buildpiechart(sheet, width = int(str_weight) ,height = int(str_height),heading = str_title); 
          return render_template('result.html', title='Pie Chart', year=datetime.now().year, graph1=graph1_url)
      #return render_template('bubblechart.html',title='Bubble Chart',year=datetime.now().year)
      return render_template('visualization.html',title='Visualization Tools',year=datetime.now().year, 
                              message='Please select a tool and provide data your data in the same format. Our system will generate selected graph/plot')

@app.route('/chart')
def chart():
        #These coordinates could be stored in DB
    x1 = [0, 1, 2, 3, 4]
    y1 = [10, 30, 40, 5, 50]
    x2 = [0, 1, 2, 3, 4]
    y2 = [50, 30, 20, 10, 50]
    x3 = [0, 1, 2, 3, 4]
    y3 = [0, 30, 10, 5, 30]
 
    graph1_url = build_graph(x2,y2);
    graph2_url = build_graph(x2,y2);
    graph3_url = build_graph(x3,y3);

    return render_template(
        'chart.html',
        title='Chart',
        year=datetime.now().year,
        message='Rendering image',
        graph1=graph1_url,
        graph2=graph2_url,
        graph3=graph3_url     
    )

def build_graph(x_coordinates, y_coordinates):
    img = io.BytesIO()
    plt.plot(x_coordinates, y_coordinates)
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return 'data:image/png;base64,{}'.format(graph_url)

@app.route('/visualization')
def visualization():
   # remove the username from the session if it is there
   if 'username' in session:    
       username = session['username']
       return render_template('visualization.html',title='Visualization Tools',year=datetime.now().year, 
                              message='Please select a tool and provide data your data in the same format. Our system will generate selected graph/plot')
   return render_template('index.html',title='Home Page',year=datetime.now().year)

@app.route('/forecasting')
def forecasting():
   if 'username' in session:    
       username = session['username']
       return render_template('forecasting.html',title='Forecasting Tools',year=datetime.now().year)
   return render_template('index.html',title='Home Page',year=datetime.now().year)

@app.route('/machine_learning')
def machine_learning():
   if 'username' in session:    
       username = session['username']
       return render_template('machine_learning.html',title='Machine learning',year=datetime.now().year)
   return render_template('index.html',title='Home Page',year=datetime.now().year)

@app.route('/trading_finance')
def trading_finance():
   if 'username' in session:    
       username = session['username']
       return render_template('trading_finance.html',title='Trading & Finance',year=datetime.now().year)
   return render_template('index.html',title='Home Page',year=datetime.now().year)


@app.route('/deep_learning')
def deep_learning():
   if 'username' in session:    
       username = session['username']
       return render_template('deep_learning.html',title='Deep learning',year=datetime.now().year)
   return render_template('index.html',title='Home Page',year=datetime.now().year)

@app.route('/rule_mining')
def rule_mining():
   if 'username' in session:    
       username = session['username']
       return render_template('rule_mining.html',title='Rule mining',year=datetime.now().year)
   return render_template('index.html',title='Home Page',year=datetime.now().year)

@app.route('/statistic')
def statistic():
   if 'username' in session:    
       username = session['username']
       return render_template('statistic.html',title='Statistical Tools',year=datetime.now().year)
   return render_template('index.html',title='Home Page',year=datetime.now().year)

@app.route('/bubblechart')
def bubblechart():
   if 'username' in session:    
       username = session['username']
       return render_template('bubblechart.html',title='Bubble Chart',year=datetime.now().year)
   return render_template('index.html',title='Home Page',year=datetime.now().year)

def buildbubblechart(sheet,xrotate = 0, yrotate = 0, width = 12 ,height =9 ,heading = "Box Plot", xLabel = '', yLabel =''):
    img = io.BytesIO()
    plt.figure(figsize=(width,height))
    #x = np.random.rand(40)
    #y = np.random.rand(40)
    #z = np.random.rand(40)
    x=sheet.iloc[:,0].values
    y=sheet.iloc[:,1].values
    z=sheet.iloc[:,2].values/1000

    #https://matplotlib.org/1.3.1/users/recipes.html
    #https://plot.ly/matplotlib/figure-labels/
    #fig = plt.figure()
 
    # use the scatter function
    plt.scatter(x, y, s=z, alpha=0.5)
    plt.xticks(rotation=xrotate)
    plt.yticks(rotation=yrotate)    
    #fig.suptitle('test title', fontsize=20)
    plt.title(heading)
    plt.xlabel(xLabel, fontsize=16)
    plt.ylabel(yLabel, fontsize=16)
    #fig.autofmt_xdate()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return 'data:image/png;base64,{}'.format(graph_url)

#boxplot
@app.route('/boxplot')
def boxplot():
   if 'username' in session:    
       username = session['username']
       return render_template('boxplot.html',title='Box Plot',year=datetime.now().year)
   return render_template('index.html',title='Home Page',year=datetime.now().year)

def buildboxplot(sheet,xrotate = 0, yrotate = 0, width = 12 ,height =9 ,heading = "Box Plot", xLabel = '', yLabel =''):
    img = io.BytesIO()

    # Dataset:
    #list(sheet.columns.values)
    # The folling coding add the first column twise but it wont affect since same group.
    dfa = pd.DataFrame({ 'group' : sheet.columns[0], 'value': sheet.iloc[:,0] })
    columnIndex = 0
    for column in sheet:
        print(column)
        b = pd.DataFrame({ 'group' : column, 'value': sheet.iloc[:,columnIndex] })
        dfa = dfa.append(b)
        columnIndex = columnIndex + 1
    
    plt.figure(figsize=(width,height))
    # Usual boxplot
    sns.boxplot(x='group', y='value', data=dfa)
    plt.xticks(rotation=xrotate)
    plt.yticks(rotation=yrotate)
    plt.title(heading)

    plt.xlabel(xLabel, fontsize=16)
    plt.ylabel(yLabel, fontsize=16)
    
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return 'data:image/png;base64,{}'.format(graph_url)

#Connectedscatterplot
@app.route('/Connectedscatterplot')
def Connectedscatterplot():
   if 'username' in session:    
       username = session['username']
       return render_template('Connectedscatterplots.html',title='Connected Scatter Chart',year=datetime.now().year)
   return render_template('index.html',title='Home Page',year=datetime.now().year)

def buildConnectedscatterplot(sheet,xrotate = 0, yrotate = 0, width = 12 ,height = 9 ,heading = "Connected Scatterplot", xLabel = '', yLabel =''):
    # https://python-graph-gallery.com/132-basic-connected-scatterplot/
    img = io.BytesIO()
    plt.figure(figsize=(width,height))
    # data
    df=pd.DataFrame({'x': sheet.iloc[:,0], 'y': sheet.iloc[:,1] }) 
    # plot
    plt.plot( 'x', 'y', data=df, linestyle='-', marker='o')
    plt.xticks(rotation=xrotate)
    plt.yticks(rotation=yrotate)
    plt.title(heading)

    plt.xlabel(xLabel, fontsize=16)
    plt.ylabel(yLabel, fontsize=16)

    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return 'data:image/png;base64,{}'.format(graph_url)

#linechart
@app.route('/linechart')
def linechart():
   if 'username' in session:    
       username = session['username']
       return render_template('linechart.html',title='Line Chart',year=datetime.now().year)
   return render_template('index.html',title='Home Page',year=datetime.now().year)

def buildLineChart(sheet,xrotate = 0, yrotate = 0, width = 12 ,height = 9 ,heading = "Line Chart", xLabel = '', yLabel =''):
    img = io.BytesIO()
    plt.figure(figsize=(width,height))
    # data
    df=pd.DataFrame({'x': sheet.iloc[:,0], 'y': sheet.iloc[:,1] }) 
    # plot
    plt.plot( 'x', 'y', data=df, linestyle='-')
    plt.xticks(rotation=xrotate)
    plt.yticks(rotation=yrotate)
    plt.title(heading)

    plt.xlabel(xLabel, fontsize=16)
    plt.ylabel(yLabel, fontsize=16)

    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return 'data:image/png;base64,{}'.format(graph_url)

#highlightedlinechart
@app.route('/highlightedlinechart')
def highlightedlinechart():
   if 'username' in session:    
       username = session['username']
       return render_template('highlightedlinechart.html',title='Highlighted Line Chart',year=datetime.now().year)
   return render_template('index.html',title='Home Page',year=datetime.now().year)

def buildhighlightedlinechart(sheet,xrotate = 0, yrotate = 0, width = 12 ,height = 9 ,heading = "Line Chart", xLabel = '', yLabel =''):
    img = io.BytesIO()
    plt.figure(figsize=(width,height))

    # multiple line plot
    columnCount = 0
    for column in sheet:
        if columnCount+1 <int(sheet.shape[1]):
            plt.plot(sheet.iloc[:,0], sheet.iloc[:,columnCount+1], marker='', color='grey', linewidth=1, alpha=0.4)
            columnCount = columnCount + 1
 
    # Now re do the interesting curve, but biger with distinct color
    plt.plot(sheet.iloc[:,0], sheet.iloc[:,columnCount], marker='', color='orange', linewidth=4, alpha=0.7)
 
    # Change xlim
    #plt.xlim(0,12)
 
    # Let's annotate the plot
    for x in range(1,columnCount-1):
        name=list(sheet)[x]
        plt.text(sheet.iloc[-1,0], sheet.iloc[-1,x], name, horizontalalignment='left', size='small', color='grey')
 
    # And add a special annotation for the group we are interested in
    plt.text(sheet.iloc[-1,0], sheet.iloc[-1,columnCount], list(sheet)[columnCount], horizontalalignment='left', size='small', color='orange')
 
    # Add titles
    plt.title(heading, loc='left', fontsize=12, fontweight=0, color='orange')
    

    plt.xticks(rotation=xrotate)
    plt.yticks(rotation=yrotate)
    #plt.title(heading)
    plt.xlabel(xLabel, fontsize=16)
    plt.ylabel(yLabel, fontsize=16)
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return 'data:image/png;base64,{}'.format(graph_url)


#piechart
@app.route('/piechart')
def piechart():
   if 'username' in session:    
       username = session['username']
       return render_template('piechart.html',title='Pie Chart',year=datetime.now().year)
   return render_template('index.html',title='Home Page',year=datetime.now().year)

def buildpiechart(sheet, width = 12 ,height = 9 ,heading = "Line Chart"):
    img = io.BytesIO()
    plt.figure(figsize=(width,height))
    sheet.dropna(axis = 0, how = 'all')

    if int(sheet.shape[1]) == 4:
        labels = sheet.iloc[:,0].values 
        sizes = sheet.iloc[:,1].values  
        explode = sheet.iloc[:,2].values 
        colors = sheet.iloc[:,3].values 
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.0f%%', shadow=True, startangle=100)
    else:
        labels = sheet.iloc[:,0].values 
        sizes = sheet.iloc[:,1].values  
        explode = sheet.iloc[:,2].values 
        # Plot
        plt.pie(sizes, explode=explode, labels=labels,
        autopct='%1.0f%%', shadow=True, startangle=100)

    
 
    plt.axis('equal')
    # Change xlim
    #plt.xlim(0,12)
    
    # Add titles
    plt.title(heading) #, loc='left', fontsize=12, fontweight=0, color='orange')
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return 'data:image/png;base64,{}'.format(graph_url)