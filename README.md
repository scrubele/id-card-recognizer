# id-card-recogniser

Examples of program running:

```bash
$ python main.py # process all test folder images
$ python main.py 1.jpg 2.jpg 3.jpg # images from test folder  
$ python main.py --logs=INFO # set log level DEBUG, INFO, PROGRESS, WARNING, ERROR
```

Log levels:

```
+ DEBUG - show images
+ PROGRESS - save images 
```
Set default log level is possible in `settings.yaml` </br>
Log files:`logs/result.csv` or `logs/result.json`
