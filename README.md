# ANPR_with_ui_integration
This anpr ui supports streaming multiple streams at a time.
<br>
claude.py: contains the code with fixed size table,database integration and all ptz controls with zoom.
<br>
FinalMain.py: is used to run the whole pipeline.

<br><br>

### ANPR_Project Summary:
<br>
The main aim of the ANPR project is to detect number plates of vehicles and extract text from it.The overall pipeline of the project also supports multiple stream and can also support multiple models at a time from different sources.


### Steps and prerequisites for running ANPR Project:<br>
1.Install flet in your system as we have developed the UI in flet.<br>
2.Download the license_plate_detector.pt file weights. This model is traned for detecting number plates from vehicles.<br>
3.Intsall Paddle OCR in your system. Go to the following link ("https://paddlepaddle.github.io/PaddleOCR/latest/quick_start.html#_1") and install it according to your system i.e if the system contains GPU or CPU install it accordingly.<br>

