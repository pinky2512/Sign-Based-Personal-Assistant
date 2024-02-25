import cv2
import operator

from string import ascii_uppercase

import tkinter as tk
from PIL import Image, ImageTk

from keras.models import model_from_json

import openai
import os

openai.api_key = os.environ.get("OPENAI")

def get_ai_response(input_text):
    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "user", "content": input_text}
      ]
    )
    return completion.choices[0].message.content


class Application:

    def __init__(self):
        
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None
        self.json_file = open("Models/model.json", "r")
        self.model_json = self.json_file.read()
        self.json_file.close()

        self.loaded_model = model_from_json(self.model_json)
        self.loaded_model.load_weights("Models/model.h5")

        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0

        for i in ascii_uppercase:
          self.ct[i] = 0
        
        print("Loaded model from disk")

        self.root = tk.Tk()
        self.root.title("Sign Language Personal Assistant")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("900x900")

        self.panel = tk.Label(self.root)
        self.panel.place(x = 100, y = 10, width = 580, height = 580)
        
        self.panel2 = tk.Label(self.root)
        self.panel2.place(x = 400, y = 65, width = 275, height = 275)

        self.T = tk.Label(self.root)
        self.T.place(x = 60, y = 5)
        self.T.config(text = "Sign Language Personal Assistant", font = ("Courier", 18, "bold"))

        self.panel3 = tk.Label(self.root)
        self.panel3.place(x = 500, y = 540)

        self.T1 = tk.Label(self.root)
        self.T1.place(x = 10, y = 540)
        self.T1.config(text = "Character :", font = ("Courier", 14, "bold"))

        self.panel4 = tk.Label(self.root) # Word
        self.panel4.place(x = 220, y = 595)

        self.T2 = tk.Label(self.root)
        self.T2.place(x = 10,y = 595)
        self.T2.config(text = "Word :", font = ("Courier", 14, "bold"))
        
        self.panel5 = tk.Label(self.root)
        self.panel5.place(x = 220, y = 540)
        
        self.T3 = tk.Label(self.root)
        self.T3.place(x = 10, y = 650)
        self.T3.config(text = "Response :", font = ("Courier", 14, "bold"))

        self.str = ""
        self.word = " "
        self.current_symbol = "Empty"
        self.photo = "Empty"
        self.character_read = False
        self.consecutive_spaces = 0  

        self.video_loop()


    def video_loop(self):
        ok, frame = self.vs.read()

        if ok:
            cv2image = cv2.flip(frame, 1)

            x1 = int(0.5 * frame.shape[1])
            y1 = 10
            x2 = frame.shape[1] - 10
            y2 = int(0.5 * frame.shape[1])

            cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0) ,1)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)

            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image = self.current_image)

            self.panel.imgtk = imgtk
            self.panel.config(image = imgtk)

            cv2image = cv2image[y1 : y2, x1 : x2]

            gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)

            blur = cv2.GaussianBlur(gray, (5, 5), 2)

            th3 = cv2.adaptiveThreshold(blur, 255 ,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

            _, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            self.predict(res)

            self.current_image2 = Image.fromarray(res)

            imgtk = ImageTk.PhotoImage(image = self.current_image2)

            self.panel2.imgtk = imgtk
            self.panel2.config(image = imgtk)

            self.panel3.config(text = self.current_symbol, font = ("Courier", 14))

            self.panel4.config(text = self.word, font = ("Courier", 14))

            if self.word.endswith(" "):
                if self.character_read: 
                    self.word += " "
                    self.character_read = False 
                    self.consecutive_spaces += 1 
                else:
                    self.consecutive_spaces = 0
            else:
                self.character_read = True
                self.consecutive_spaces = 0 

            if self.consecutive_spaces == 4:
                self.panel5.config(text  = get_ai_response(self.word), font = ("Courier", 14))

        self.root.after(5, self.video_loop)

    def predict(self, test_image):
        test_image = cv2.resize(test_image, (128, 128))
        result = self.loaded_model.predict(test_image.reshape(1, 128, 128, 1))

        prediction = {}
        prediction['blank'] = result[0][0]
        inde = 1

        for i in ascii_uppercase:
            prediction[i] = result[0][inde]
            inde += 1

        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        self.current_symbol = prediction[0][0]

        if self.current_symbol == 'blank':
            for i in ascii_uppercase:
                self.ct[i] = 0

        self.ct[self.current_symbol] += 1

        if self.ct[self.current_symbol] > 60:
            for i in ascii_uppercase:
                if i == self.current_symbol:
                    continue

                tmp = self.ct[self.current_symbol] - self.ct[i]

                if tmp < 0:
                    tmp *= -1

                if tmp <= 20:
                    self.ct['blank'] = 0

                    for i in ascii_uppercase:
                        self.ct[i] = 0
                    return

            self.ct['blank'] = 0

            for i in ascii_uppercase:
                self.ct[i] = 0

            if self.current_symbol == 'blank':
                if self.blank_flag == 0:
                    self.blank_flag = 1

                    if len(self.str) > 0:
                        self.str += " "

                    self.str += self.word
                    self.word = ""

            else:
                if len(self.str) > 16:
                    self.str = ""

                self.blank_flag = 0
                self.word += self.current_symbol
            
    def destructor(self):

        print("Closing Application...")

        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()
    
print("Starting Application...")

(Application()).root.mainloop()