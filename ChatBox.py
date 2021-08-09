import tkinter
from tkinter import *
from nltk_functions import predict_class, getResponse
import json

BG_COLOR = "#133337"
HEADTEXT_COLOR = "#FFFFFF"
TEXT_COLOR = "#000000"
BG_CHAT="#ede5f2"
LINE_COLOR="#000000"

FONT ="Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

intents = json.loads(open('intents.json').read())


class chatBox:

    def __init__(self):
        self.window = Tk()
        self._setup_main_window()

    def run(self):
        self.window.mainloop()

    def _setup_main_window(self):
        self.window.title("ChatRoom")
        self.window.resizable(height=False, width=False)
        self.window.configure(width=600,heigh=850, bg=BG_COLOR)

        #head label
        head_label = Label(self.window, bg=BG_COLOR, fg=HEADTEXT_COLOR, text="Welcome", font=FONT_BOLD, pady=17)
        head_label.place(relwidth=1)

        # tiny devider
        line = Label(self.window, width=540, bg=LINE_COLOR)
        line.place(relwidth=1, rely=0.08, relheight=0.015)

        #text widget
        self.text_widget = Text(self.window, width=20, wrap=WORD, heigh=2, bg=BG_CHAT, fg=TEXT_COLOR, font=FONT, padx=10,pady=5)
        self.text_widget.insert(tkinter.INSERT, "Ana: Hello! How can I help you?\n\n")
        self.text_widget.place(relheigh=0.57, relwidth=0.95, rely=0.1, relx=0.025,)
        self.text_widget.configure(cursor="arrow", state=DISABLED)
        #scroll bar
        scrollbar = Scrollbar(self.text_widget, command=self.text_widget.yview, cursor="arrow")
        scrollbar.place(relheight=1, relx=1)
        self.text_widget["yscrollcommand"]=scrollbar.set

        # tiny devider
        line = Label(self.window, width=540, bg=LINE_COLOR)
        line.place(relwidth=1, rely=0.67, relheight=0.015)

        #bottom label
        bottom_label=Label(self.window, bg=BG_COLOR, heigh=80)
        bottom_label.place(relwidth=1, rely=0.7)

        #entry label
        head_label = Label(bottom_label, bg=BG_COLOR, fg=HEADTEXT_COLOR, text="Entry Box", font=FONT_BOLD)
        head_label.place(relwidth=1, rely=0.001)

        #message entry box
        self.msg_entry = Entry(bottom_label, bg=BG_CHAT, fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.95, relheigh=0.06, rely=0.03, relx=0.025)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)

        #send button
        send_button = Button(bottom_label, text="Send", font=TEXT_COLOR, bg=BG_CHAT, width=20,
                             command=lambda: self._on_enter_pressed(None))
        send_button.place(relwidth=0.25, relheight=0.05, rely=0.1, relx=0.37)

    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get().strip()
        self.msg_entry.delete(0, END)
        self._insert_message(msg)

    def _insert_message(self, msg):
        if not msg:
            return
        self.text_widget.config(state=NORMAL)
        self.text_widget.insert(END, "You: " + msg + '\n\n')

        ints = predict_class(msg)
        res = getResponse(ints, intents)

        self.text_widget.insert(END, "Ana: " + res + '\n\n')

        self.text_widget.config(state=DISABLED)
        self.text_widget.yview(END)

app = chatBox()
app.run()


