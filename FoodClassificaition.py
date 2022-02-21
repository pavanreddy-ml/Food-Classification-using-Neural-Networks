from kivy.app import App
from kivy.uix.widget import Widget
from kivy.core.window import Window
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture
from kivy.config import Config
import cv2
from Helpers import crop_image
import tensorflow as tf
import os
import random
import numpy as np


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

food_list = ['Apple Pie', 'Baby Back Ribs', 'Baklava', 'Beef Carpaccio', 'Beef Tartare', 'Beet Salad', 'Beignets', 'Bibimbap',
             'Bread Pudding', 'Breakfast Burrito', 'Bruschetta', 'Caesar Salad', 'Cannoli', 'Caprese Salad', 'Carrot Cake',
             'Ceviche', 'Cheesecake', 'Cheese Plate', 'Chicken Curry', 'Chicken Quesadilla', 'Chicken Wings', 'Chocolate Cake',
             'Chocolate Mousse', 'Churros', 'Clam Chowder', 'Club Sandwich', 'Crab Cakes', 'Creme Brulee', 'Croque Madame',
             'Cup Cakes', 'Deviled Eggs', 'Donuts', 'Dumplings', 'Edamame', 'Eggs Benedict', 'Escargots', 'Falafel',
             'Filet Mignon', 'Fish And Chips', 'Foie Gras', 'French Fries', 'French Onion Soup', 'French Toast',
             'Fried Calamari', 'Fried Rice', 'Frozen Yogurt', 'Garlic Bread', 'Gnocchi', 'Greek Salad',
             'Grilled Cheese Sandwich', 'Grilled Salmon', 'Guacamole', 'Gyoza', 'Hamburger', 'Hot And Sour Soup', 'Hot Dog',
             'Huevos Rancheros', 'Hummus', 'Ice Cream', 'Lasagna', 'Lobster Bisque', 'Lobster Roll Sandwich',
             'Macaroni And Cheese', 'Macarons', 'Miso Soup', 'Mussels', 'Nachos', 'Omelette', 'Onion Rings', 'Oysters',
             'Pad Thai', 'Paella', 'Pancakes', 'Panna Cotta', 'Peking Duck', 'Pho', 'Pizza', 'Pork Chop', 'Poutine', 'Prime Rib',
             'Pulled Pork Sandwich', 'Ramen', 'Ravioli', 'Red Velvet Cake', 'Risotto', 'Samosa', 'Sashimi', 'Scallops',
             'Seaweed Salad', 'Shrimp And Grits', 'Spaghetti Bolognese', 'Spaghetti Carbonara', 'Spring Rolls', 'Steak',
             'Strawberry Shortcake', 'Sushi', 'Tacos', 'Takoyaki', 'Tiramisu', 'Tuna Tartare', 'Waffles']

Window.size = (800, 520)
Window.clearcolor = (27/255, 36/255, 52/255, 1)
Config.set('graphics', 'resizable', '0')

drop_image = False
uploaded_image = None
show = 0

path = None
upload_file_path = None
images_dir = 'Final/'
dir_list = os.listdir(images_dir)
food_class_actual = None
TF_flag = True

img = None
model = tf.keras.models.load_model('Model/temp_model.h5')


class Boxes(Widget):

    def __init__(self, **kwargs):
        super(Boxes, self).__init__(**kwargs)
        Window.bind(on_dropfile=self.on_file_drop)

        self.wid = Widget()
        self.second = BoxLayout()
        self.img1 = Image()

        self.add_widget(self.second)
        self.add_widget(self.wid)
        self.add_widget(self.img1, index=50)


    def display_image(self, frame_to_conv):
        buf1 = cv2.flip(frame_to_conv, 0)
        buf = buf1.tobytes()
        texture = Texture.create(size=(frame_to_conv.shape[1], frame_to_conv.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

        pos_hint = (5, 4)
        size_hint = (512, 512)
        allow_stretch = False
        keep_ratio = False

        self.img1.texture = texture
        self.img1.pos = pos_hint[0], pos_hint[1]
        self.img1.size = (size_hint[0], size_hint[1])
        self.img1.allow_stretch = allow_stretch
        self.img1.keep_ratio = keep_ratio


    def generate_random(self, value):
        global path, dir_list, food_class_actual, img, TF_flag

        rand_food = random.choice(dir_list)

        food_class_actual = rand_food
        food_class_actual = food_class_actual.replace('_', ' ')
        food_class_actual = food_class_actual.title()

        images_list = os.listdir(images_dir + rand_food + '/')
        rand_image = random.choice(images_list)
        path = images_dir + rand_food + '/' + rand_image

        img = cv2.imread(path)
        img = crop_image(img, res=(512, 512))

        self.display_image(img)

        TF_flag = True


    def upload_image(self, value):
        global drop_image, uploaded_image, path, upload_file_path, img, TF_flag

        if drop_image == False:
            print('Drag and drop image below')
            self.ids.drag_drop.text = 'Drag and Drop Image Here'
        elif drop_image == True:
            path = upload_file_path
            img = cv2.imread(path)
            img = crop_image(img, res=(512, 512))
            self.display_image(img)
            TF_flag = False



    def on_file_drop(self, window, file_path):
        global drop_image, uploaded_image, upload_file_path

        self.ids.drag_drop.text = ''

        self.filePath = file_path.decode("utf-8")
        self.ids.img.source = self.filePath
        self.ids.img.reload()
        upload_file_path = self.filePath
        drop_image = True


    def predict(self, value):
        global img, TF_flag

        img_predict = tf.convert_to_tensor(img, dtype=tf.float32)
        img_predict = tf.image.resize(img_predict, [224, 224])
        img_predict = tf.expand_dims(img_predict, axis=0)

        pred_probs = model.predict(img_predict)

        pred_probs = np.squeeze(pred_probs)
        indexes = pred_probs.argsort()[-3:][::-1]
        probablities = [x for x in pred_probs[indexes]]

        prediction1 = food_list[indexes[0]]
        prediction2 = food_list[indexes[1]]
        prediction3 = food_list[indexes[2]]

        self.ids.pred1.text = prediction1
        self.ids.pred2.text = prediction2
        self.ids.pred3.text = prediction3

        self.ids.Prob1.text = str(round((probablities[0] * 100), 2))
        self.ids.Prob2.text = str(round((probablities[1] * 100), 2))
        self.ids.Prob3.text = str(round((probablities[2] * 100), 2))

        if TF_flag == True:
            if prediction1 != food_class_actual:
                self.ids.TF1.text = 'False'
                self.ids.TF1.color = 1, 0, 0, 1
            if prediction1 == food_class_actual:
                self.ids.TF1.text = 'True'
                self.ids.TF1.color = 0, 1, 0, 1

            print(prediction1)
            print(food_class_actual)

            if prediction2 != food_class_actual:
                self.ids.TF2.text = 'False'
                self.ids.TF2.color = 1, 0, 0, 1
            if prediction2 == food_class_actual:
                self.ids.TF2.text = 'True'
                self.ids.TF2.color = 0, 1, 0, 1

            if prediction3 != food_class_actual:
                self.ids.TF3.text = 'False'
                self.ids.TF3.color = 1, 0, 0, 1
            if prediction3 == food_class_actual:
                self.ids.TF3.text = 'True'
                self.ids.TF3.color = 0, 1, 0, 1

        if TF_flag == False:
            self.ids.TF1.text = '-'
            self.ids.TF1.color = 1, 1, 1, 1
            self.ids.TF2.text = '-'
            self.ids.TF2.color = 1, 1, 1, 1
            self.ids.TF3.text = '-'
            self.ids.TF3.color = 1, 1, 1, 1

    pass


class FoodClassificationApp(App):

    def build(self):
        self.icon = "Favicon.png"
        layout = Boxes()
        layout.generate_random(None)
        return layout
    pass





if __name__ == "__main__":
    FoodClassificationApp().run()