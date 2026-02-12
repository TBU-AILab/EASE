from ..message import Message
from .llmconnector import LLMConnector
from ..loader import Parameter, PrimitiveType
from ..utils.connector_utils import get_available_models


class LLMConnectorMock(LLMConnector):

    @classmethod
    def get_parameters(cls) -> dict[str, Parameter]:

        av_models = get_available_models(cls.get_short_name())

        return {
            'response': Parameter(short_name="response", type=PrimitiveType.enum, long_name='Response type',
                                  enum_options=av_models['model_names'], enum_descriptions=av_models['model_longnames'], default='Meta: random search')
        }

    def _init_params(self):
        super()._init_params()
        self._response_type = self.parameters.get('response', self.get_parameters().get('response').default)

        self._type = 'Mock'  # Type of LLM (OpenAI, Meta, Google, ...)
        self._model = 'gpt-3.5-turbo'  # Model ('gpt-3.5-turbo', ...)

    ####################################################################
    #########  Public functions
    ####################################################################
    def get_role_user(self) -> str:
        """
        Get role specification string for USER.
        Returns string.
        """
        return 'user'

    def get_role_system(self) -> str:
        """
        Get role specification string for SYSTEM.
        Returns string.
        """
        return 'system'

    def get_role_assistant(self) -> str:
        """
        Get role specification string for ASSISTANT.
        Returns string.
        """
        return 'assistant'

    def send(self, context) -> Message:
        """
        Send context to mock LLM.
        Returns mock response as Message from LLM.
        """

        meta_random_search = """```import random
import math
import numpy as np


def run(func, dim, bounds, max_evals):
    best_solution = None
    best_score = float('inf')

    for i in range(max_evals):
        candidate_solution = [np.random.uniform(low, high) for low, high in bounds]
        candidate_score = func(candidate_solution)

        if candidate_score < best_score:
            best_score = candidate_score
            best_solution = candidate_solution


    return best_score
```"""

        """ lmsg_text = ```
import cv2
import dlib
import os
from PIL import Image
import numpy as np

def select(folder_path, show_genre):
    model_file = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(model_file):
        # Download shape predictor model if it doesn't exist
        import requests
        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        response = requests.get(url)
        with open("shape_predictor_68_face_landmarks.dat.bz2", "wb") as file:
            file.write(response.content)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_file)

    selected_images = []
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        if os.path.isfile(image_path) and image_path.endswith(('.png', '.jpg', '.jpeg')):
            try:
                img = cv2.imread(image_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Face detection
                faces = detector(gray, 1)
                rating = 0
                for face in faces:
                    landmarks = predictor(gray, face)
                    rating += len(landmarks.parts()) / 68

                if show_genre == "nature documentary":
                    # Use a metric based on image complexity (e.g., number of edges)
                    img_edges = cv2.Canny(img, 100, 200)
                    rating += np.count_nonzero(img_edges) / (img.shape[0] * img.shape[1])
                elif show_genre == "soap opera":
                    # Use a metric based on face expression (e.g., number of facial landmarks)
                    rating += len(landmarks.parts()) / 68
                selected_images.append({'path': image_path, 'rating': rating})
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

    # Return the top 5 images with the highest ratings
    return sorted(selected_images, key=lambda x: x['rating'], reverse=True)[:5]
```"""

        """  msg_text = ```
import os
import cv2
import numpy as np
import dlib
import face_recognition
from skimage import color, exposure

def select(folder_path, show_genre):
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    rated_images = []

    detector = dlib.get_frontal_face_detector()

    for image_path in image_files:
        image = cv2.imread(image_path)
        if image is None:
            continue

        if show_genre == 'nature documentary':
            score = assess_nature(image)
        elif show_genre == 'soap opera':
            score = assess_soap_opera(image, detector)
        elif show_genre == 'action-adventure show':
            score = assess_action(image)
        elif show_genre == 'comedy':
            score = assess_comedy(image, detector)
        elif show_genre == 'historical drama':
            score = assess_historical(image, detector)
        elif show_genre == 'science fiction':
            score = assess_science_fiction(image, detector)
        elif show_genre == 'horror':
            score = assess_horror(image, detector)
        elif show_genre == 'animated':
            score = assess_animated(image)
        elif show_genre == 'talk show':
            score = assess_talk_show(image, detector)
        elif show_genre == 'news':
            score = assess_news(image, detector)
        else:
            continue

        rated_images.append({'path': image_path, 'rating': score})

    # Sort rated images by score and select top 5
    rated_images.sort(key=lambda x: x['rating'], reverse=True)
    selected_images = rated_images[:5]

    return selected_images

def assess_nature(image):
    # For nature images: clarity and beauty
    brightness = np.mean(image)
    return min(1, brightness / 255.0)  # Normalize brightness to [0, 1]

def assess_soap_opera(image, detector):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_image)
    score = len(faces) * 0.2  # Each detected face adds to score
    return min(1.0, score)

def assess_action(image):
    # Determine motion via optical flow, color intensity
    # Placeholder: using color variance
    color_variance = np.std(image)
    return min(1, color_variance / 255.0)

def assess_comedy(image, detector):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_image)
    score = len(faces) * 0.2  # Each detected face adds to score
    return min(1.0, score)

def assess_historical(image, detector):
    # Assume historical images contain specific colors or features
    color_variance = np.std(image)
    faces = detector(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    score = (color_variance / 255.0 + len(faces) * 0.2) / 2
    return min(1.0, score)

def assess_science_fiction(image, detector):
    # Assess both futuristic elements and faces
    color_saturation = np.mean(color.rgb2hsv(image)[:, :, 1])
    faces = detector(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    score = (color_saturation + len(faces) * 0.2) / 2
    return min(1.0, score)

def assess_horror(image, detector):
    # Focus on dark themes and emotional expressions
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray_image)
    faces = detector(gray_image)
    score = (1 - (brightness / 255.0) + len(faces) * 0.2) / 2
    return min(1.0, score)

def assess_animated(image):
    # Rich colors and key visual elements
    saturation = np.mean(color.rgb2hsv(image)[:, :, 1])
    return min(1.0, saturation)

def assess_talk_show(image, detector):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_image)
    score = len(faces) * 0.3  # More faces mean higher scores
    return min(1.0, score)

def assess_news(image, detector):
    # Focus on clarity and presence of faces
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clarity = np.std(gray_image)
    faces = detector(gray_image)
    score = (clarity / 255.0 + len(faces) * 0.3) / 2
    return min(1.0, score)
```"""

        slow_2048 = """```
import numpy as np
import time

def move(grid: np.array, score: int) -> str:

    time.sleep(10)

    return 'left'
```"""

        transitions = """```
import random

def predict(X_train, y_train, X_test):

    return random.choices([0, 1], k = len(X_test))
```"""

        paper_context = """```
import numpy as np
from datetime import datetime, timedelta
        
def run(func, dim, bounds, max_time):
    start = datetime.now()
    best = float('inf')

    # Algorithm body
    while True:
        passed_time = (datetime.now() - start)
        if passed_time >= timedelta(seconds=max_time):
            return best

        params = [np.random.uniform(low, high) for low, high in bounds]
        fitness = func(params)
        if best is None or fitness <= best:
            best = fitness
```"""

        match self._response_type:
            case 'Meta: random search':
                msg_text = meta_random_search
            case '2048: left and slow':
                msg_text = slow_2048
            case 'ModernTV: video transitions':
                msg_text = transitions
            case 'Paper: context':
                msg_text = paper_context
            case _:
                msg_text = 'This is simple response.'

        msg = Message(
            role=self.get_role_assistant(),
            model_encoding=None,
            message=msg_text
        )
        msg.set_tokens(10)  # funny number
        return msg

    def get_model(self) -> str:
        """
        Returns current LLM model as string.
        """
        return self._model

    @classmethod
    def get_short_name(cls) -> str:
        return "llm.mock"

    @classmethod
    def get_long_name(cls) -> str:
        return "Mock LLM connector"

    @classmethod
    def get_description(cls) -> str:
        return "Connector only for testing and demonstrations."

    @classmethod
    def get_tags(cls) -> dict:
        return {
            'input': set(),
            'output': {'text'}
        }
    ####################################################################
    #########  Private functions
    ####################################################################
