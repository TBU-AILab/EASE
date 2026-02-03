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

        planet_wars_nothing = """```
import random
import math

def get_action(state: dict) -> dict or None:

    # Get planets from the state
    planets = get_planets(state)

    # Filter our planets and enemy planets
    our_planets = [p for p in planets if p.get('owner') == 1]  # Assuming player ID is 1
    enemy_planets = [p for p in planets if p.get('owner') != 1]

    # If we don't have any planets, do nothing
    if not our_planets:
        return None

    # Find the best target planet to attack (heuristic: choose the enemy planet with the most ships)
    best_target = min(enemy_planets, key=lambda p: p.get('ships', 0), default=None)

    # If there's no good target, do nothing
    if not best_target:
        return None

    # Calculate the number of ships to send based on our available resources (heuristic: 1/4 of total ships)
    source = min(our_planets, key=lambda p: p.get('ships', 0), default=None)
    if source and source['ships'] >= 4:
        ships_to_send = 4
    else:
        return None

    # Return the action
    return {
        'source': source['id'],
        'target': best_target['id'],
        'ships': ships_to_send,
    }


def get_planets(state: dict) -> list or None:
    return state.get('planets', [])
```"""

        transitions_rnd = """```
import os
import random

def extract_features(filepath):
    return random.choices([0, 1], k = 10)
    
def train(X_train, y_train, extract_features):
    model = {"name": "mock model", "id": 1234}
    return model

def predict(model, extract_features, x):

    def file_exists(path: str) -> bool:
        return os.path.isfile(path)

    if file_exists(x):
        print(f"File {x} is OK")
    else:
        print(f"File {x} is NOT OK")

    return {"prediction": random.choice([0, 1]), "probability": random.random()}
```"""

        transitions_def = """```
import random
import ffmpeg
import cv2
import xgboost
import numpy as np
import sklearn
import librosa
import datetime
        
def extract_features(filepath):
    def _safe_int(x, default=0):
        try:
            return int(x)
        except Exception:
            return default

    def _safe_float(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return default

    def _probe(path):
        try:
            return ffmpeg.probe(path)
        except Exception:
            return None

    def _get_stream_info(meta, stype):
        if not meta or 'streams' not in meta:
            return None
        for s in meta['streams']:
            if s.get('codec_type') == stype:
                return s
        return None

    def _run_ffmpeg_video(path, w, h, fps, max_frames):
        n = max(1, _safe_int(max_frames, 1))
        try:
            out, _ = (
                ffmpeg
                .input(path, threads=1)
                .filter('fps', fps=fps, round='up')
                .filter('scale', w, h)
                .output('pipe:', format='rawvideo', pix_fmt='rgb24', vframes=n, loglevel='error')
                .run(capture_stdout=True, capture_stderr=True)
            )
            arr = np.frombuffer(out, np.uint8)
            expected = n * w * h * 3
            if arr.size < expected:
                if arr.size == 0:
                    return None
                n2 = arr.size // (w * h * 3)
                if n2 <= 0:
                    return None
                arr = arr[:n2 * w * h * 3]
                n = n2
            else:
                arr = arr[:expected]
            frames = arr.reshape((n, h, w, 3)).astype(np.float32) / 255.0
            return frames
        except Exception:
            return None

    def _run_ffmpeg_audio(path, sr, max_sec):
        max_sec = _safe_float(max_sec, 10.0)
        try:
            out, _ = (
                ffmpeg
                .input(path, threads=1)
                .output('pipe:', format='f32le', acodec='pcm_f32le', ac=1, ar=sr, t=max_sec, loglevel='error')
                .run(capture_stdout=True, capture_stderr=True)
            )
            y = np.frombuffer(out, dtype=np.float32)
            if y.size == 0:
                return None
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            m = int(sr * max_sec)
            if y.size > m:
                y = y[:m]
            return y
        except Exception:
            return None

    def _video_features(frames, bins=24):
        if frames is None or not isinstance(frames, np.ndarray) or frames.ndim != 4 or frames.shape[-1] != 3:
            return None
        n = frames.shape[0]
        if n < 2:
            return None

        gray = 0.2989 * frames[..., 0] + 0.5870 * frames[..., 1] + 0.1140 * frames[..., 2]
        gray = np.clip(gray, 0.0, 1.0)

        mean_luma = float(np.mean(gray))
        std_luma = float(np.std(gray))

        diffs = np.abs(gray[1:] - gray[:-1])
        per_frame_diff = np.mean(np.mean(diffs, axis=2), axis=1).astype(np.float32)
        diff_mean = float(np.mean(diffs))
        diff_std = float(np.std(diffs))
        qd = np.quantile(per_frame_diff, [0.01, 0.05, 0.10, 0.20, 0.25, 0.5, 0.75, 0.80, 0.90, 0.95, 0.99]).astype(np.float32)

        d = np.clip(per_frame_diff, 0.0, 1.0)
        hist, _ = np.histogram(d, bins=bins, range=(0.0, 1.0), density=True)
        hist = hist.astype(np.float32)

        hard = np.array([
            float(np.mean(d >= 0.10)),
            float(np.mean(d >= 0.14)),
            float(np.mean(d >= 0.18)),
            float(np.mean(d >= 0.22)),
            float(np.mean(d >= 0.26)),
            float(np.mean(d >= 0.30)),
            float(np.mean(d >= 0.34)),
            float(np.mean(d >= 0.38)),
            float(np.mean(d >= 0.42)),
            float(np.max(d) if d.size else 0.0),
            float(np.std(d)),
            float(np.mean(np.abs(d[1:] - d[:-1]))) if d.size > 1 else 0.0
        ], dtype=np.float32)

        f = frames.reshape((n, -1, 3))
        mean_rgb = np.mean(f, axis=1).astype(np.float32)
        std_rgb = np.std(f, axis=1).astype(np.float32)
        rgb_mean = np.mean(mean_rgb, axis=0).astype(np.float32)
        rgb_std = np.mean(std_rgb, axis=0).astype(np.float32)
        rgb_diff_vec = np.abs(mean_rgb[1:] - mean_rgb[:-1]).astype(np.float32)
        rgb_diff = np.mean(rgb_diff_vec, axis=0).astype(np.float32)
        rgb_diff_scalar = np.mean(rgb_diff_vec, axis=1).astype(np.float32) if rgb_diff_vec.shape[0] > 0 else np.zeros((1,), dtype=np.float32)
        rgb_diff_q = np.quantile(rgb_diff_scalar, [0.05, 0.25, 0.5, 0.75, 0.95]).astype(np.float32)

        edges = []
        for i in range(n):
            g8 = np.clip(gray[i] * 255.0, 0, 255).astype(np.uint8)
            e = cv2.Canny(g8, 22, 85)
            edges.append(np.mean(e > 0))
        edges = np.array(edges, dtype=np.float32)
        edge_mean = float(np.mean(edges))
        edge_std = float(np.std(edges))
        edge_diff = np.abs(edges[1:] - edges[:-1]) if edges.size > 1 else np.zeros((1,), dtype=np.float32)
        edge_diff_m = float(np.mean(edge_diff))
        edge_diff_s = float(np.std(edge_diff))
        edge_q = np.quantile(edges, [0.05, 0.25, 0.5, 0.75, 0.95]).astype(np.float32)
        edge_diff_q = np.quantile(edge_diff, [0.10, 0.5, 0.90, 0.97]).astype(np.float32)

        feats = np.concatenate([
            np.array([mean_luma, std_luma, diff_mean, diff_std, edge_mean, edge_std, edge_diff_m, edge_diff_s], dtype=np.float32),
            qd,
            hard,
            edge_q,
            edge_diff_q,
            rgb_mean,
            rgb_std,
            rgb_diff,
            rgb_diff_q,
            hist
        ], axis=0)
        return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _audio_features(y, sr):
        if y is None or not isinstance(y, np.ndarray) or y.size < int(0.25 * sr):
            return None
        y = y.astype(np.float32)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        y = y - np.mean(y)
        eps = 1e-12
        rms = float(np.sqrt(np.mean(y * y) + eps))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=512)))

        try:
            sc = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=1024, hop_length=512)
            sb = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=1024, hop_length=512)
            srll = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=1024, hop_length=512, roll_percent=0.85)
            flat = librosa.feature.spectral_flatness(y=y, n_fft=1024, hop_length=512)
            sc_m, sc_s = float(np.mean(sc)), float(np.std(sc))
            sb_m, sb_s = float(np.mean(sb)), float(np.std(sb))
            sr_m, sr_s = float(np.mean(srll)), float(np.std(srll))
            fl_m, fl_s = float(np.mean(flat)), float(np.std(flat))
        except Exception:
            sc_m = sc_s = sb_m = sb_s = sr_m = sr_s = fl_m = fl_s = 0.0

        try:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=26, n_fft=1024, hop_length=512)
            mfcc_m = np.mean(mfcc, axis=1).astype(np.float32)
            mfcc_s = np.std(mfcc, axis=1).astype(np.float32)
            dm = np.mean(np.abs(mfcc[:, 1:] - mfcc[:, :-1]), axis=1).astype(np.float32) if mfcc.shape[1] > 1 else np.zeros((26,), dtype=np.float32)
        except Exception:
            mfcc_m = np.zeros((26,), dtype=np.float32)
            mfcc_s = np.zeros((26,), dtype=np.float32)
            dm = np.zeros((26,), dtype=np.float32)

        try:
            hop = 512
            frame = 1024
            if y.size < frame:
                yy = np.pad(y, (0, frame - y.size), mode='constant')
                n_frames = 1
            else:
                yy = y
                n_frames = max(1, 1 + (y.size - frame) // hop)
            energies = []
            for i in range(n_frames):
                a = i * hop
                b = a + frame
                if b > yy.size:
                    seg = yy[a:]
                    if seg.size < frame:
                        seg = np.pad(seg, (0, frame - seg.size), mode='constant')
                else:
                    seg = yy[a:b]
                energies.append(np.mean(seg * seg))
            energies = np.array(energies, dtype=np.float32)
            energies = np.nan_to_num(energies, nan=0.0, posinf=0.0, neginf=0.0)
            e_m = float(np.mean(energies))
            e_s = float(np.std(energies))
            if energies.size >= 2:
                e_diff = np.abs(energies[1:] - energies[:-1])
                e_diff_m = float(np.mean(e_diff))
                e_diff_s = float(np.std(e_diff))
                e_q = np.quantile(e_diff, [0.01, 0.05, 0.10, 0.20, 0.25, 0.5, 0.75, 0.80, 0.90, 0.95, 0.99]).astype(np.float32)
                e_rate = np.array([
                    float(np.mean(e_diff >= 0.0008)),
                    float(np.mean(e_diff >= 0.0015)),
                    float(np.mean(e_diff >= 0.0025)),
                    float(np.mean(e_diff >= 0.0035)),
                    float(np.mean(e_diff >= 0.0050)),
                    float(np.max(e_diff))
                ], dtype=np.float32)
            else:
                e_diff_m = 0.0
                e_diff_s = 0.0
                e_q = np.zeros((11,), dtype=np.float32)
                e_rate = np.zeros((6,), dtype=np.float32)
        except Exception:
            e_m = e_s = e_diff_m = e_diff_s = 0.0
            e_q = np.zeros((11,), dtype=np.float32)
            e_rate = np.zeros((6,), dtype=np.float32)

        feats = np.concatenate([
            np.array([rms, zcr, sc_m, sc_s, sb_m, sb_s, sr_m, sr_s, fl_m, fl_s, e_m, e_s, e_diff_m, e_diff_s], dtype=np.float32),
            e_q.astype(np.float32),
            e_rate.astype(np.float32),
            mfcc_m,
            mfcc_s,
            dm
        ], axis=0)
        return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # --- original extraction settings ---
    video_w, video_h, video_fps, video_frames = 128, 72, 6, 60
    audio_sr, audio_sec = 16000, 10.0

    meta = _probe(filepath)
    vstream = _get_stream_info(meta, 'video')
    astream = _get_stream_info(meta, 'audio')

    has_v = 1.0 if vstream is not None else 0.0
    has_a = 1.0 if astream is not None else 0.0

    v_feats = None
    a_feats = None

    if has_v > 0:
        frames = _run_ffmpeg_video(filepath, video_w, video_h, video_fps, video_frames)
        v_feats = _video_features(frames)

    if has_a > 0:
        y = _run_ffmpeg_audio(filepath, audio_sr, audio_sec)
        a_feats = _audio_features(y, audio_sr)

    if v_feats is None:
        v_feats = np.zeros((8 + 11 + 12 + 5 + 4 + 3 + 3 + 3 + 5 + 24,), dtype=np.float32)
    if a_feats is None:
        a_feats = np.zeros((14 + 11 + 6 + 26 + 26 + 26), dtype=np.float32)

    if meta and 'format' in meta and isinstance(meta['format'], dict):
        dur = meta['format'].get('duration', None)
        br = meta['format'].get('bit_rate', None)
    else:
        dur = None
        br = None
    dur = _safe_float(dur, 0.0)
    br = _safe_float(br, 0.0)

    fpsv = 0.0
    wv = 0.0
    hv = 0.0
    try:
        if vstream is not None:
            r = vstream.get('r_frame_rate', '0/1')
            if isinstance(r, str) and '/' in r:
                a, b = r.split('/', 1)
                fpsv = _safe_float(a, 0.0) / max(1e-6, _safe_float(b, 1.0))
            wv = _safe_float(vstream.get('width', 0.0), 0.0)
            hv = _safe_float(vstream.get('height', 0.0), 0.0)
    except Exception:
        fpsv = 0.0
        wv = 0.0
        hv = 0.0

    vcodec = 0.0
    acodec = 0.0
    try:
        if vstream is not None:
            vc = vstream.get('codec_name', '')
            vcodec = float((hash(str(vc)) % 997) / 997.0)
    except Exception:
        vcodec = 0.0
    try:
        if astream is not None:
            ac = astream.get('codec_name', '')
            acodec = float((hash(str(ac)) % 997) / 997.0)
    except Exception:
        acodec = 0.0

    extra = np.array([has_v, has_a, dur, br, fpsv, wv, hv, vcodec, acodec], dtype=np.float32)
    feats = np.concatenate([v_feats, a_feats, extra], axis=0)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return feats
    
def train(X_train, y_train, extract_features):
    random.seed(0)
    np.random.seed(0)

    def _probas(mdl_list, X):
        ps = []
        for mdl in mdl_list:
            try:
                ps.append(mdl.predict_proba(X)[:, 1].astype(np.float32))
            except Exception:
                try:
                    ps.append(np.clip(mdl.predict(X).astype(np.float32), 0.0, 1.0))
                except Exception:
                    ps.append(np.full((X.shape[0],), 0.5, dtype=np.float32))
        return np.vstack(ps)

    def _calibrate_isotonic(scores, labels):
        try:
            ir = sklearn.isotonic.IsotonicRegression(out_of_bounds='clip')
            ir.fit(scores.astype(np.float64), labels.astype(np.float64))
            return ir
        except Exception:
            return None

    Xtr = np.vstack([extract_features(p) for p in X_train]).astype(np.float32)

    y = np.array(y_train, dtype=np.int32)
    if y.size != Xtr.shape[0]:
        m = min(y.size, Xtr.shape[0])
        Xtr = Xtr[:m]
        y = y[:m]

    model = {
        "mode": "xgb_ensemble",
        "models": None,
        "scaler": None,
        "thr": 0.5,
        "iso": None,
        "weights": None,
        "pos": int(np.sum(y == 1)),
        "neg": int(np.sum(y == 0)),
    }

    uniq = np.unique(y)
    if uniq.size == 1:
        only = int(uniq[0])
        model["mode"] = "constant"
        model["constant_label"] = only
        return model

    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    spw = float(neg) / float(max(1, pos))
    model["pos"] = pos
    model["neg"] = neg

    # --- stratified split ---
    try:
        split = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.22, random_state=0)
        tr_idx, va_idx = next(split.split(Xtr, y))
        Xtr2, ytr2 = Xtr[tr_idx], y[tr_idx]
        Xva, yva = Xtr[va_idx], y[va_idx]
    except Exception:
        Xtr2, ytr2 = Xtr, y
        Xva, yva = None, None

    # --- scaling ---
    try:
        scaler = sklearn.preprocessing.StandardScaler(with_mean=True, with_std=True)
        Xtr2s = scaler.fit_transform(Xtr2.astype(np.float64)).astype(np.float32)
        Xvas = scaler.transform(Xva.astype(np.float64)).astype(np.float32) if Xva is not None else None
    except Exception:
        scaler = None
        Xtr2s = Xtr2
        Xvas = Xva

    def _fit_model(seed, depth, lr, subs, col, mincw, reglam, alpha, gamma, mdw):
        m = xgboost.XGBClassifier(
            n_estimators=950,
            max_depth=int(depth),
            learning_rate=float(lr),
            subsample=float(subs),
            colsample_bytree=float(col),
            min_child_weight=float(mincw),
            reg_lambda=float(reglam),
            reg_alpha=float(alpha),
            gamma=float(gamma),
            max_delta_step=float(mdw),
            objective='binary:logistic',
            eval_metric='logloss',
            tree_method='hist',
            random_state=int(seed),
            n_jobs=1,
            scale_pos_weight=spw
        )
        m.fit(Xtr2s, ytr2)
        return m

    candidates = [
        (0, 5, 0.040, 0.88, 0.88, 1.0, 1.3, 0.0, 0.0, 1.0),
        (1, 4, 0.060, 0.92, 0.92, 1.0, 1.0, 0.0, 0.0, 1.0),
        (2, 6, 0.035, 0.82, 0.86, 1.5, 1.4, 0.0, 0.0, 1.0),
        (3, 5, 0.045, 0.78, 0.82, 2.0, 1.1, 0.0, 0.0, 1.0),
        (4, 4, 0.070, 0.86, 0.80, 1.0, 1.0, 0.0, 0.1, 1.0),
        (5, 3, 0.085, 0.94, 0.94, 1.0, 1.0, 0.0, 0.0, 1.0),
        (6, 5, 0.050, 0.90, 0.78, 1.0, 1.0, 0.7, 0.0, 1.0),
        (7, 6, 0.040, 0.88, 0.80, 1.0, 1.6, 0.0, 0.0, 1.0)
    ]

    models = []
    for c in candidates:
        try:
            models.append(_fit_model(*c))
        except Exception:
            continue

    if len(models) == 0:
        fallback = xgboost.XGBClassifier(
            n_estimators=450,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective='binary:logistic',
            eval_metric='logloss',
            tree_method='hist',
            random_state=0,
            n_jobs=1,
            scale_pos_weight=spw
        )
        fallback.fit(Xtr2s, ytr2)
        models = [fallback]

    thr = 0.5
    iso = None
    weights = None

    # --- validation calibration + weight/threshold search ---
    if Xvas is not None and yva is not None and yva.size > 0:
        try:
            Pva = _probas(models, Xvas)
            pva_raw = np.mean(Pva, axis=0).astype(np.float32)

            iso = _calibrate_isotonic(pva_raw, yva)

            best = (-1.0, 0.5, None)
            ts = np.linspace(0.02, 0.98, 171)
            base_w = np.ones((Pva.shape[0],), dtype=np.float32)

            grids = [
                base_w,
                np.array([1.2] + [1.0] * (Pva.shape[0] - 1), dtype=np.float32),
                np.array([0.8] + [1.0] * (Pva.shape[0] - 1), dtype=np.float32)
            ]

            for w in grids:
                w = w[:Pva.shape[0]]
                w = w / max(1e-6, float(np.sum(w)))
                pv = np.sum(Pva * w.reshape((-1, 1)), axis=0).astype(np.float32)

                if iso is not None:
                    try:
                        pv = iso.transform(pv.astype(np.float64)).astype(np.float32)
                        pv = np.nan_to_num(pv, nan=0.5, posinf=1.0, neginf=0.0).astype(np.float32)
                    except Exception:
                        pv = np.nan_to_num(pv, nan=0.5, posinf=1.0, neginf=0.0).astype(np.float32)

                for t in ts:
                    predv = (pv >= t).astype(np.int32)
                    tp = int(np.sum((predv == 1) & (yva == 1)))
                    fp = int(np.sum((predv == 1) & (yva == 0)))
                    fn = int(np.sum((predv == 0) & (yva == 1)))
                    prec = tp / float(max(1, tp + fp))
                    rec = tp / float(max(1, tp + fn))
                    f1 = 0.0 if (prec + rec) == 0 else (2.0 * prec * rec / (prec + rec))
                    if f1 > best[0]:
                        best = (f1, float(t), w.copy())

            thr = best[1]
            weights = best[2]
        except Exception:
            thr = 0.5
            iso = None
            weights = None

    model["models"] = models
    model["scaler"] = scaler
    model["thr"] = float(thr)
    model["iso"] = iso
    if weights is not None:
        model["weights"] = weights.astype(np.float32)
    else:
        model["weights"] = None

    return model
    
def predict(model, extract_features, filepath):
    paths = [filepath]
    if model is None:
        return {"prediction": 0, "probability": -1}
    if len(paths) == 0:
        return []

    mode = model.get("mode", "xgb_ensemble")

    # Constant-mode batch rule preserved, but not tied to any stored X_test:
    # if only label is 0 => flip the first item in the requested batch to 1 (if any).
    if mode == "constant":
        only = int(model.get("constant_label", 0))
        if only == 0:
            out = [0 for _ in range(len(paths))]
            out[0] = 1
            return {"prediction": 0, "probability": -1}
        return {"prediction": only, "probability": -1}

    models = model.get("models", None)
    if not models:
        return {"prediction": 0, "probability": -1}

    # Feature extraction for the whole batch
    X = np.vstack([extract_features(p) for p in paths]).astype(np.float32)

    scaler = model.get("scaler", None)
    if scaler is not None:
        try:
            X = scaler.transform(X.astype(np.float64)).astype(np.float32)
        except Exception:
            pass

    # Predict probabilities for each model (n_models, n_samples)
    ps = []
    for mdl in models:
        try:
            ps.append(mdl.predict_proba(X)[:, 1].astype(np.float32))
        except Exception:
            try:
                ps.append(np.clip(mdl.predict(X).astype(np.float32), 0.0, 1.0))
            except Exception:
                ps.append(np.full((X.shape[0],), 0.5, dtype=np.float32))
    P = np.vstack(ps)

    weights = model.get("weights", None)
    if weights is None:
        w = np.ones((P.shape[0],), dtype=np.float32) / float(P.shape[0])
    else:
        w = np.asarray(weights, dtype=np.float32).ravel()
        w = w[:P.shape[0]]
        w = w / max(1e-6, float(np.sum(w)))

    p = np.sum(P * w.reshape((-1, 1)), axis=0).astype(np.float32)
    p = np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0).astype(np.float32)
    p = np.clip(p, 0.0, 1.0).astype(np.float32)

    iso = model.get("iso", None)
    if iso is not None:
        try:
            p2 = iso.transform(p.astype(np.float64)).astype(np.float32)
            p2 = np.nan_to_num(p2, nan=0.5, posinf=1.0, neginf=0.0).astype(np.float32)
            p = np.clip(p2, 0.0, 1.0).astype(np.float32)
        except Exception:
            pass

    thr = float(model.get("thr", 0.5))
    pred = (p >= thr).astype(np.int32)

    # Batch post-processing preserved (but now applied to the *input batch*):
    # If all batch outputs are 0 and the batch size is at least 50, force some (~2%, min 1) positive predictions
    pos = int(model.get("pos", 0))
    if int(np.sum(pred)) == 0 and pos > 0 and len(pred) >= 50:
        k = max(1, int(0.02 * len(pred)))
        idx = np.argsort(-p)[:k]
        pred[idx] = 1

    return {"prediction": [int(v) for v in pred.tolist()][0], "probability": -1}
```"""

        match self._response_type:
            case 'Meta: random search':
                msg_text = meta_random_search
            case '2048: left and slow':
                msg_text = slow_2048
            case 'ModernTV: video transitions':
                msg_text = transitions_rnd
            case 'PlanetWars: Do nothing agent':
                msg_text = planet_wars_nothing
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
