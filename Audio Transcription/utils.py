from urllib.parse import urlparse
from transformers import Wav2Vec2Config
from transformers import pipeline,AutoModelForCTC, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM
import socket
from urllib3.connection import HTTPConnection
import os 
import requests

HTTPConnection.default_socket_options = ( 
    HTTPConnection.default_socket_options + [
    (socket.SOL_SOCKET, socket.SO_SNDBUF, 1000000), #1MB in byte
    (socket.SOL_SOCKET, socket.SO_RCVBUF, 1000000)
])

#Variables
LARGE_MODEL_BY_LANGUAGE = {
    "Hindi": {"model_id": "ai4bharat/indicwav2vec-hindi", "has_lm": True},
}
LANGUAGES = sorted(LARGE_MODEL_BY_LANGUAGE.keys())
CACHED_MODELS_BY_ID = {}
# DEVICE_ID = 0 if torch.cuda.is_available() else -1
DEVICE_ID = -1

def getTranscription(input_file, language, decoding_type):
    history = {}
    model = LARGE_MODEL_BY_LANGUAGE.get(language, None)

    if decoding_type == "lm" and not model["has_lm"]:
        history = {
            "error_message": f"lm not available for {language} language :("
        }
    else:
        model_instance = CACHED_MODELS_BY_ID.get(model["model_id"], None)
        if model_instance is None:
            model_instance = AutoModelForCTC.from_pretrained(model["model_id"])
            CACHED_MODELS_BY_ID[model["model_id"]] = model_instance

        if decoding_type == "lm":
            processor = Wav2Vec2ProcessorWithLM.from_pretrained(model["model_id"])
            asr = pipeline("automatic-speech-recognition", model=model_instance, tokenizer=processor.tokenizer,
                          feature_extractor=processor.feature_extractor, decoder=processor.decoder, device=DEVICE_ID)
        else:
            processor = Wav2Vec2Processor.from_pretrained(model["model_id"])
            asr = pipeline("automatic-speech-recognition", model=model_instance, tokenizer=processor.tokenizer,
                          feature_extractor=processor.feature_extractor, decoder=None, device=DEVICE_ID)

        transcription = asr(input_file, chunk_length_s=5, stride_length_s=1)["text"]
        # logger.info(f"Transcription for {input_file}: {transcription}")
        history = {
            "model_id": model["model_id"],
            "language": language,
            "decoding_type": decoding_type,
            "transcription": transcription,
            "error_message": None
        }

    return {"text": history, "language": language}


def download_data_from_FTP(url):
    a = urlparse(url)
    name = os.path.basename(a.path)                     
    request_obj = requests.get(url)
    with open(name, "wb") as file:
        file.write(request_obj.content)
    file_path = name
    return file_path, name


# print(download_data_from_FTP("http://192.168.1.17:5500/file/mrvxuocbdz.wav"))