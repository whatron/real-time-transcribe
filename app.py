import asyncio
import sys
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import soundfile as sf

async def record_buffer(buffer, samplerate, **kwargs):
    loop = asyncio.get_event_loop()
    event = asyncio.Event()
    idx = 0

    def callback(indata, frame_count, time_info, status):
        nonlocal idx
        if status:
            print(status)
        remainder = len(buffer) - idx
        if remainder == 0:
            loop.call_soon_threadsafe(event.set)
            raise sd.CallbackStop
        indata = indata[:remainder]
        buffer[idx:idx + len(indata)] = indata
        idx += len(indata)

    stream = sd.InputStream(callback=callback, dtype=buffer.dtype,
                            channels=buffer.shape[1],samplerate=samplerate, **kwargs)
    with stream:
        await event.wait()

def load_model(model_size="base"):
    try:
        # Run on GPU with FP16
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
        print("Running on GPU with FP16")
    except:
        try:
            # Run on CPU with FP32
            model = WhisperModel(model_size, device="cpu", compute_type="float32")
            print("Running on CPU with FP32")
        except:
            try:
                # or run on GPU with INT8
                model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
                print("Running on GPU with INT8")
            except:
                try:
                    # or run on CPU with INT8
                    model = WhisperModel(model_size, device="cpu", compute_type="int8")
                    print("Running on CPU with INT8")
                except:
                    raise Exception("No supported device found")
        return model, model.compute_type


async def transcribe(model):

    segments, info = model.transcribe("op.wav", beam_size=5, vad_filter=True, word_timestamps=True)

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    word_list = []

    for segment in segments:
        for word in segment.words:
            word_list.append(word)
            print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))
            
    # print(word_list)

async def main(samplerate=8000, model=load_model(), channels=1, dtype='float32', **kwargs):
    buffer = np.empty((samplerate * 10, channels), dtype=dtype)
    print('recording ...')
    await record_buffer(buffer, samplerate, **kwargs)
    print('done')

    # sf.write('op.wav', buffer, samplerate)
    await transcribe(model)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit('\nInterrupted by user')