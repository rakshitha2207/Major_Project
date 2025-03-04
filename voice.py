import edge_tts
import pygame
import asyncio

pygame.mixer.init()
async def test_tts():
    await edge_tts.Communicate("Hello, Master", "en-US-MichelleNeural").save("test.mp3")
    pygame.mixer.music.load("test.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
asyncio.run(test_tts())