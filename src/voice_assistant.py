from typing import List, Optional

import speech_recognition as sr

from spotify import Spotify
from vision.commands import (
    GestureCommand,
    open_spotify_command_factory,
    play_next_track_command_factory,
    play_prev_track_command_factory,
    set_volume_command_factory,
    shuffle_saved_tracks_command_factory,
)


class VoiceAssistant:
    """
    Primary class for Natasha's Voice Assistant functionality.
    """

    WAKE_WORD = "natasha"

    def __init__(self) -> None:
        self.voice_commands = []
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.stop_listening = None
        self.active = True

    def register_commands(self, commands: List[GestureCommand]):
        self.voice_commands.extend(commands)

    def activate(self):
        self.start_listening()
        try:
            while self.active:
                pass
        except KeyboardInterrupt:
            self.stop()
        finally:
            self.stop()
            print("Listening terminated. Goodbye!")

    def start_listening(self):
        self.stop_listening = self.recognizer.listen_in_background(
            self.microphone, self.callback
        )

    def callback(self, recognizer, audio):
        speech_as_text = self.recognize_speech(recognizer, audio)
        if speech_as_text and self.WAKE_WORD in speech_as_text.lower():
            self.handle_command(speech_as_text)

    def recognize_speech(self, recognizer, audio) -> Optional[str]:
        try:
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
        return None

    def handle_command(self, speech_as_text: str):
        if "stop listening" in speech_as_text.lower():
            self.active = False
            print("Stopping voice recognition.")
            return

        command_name = self.extract_command(speech_as_text)
        command = self.get_activated_command(command_name)
        if command:
            print(f"{command.name} triggered by `{speech_as_text}`")
            self.execute_command(command)

    def extract_command(self, speech_as_text: str) -> str:
        return "".join(
            speech_as_text.lower().replace(self.WAKE_WORD, "").strip().split(" ")
        )

    def get_activated_command(self, command_name: str) -> Optional[GestureCommand]:
        for command in self.voice_commands:
            if command.name.lower() in command_name:
                return command
        return None

    def execute_command(self, command: GestureCommand):
        try:
            command.callback()
        except Exception as e:
            print(f"Error executing command {command.name}: {e}")

    def stop(self):
        if self.stop_listening:
            self.stop_listening(wait_for_stop=False)


if __name__ == "__main__":
    spotify_client = Spotify()

    voice_commands = [
        open_spotify_command_factory(),
        shuffle_saved_tracks_command_factory(spotify_client),
        play_next_track_command_factory(spotify_client),
        play_prev_track_command_factory(spotify_client),
        set_volume_command_factory(spotify_client),
    ]

    assistant = VoiceAssistant()
    assistant.register_commands(voice_commands)
    assistant.activate()
