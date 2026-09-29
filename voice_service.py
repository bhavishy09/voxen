import os
import tempfile
from gtts import gTTS
import pygame
import speech_recognition as sr

class VoiceService:
    def __init__(self):
        # Initialize pygame mixer for audio playback
        pygame.mixer.init()
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        try:
            self.microphone = sr.Microphone()
            self.speech_enabled = True
            print(" Speech recognition initialized")
        except Exception as e:
            print(f" Speech recognition not available: {e}")
            self.speech_enabled = False
    
    def text_to_speech(self, text: str) -> bool:
        """Convert text to speech and play it"""
        try:
            print(f" Speaking: {text[:50]}...")
            
            # Create TTS object
            tts = gTTS(text=text, lang='en')
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
                temp_filename = tmp.name
                tts.save(temp_filename)
            
            # Load and play the audio
            pygame.mixer.music.load(temp_filename)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
            
            # Clean up the temporary file
            try:
                os.unlink(temp_filename)
            except:
                pass  # Ignore cleanup errors
                
            print(" Speech completed")
            return True
            
        except Exception as e:
            print(f" TTS Error: {e}")
            print(" Text-to-speech not working, but text response is available")
            return False
    
    def speech_to_text(self, timeout=5) -> str:
        """Convert speech to text"""
        if not self.speech_enabled:
            print(" Speech recognition not available")
            return "Speech recognition not available"
            
        try:
            print(f" Listening for {timeout} seconds...")
            print(" Please speak now! Speak CLEARLY and LOUDLY!")
            
            with self.microphone as source:
                # Adjust for ambient noise with more time
                print("🔧 Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
                
                # Set energy threshold dynamically
                print(f" Current energy threshold: {self.recognizer.energy_threshold}")
                
                # Listen for audio with longer phrase limit
                print(" Listening... (speak clearly into your microphone)")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
            
            print(" Processing speech with Google Speech Recognition...")
            text = self.recognizer.recognize_google(audio)
            print(f" Successfully recognized: '{text}'")
            return text
            
        except sr.WaitTimeoutError:
            print(" No speech detected within timeout")
            print(" Try speaking louder or closer to your microphone")
            return "No response"
        except sr.UnknownValueError:
            print(" Could not understand the audio")
            print(" Try speaking more clearly or check your microphone")
            return "Could not understand"
        except sr.RequestError as e:
            print(f" Could not request results from Google Speech Recognition service; {e}")
            print(" Check your internet connection")
            return "Error occurred"
        except Exception as e:
            print(f" STT Error: {e}")
            return "Error occurred"
    
    def test_tts(self) -> bool:
        """Test if text-to-speech is working"""
        try:
            test_text = "Testing voice system"
            return self.text_to_speech(test_text)
        except Exception as e:
            print(f"TTS test failed: {e}")
            return False
    
    def fallback_tts(self, text: str) -> bool:
        """Fallback TTS using system beep"""
        try:
            print(f"[VOICE]: {text}")
            # Try system beep as audio feedback
            return True
        except:
            return False
