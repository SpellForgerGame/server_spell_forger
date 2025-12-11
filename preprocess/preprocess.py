

import nltk
from nltk.corpus import stopwords
from googletrans import Translator
import asyncio

nltk.download('stopwords')

_translator_detector = Translator(service_urls=['translate.google.com', 'translate.google.co.kr'])

DEFAULT_LANGUAGE = 'pt'  # Default language if detection fails or is not specified

async def _detect_language(text):
    """
    Detects the language of the given text using googletrans.
    Returns a 2-letter language code (e.g., 'en', 'pt').
    """
    if not text.strip(): # Handle empty strings
        return DEFAULT_LANGUAGE # Default language for empty input

    try:
        # The detect method returns a Detection object, which has a .lang attribute
        detection = await _translator_detector.detect(text)
        return detection.lang
    except Exception as e:
        #print(f"Error detecting language for text: '{text[:50]}...': {e}. Defaulting to '{DEFAULT_LANGUAGE}'.")
        return DEFAULT_LANGUAGE # Fallback language if detection fails


english_stop_words = set(stopwords.words('english'))
portuguese_stop_words = set(stopwords.words('portuguese'))

stop_words_mapping = {
    'en': english_stop_words,
    'pt': portuguese_stop_words
}

def remove_stop_words(text, language):
    stop_words = stop_words_mapping.get(language, set())
    words = text.split()
    return ' '.join([word for word in words if word.lower() not in stop_words])


async def preprocess_prompt(text):
    
    language = await _detect_language(text)
    cleaned_text = remove_stop_words(text, language)
    
    # Additional preprocessing can be added here
    return cleaned_text.lower().strip()


async def main(): # Define an async main function
    text = input("Enter your prompt: ")
    cleaned_text = await preprocess_prompt(text) # Await the async function
    print(f"Cleaned text: {cleaned_text}")

if __name__ == "__main__":
    asyncio.run(main()) # Run the async main function