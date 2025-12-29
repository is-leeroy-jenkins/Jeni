'''
  ******************************************************************************************
      Assembly:                Jeni
      Filename:                Config.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="webconfig.py" company="Terry D. Eppler">

	     Jeni is a df analysis tool integrating GenAI, GptText Processing, and Machine-Learning
	     algorithms for federal analysts.
	     Copyright ©  2022  Terry Eppler

     Permission is hereby granted, free of charge, to any person obtaining a copy
     of this software and associated documentation files (the “Software”),
     to deal in the Software without restriction,
     including without limitation the rights to use,
     copy, modify, merge, publish, distribute, sublicense,
     and/or sell copies of the Software,
     and to permit persons to whom the Software is furnished to do so,
     subject to the following conditions:

     The above copyright notice and this permission notice shall be included in all
     copies or substantial portions of the Software.

     THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
     INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
     FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
     IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
     ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
     DEALINGS IN THE SOFTWARE.

     You can contact me at:  terryeppler@gmail.com or eppler.terry@epa.gov

  </copyright>
  <summary>
    config.py
  </summary>
  ******************************************************************************************
'''
import os
from typing import Optional, List, Dict
from pathlib import Path

GEOCODING_API_KEY = os.getenv( 'GEOCODING_API_KEY' )
GEMINI_API_KEY = os.getenv( 'GEMINI_API_KEY' )
GROQ_API_KEY = os.getenv( 'GROQ_API_KEY' )
GOOGLE_API_KEY = os.getenv( 'GOOGLE_API_KEY' )
GOOGLE_CSE_ID = os.getenv( 'GOOGLE_CSE_ID' )
GOOGLE_CLOUD_LOCATION = os.getenv( 'GOOGLE_CLOUD_LOCATION' )
GOOGLE_CLOUD_PROJECT = os.getenv( 'GOOGLE_CLOUD_PROJECT' )
HUGGINGFACE_API_KEY = os.getenv( 'HUGGINGFACE_API_KEY' )
OPENAI_API_KEY = os.getenv( 'OPENAI_API_KEY' )
OUTPUT_FILE_NAME = "jeni.wav"
SAMPLE_RATE = 48000
MODELS = [ 'gpt-5-nano-2025-08-07', 'gpt-4.1-nano-2025-04-14', 'gpt-4o-mini', ]
DEFAULT_MODEL = MODELS[ 0 ]
SQLALCHEMY_DATABASE_URI = f'sqlite:///' + r'C:\Users\terry\source\repos\Jeni\stores\sqlite\datamodels\Data.db'
BASE_DIR = Path(__file__).resolve().parent
FAVICON_PATH = BASE_DIR / 'resources' / 'images' / 'favicon.ico'

def set_environment( ):
	"""

		Purpose:
		--------
		Gets availible environment vaariables for configuration


	"""
	variable_dict = globals( ).items( )
	for key, value in variable_dict:
		if 'API' in key or 'ID' in key:
			os.environ[ key ] = value

