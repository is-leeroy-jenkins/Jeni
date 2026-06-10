"""Gemini provider wrappers for the Jeni Streamlit application.

Purpose:
    Provides application-facing wrapper classes for Gemini text generation, image generation
    and analysis, embeddings, transcription, translation, text-to-speech, Gemini file
    operations, file-search stores, and Google Cloud Storage bucket workflows.

    The module centralizes provider request construction, option lists, response extraction,
    and error logging so the Streamlit application can call stable Python interfaces instead
    of provider-specific SDK objects directly.

Classes:
    Gemini: Shared base class for Gemini wrapper configuration and runtime state.
    Chat: Text-generation wrapper for Gemini chat and grounding workflows.
    Images: Image generation, analysis, and editing wrapper.
    Embeddings: Text embedding wrapper.
    TTS: Text-to-speech wrapper.
    Transcription: Audio transcription wrapper.
    Translation: Audio translation wrapper.
    Files: Gemini file and document workflow wrapper.
    FileSearch: Gemini file-search store wrapper.
    CloudBuckets: Google Cloud Storage bucket wrapper.

Copyright:
    Copyright © 2024 Terry Eppler.

License:
    Permission is hereby granted, free of charge, to any person obtaining a copy of this
    software and associated documentation files to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the software, subject to inclusion of
    the copyright notice and permission notice in substantial portions of the software.

Contact:
    Terry Eppler, terryeppler@gmail.com or eppler.terry@epa.gov.
"""

from google.genai.file_search_stores import FileSearchStores
import config as cfg
import base64
from boogr import Error, Logger
import json
import os
import requests
import PIL.Image
from pathlib import Path
from typing import Any, List, Optional, Dict, Union
from google import genai
from google.cloud import storage
from google.genai import types
from google.genai.pagers import Pager
from google.genai.types import (
	Part,
	GenerateContentConfig,
	ImageConfig,
	FunctionCallingConfig,
	GenerateImagesConfig,
	GenerateVideosConfig,
	ThinkingConfig,
	GeneratedImage,
	EmbedContentConfig,
	Content,
	ContentEmbedding,
	Candidate,
	HttpOptions,
	GenerateImagesResponse,
	Field,
	FileSearchStore,
	FileSearch,
	GenerateContentResponse,
	GenerateVideosResponse,
	Image,
	File,
	SpeakerVoiceConfig,
	VoiceConfig,
	SpeechConfig,
	Tool,
	ToolConfig,
	GoogleSearch,
	UrlContext,
	SafetySetting,
	HarmCategory,
	HarmBlockThreshold,
)

def throw_if( name: str, value: object ) -> None:
	"""Validate that a required argument has a usable value.

	Purpose:
		Provides a shared argument guard for provider calls, file operations, and configuration
		builders. The helper raises a clear ``ValueError`` when a required value is missing,
		blank, or an empty collection.

	Args:
		name (str): Name of the argument being validated.
		value (object): Runtime value to validate.

	Raises:
		ValueError: Raised when ``value`` is ``None``, blank, or an empty collection.
	"""
	if value is None:
		raise ValueError( f'Argument "{name}" cannot be empty!' )
	
	if isinstance( value, str ) and not value.strip( ):
		raise ValueError( f'Argument "{name}" cannot be empty!' )
	
	if isinstance( value, (list, tuple, dict, set) ) and len( value ) == 0:
		raise ValueError( f'Argument "{name}" cannot be empty!' )

def encode_image( image_path: str ) -> str:
	"""Encode a local image file as base64 text.

	Purpose:
		Reads a local image file and converts its binary content into a base64-encoded string.
		This supports workflows that need inline image data instead of a file handle or URI.

	Args:
		image_path (str): Path to the local image file.

	Returns:
		str: Base64-encoded image content.
	"""
	with open( image_path, "rb" ) as image_file:
		return base64.b64encode( image_file.read( ) ).decode( "utf-8" )

class Gemini:
	"""Shared base class for Gemini wrapper configuration and runtime state.

	Purpose:
		Provides common configuration fields used by the Gemini wrapper classes. The base class
		centralizes API keys, model settings, sampling options, modality settings, tool settings,
		and response placeholders used by chat, image, embedding, audio, file, file-search, and
		storage workflows.

	Attributes:
		number (Optional[int]): Candidate or response-count setting.
		google_api_key (Optional[str]): Google API key loaded from configuration.
		gemini_api_key (Optional[str]): Gemini API key loaded from configuration.
		instructions (Optional[str]): System instruction text for model requests.
		prompt (Optional[str]): Prompt text for model requests.
		model (Optional[str]): Active Gemini model name.
		api_version (Optional[str]): Optional API version setting.
		max_tokens (Optional[int]): Maximum output-token setting.
		temperature (Optional[float]): Sampling temperature.
		top_p (Optional[float]): Top-p sampling value.
		top_k (Optional[int]): Top-k sampling value.
		candidate_count (Optional[int]): Candidate count for supported requests.
		media_resolution (Optional[str]): Requested media-resolution option.
		response_modalities (Optional[List[str]]): Requested response modalities.
		stops (Optional[List[str]]): Stop sequences.
		domains (Optional[List[str]]): Domain constraints for supported workflows.
		frequency_penalty (Optional[float]): Frequency penalty for supported requests.
		presence_penalty (Optional[float]): Presence penalty for supported requests.
		response_format (Optional[str]): Requested response MIME type.
		content_response (Optional[GenerateContentResponse]): Last content-generation response.
		image_response (Optional[GenerateImagesResponse]): Last image-generation response.
		content_config (Optional[GenerateContentConfig]): Last content-generation config.
		function_config (Optional[FunctionCallingConfig]): Function-calling configuration.
		thought_config (Optional[ThinkingConfig]): Thinking configuration.
		genimg_config (Optional[GenerateImagesConfig]): Image-generation configuration.
		image_config (Optional[ImageConfig]): Image-specific configuration.
		tool_config (Optional[List[types.Tool]]): Tool configuration.
		tool_choice (Optional[str]): Tool-choice mode.
		tools (Optional[List[str]]): Tool names selected by the caller.
	"""
	
	number: Optional[ int ]
	google_api_key: Optional[ str ]
	gemini_api_key: Optional[ str ]
	instructions: Optional[ str ]
	prompt: Optional[ str ]
	model: Optional[ str ]
	api_version: Optional[ str ]
	max_tokens: Optional[ int ]
	temperature: Optional[ float ]
	top_p: Optional[ float ]
	top_k: Optional[ int ]
	candidate_count: Optional[ int ]
	media_resolution: Optional[ str ]
	response_modalities: Optional[ List[ str ] ]
	stops: Optional[ List[ str ] ]
	domains: Optional[ List[ str ] ]
	frequency_penalty: Optional[ float ]
	presence_penalty: Optional[ float ]
	response_format: Optional[ str ]
	content_response: Optional[ GenerateContentResponse ]
	image_response: Optional[ GenerateImagesResponse ]
	content_config: Optional[ GenerateContentConfig ]
	function_config: Optional[ FunctionCallingConfig ]
	thought_config: Optional[ ThinkingConfig ]
	genimg_config: Optional[ GenerateImagesConfig ]
	image_config: Optional[ ImageConfig ]
	tool_config: Optional[ List[ types.Tool ] ]
	tool_choice: Optional[ str ]
	tools: Optional[ List[ str ] ]
	
	def __init__( self ) -> None:
		"""Initialize the shared Gemini wrapper state.

		Purpose:
			Initializes provider keys, model settings, sampling fields, request placeholders, and
			response placeholders used by child wrapper classes. The constructor performs local
			state assignment only and does not submit provider requests.
		"""
		self.google_api_key = cfg.GOOGLE_API_KEY
		self.gemini_api_key = cfg.GEMINI_API_KEY
		self.model = None
		self.api_version = None
		self.temperature = None
		self.top_p = None
		self.top_k = None
		self.candidate_count = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.max_tokens = None
		self.instructions = None
		self.prompt = None
		self.response_format = None
		self.number = None
		self.response_modalities = [ ]
		self.stops = [ ]
		self.tools = [ ]

class Chat( Gemini ):
	"""Gemini text-generation wrapper.

	Purpose:
		Handles Gemini text-generation workflows for the application. The class builds request
		contents, configures model options, coordinates optional grounding and tool settings,
		captures response metadata, and exposes generated text and structured history to callers.

	Attributes:
		use_vertex (Optional[bool]): Whether the wrapper is using Vertex configuration.
		http_options (Optional[HttpOptions]): Optional HTTP client configuration.
		client (Optional[genai.Client]): Gemini SDK client.
		storage_client (Optional[storage.Client]): Optional Google Cloud Storage client.
		contents (Optional[Union[str, List[str], List[Content]]]): Request contents.
		image_uri (Optional[str]): Optional image URI.
		audio_uri (Optional[str]): Optional audio URI.
		file_path (Optional[str]): Optional local file path.
		files (Optional[List[str]]): File paths or file identifiers used by the workflow.
		content_block (Optional[str]): Additional content prepended to a prompt.
		context (Optional[List[Dict[str, Any]]]): Structured chat history or context.
		urls (Optional[List[str]]): Reference URLs appended to the prompt.
		max_urls (Optional[int]): Maximum number of URLs to include.
		response_schema (Optional[Any]): Structured-output schema.
		safety_profile (Optional[str]): Safety-profile selector.
		safety_settings (Optional[List[SafetySetting]]): Gemini safety settings.
	"""
	
	use_vertex: Optional[ bool ]
	http_options: Optional[ HttpOptions ]
	client: Optional[ genai.Client ]
	storage_client: Optional[ storage.Client ]
	contents: Optional[ Union[ str, List[ str ], List[ Content ] ] ]
	image_uri: Optional[ str ]
	audio_uri: Optional[ str ]
	file_path: Optional[ str ]
	files: Optional[ List[ str ] ]
	content_block: Optional[ str ]
	context: Optional[ List[ Dict[ str, Any ] ] ]
	urls: Optional[ List[ str ] ]
	max_urls: Optional[ int ]
	response_schema: Optional[ Any ]
	safety_profile: Optional[ str ]
	safety_settings: Optional[ List[ SafetySetting ] ]
	
	def __init__( self, model: str = "gemini-2.5-flash-lite" ) -> None:
		"""Initialize the Chat wrapper.

		Purpose:
			Initializes default text-generation state, option placeholders, tool settings,
			grounding metadata, content history, and response placeholders used by later Chat
			method calls.

		Args:
			model (str): Default Gemini text-generation model.
		"""
		super( ).__init__( )
		self.gemini_api_key = cfg.GEMINI_API_KEY
		self.google_api_key = cfg.GOOGLE_API_KEY
		self.api_version = None
		self.client = None
		self.content_config = None
		self.image_config = None
		self.function_tool_config = None
		self.thought_config = None
		self.genimg_config = None
		self.tool_objects = None
		self.tools = [ ]
		self.response_modalities = [ ]
		self.files = [ ]
		self.http_options = { }
		self.number = None
		self.candidate_count = None
		self.model = model
		self.top_p = None
		self.top_k = None
		self.temperature = None
		self.frequency_penalty = None
		self.presence_penalty = None
		self.max_tokens = None
		self.use_vertex = None
		self.instructions = None
		self.media_resolution = None
		self.tool_choice = None
		self.contents = None
		self.grounding_metadata = None
		self.content_block = None
		self.context = [ ]
		self.client = None
		self.storage_client = None
		self.content_response = None
		self.image_response = None
		self.image_uri = None
		self.audio_uri = None
		self.file_path = None
		self.stops = [ ]
		self.response_mime_type = None
		self.response_schema = None
		self.urls = [ ]
		self.max_urls = None
		self.safety_profile = None
		self.safety_settings = None
		self.file_search_store_names = [ ]
		self.include_server_side_tool_invocations = None
	
	@property
	def model_options( self ) -> List[ str ]:
		"""Return supported Gemini text-generation models.

		Purpose:
			Provides model choices for Streamlit controls and documentation.

		Returns:
			List[str]: Supported Gemini text-generation model names.
		"""
		return [
				"gemini-2.5-flash",
				"gemini-2.5-flash-lite",
				"gemini-2.5-pro",
				"gemini-3-flash-preview",
				"gemini-3.1-flash-lite-preview",
				"gemini-3.1-pro-preview",
				"gemini-2.0-flash",
				"gemini-2.0-flash-lite",
		]
	
	@property
	def tool_options( self ) -> List[ str ]:
		"""Return supported tool names for text-generation workflows.

		Purpose:
			Provides selectable tool options for UI controls and request builders.

		Returns:
			List[str]: Supported tool names.
		"""
		return [
				"google_search",
				"google_maps",
				"url_context",
				"file_search",
				"code_execution",
		]
	
	@property
	def reasoning_options( self ) -> List[ str ]:
		"""Return supported reasoning-level options.

		Purpose:
			Provides selectable thinking-level values for supported Gemini models.

		Returns:
			List[str]: Supported reasoning-level option values.
		"""
		return [
				"THINKING_LEVEL_UNSPECIFIED",
				"MINIMAL",
				"LOW",
				"MEDIUM",
				"HIGH",
		]
	
	@property
	def media_options( self ) -> List[ str ]:
		"""Return supported media-resolution options.

		Purpose:
			Provides selectable media-resolution values for multimodal requests.

		Returns:
			List[str]: Supported media-resolution option values.
		"""
		return [
				"media_resolution_high",
				"media_resolution_medium",
				"media_resolution_low",
		]
	
	@property
	def choice_options( self ) -> List[ str ]:
		"""Return supported tool-choice options.

		Purpose:
			Provides selectable tool-choice values for request configuration.

		Returns:
			List[str]: Supported tool-choice values.
		"""
		return [ "auto", "any", "none", "validated" ]
	
	@property
	def include_options( self ) -> List[ str ]:
		"""Return supported include options.

		Purpose:
			Provides selectable include values for workflows that support additional response
			content.

		Returns:
			List[str]: Supported include option values.
		"""
		return [
				"file_search_call.results",
				"message.input_image.image_url",
				"message.output_text.logprobs",
				"reasoning.encrypted_content",
		]
	
	@property
	def modality_options( self ) -> List[ str ]:
		"""Return supported response-modality options.

		Purpose:
			Provides selectable modality values for supported Gemini requests.

		Returns:
			List[str]: Supported modality option values.
		"""
		return [ "", "text", "image", "audio" ]
	
	@property
	def format_options( self ) -> List[ str ]:
		"""Return supported response MIME types.

		Purpose:
			Provides selectable response-format values for supported Gemini requests.

		Returns:
			List[str]: Supported response MIME types.
		"""
		return [
				"text/plain",
				"application/json",
				"text/x.enum",
		]
	
	def get_supported_tools( self, model: str ) -> List[ str ]:
		"""Return tool options supported by a model.

		Purpose:
			Builds a model-specific tool list so the UI exposes only compatible options.

		Args:
			model (str): Gemini model name.

		Returns:
			List[str]: Supported tool names for the selected model.

		Raises:
			Error: Re-raised after validation or provider exceptions are wrapped and logged.
			ValueError: Raised when ``model`` is missing.
		"""
		try:
			throw_if( "model", model )
			self.model_name = str( model ).strip( ).lower( )
			self.options = [ "google_search", "url_context", "file_search", "code_execution" ]
			
			if self.supports_google_maps( self.model_name ):
				self.options.append( "google_maps" )
			
			return self.options
		except Exception as e:
			exception = Error( e )
			exception.module = "gemini"
			exception.cause = "Chat"
			exception.method = "get_supported_tools(self, model: str) -> List[str]"
			Logger( ).write( exception )
			raise exception
	
	def supports_google_maps( self, model: str ) -> bool:
		"""Return whether a model supports Google Maps grounding.

		Purpose:
			Centralizes feature gating for Google Maps support.

		Args:
			model (str): Gemini model name.

		Returns:
			bool: True when the selected model supports Google Maps grounding; otherwise False.

		Raises:
			Error: Re-raised after validation or provider exceptions are wrapped and logged.
			ValueError: Raised when ``model`` is missing.
		"""
		try:
			throw_if( "model", model )
			self.model_name = model.strip( ).lower( )
			self.maps_models = {
					"gemini-3.1-pro-preview",
					"gemini-3.1-flash-lite-preview",
					"gemini-3-flash-preview",
					"gemini-2.5-pro",
					"gemini-2.5-flash",
					"gemini-2.5-flash-lite",
					"gemini-2.0-flash",
			}
			return self.model_name in self.maps_models
		except Exception as e:
			exception = Error( e )
			exception.module = "gemini"
			exception.cause = "Chat"
			exception.method = "supports_google_maps(self, model: str) -> bool"
			Logger( ).write( exception )
			raise exception
	
	def build_urls( self, urls: List[ str ], max_urls: int = 10 ) -> List[ str ]:
		"""Build the URL context list for a text request.

		Purpose:
			Normalizes caller-provided URLs, removes blank values, and limits the list to the
			configured maximum.

		Args:
			urls (List[str]): Candidate URL strings.
			max_urls (int): Maximum number of URLs to include.

		Returns:
			List[str]: Normalized URL list.

		Raises:
			Error: Re-raised after validation or provider exceptions are wrapped and logged.
			ValueError: Raised when ``max_urls`` is missing.
		"""
		try:
			throw_if( "max_urls", max_urls )
			self.urls = [ ]
			
			for url in urls or [ ]:
				if url is None:
					continue
				
				self.url = url.strip( )
				if not self.url:
					continue
				
				self.urls.append( self.url )
			
			self.max_urls = max_urls
			if self.max_urls is not None:
				self.urls = self.urls[ : self.max_urls ]
			
			return self.urls
		except Exception as e:
			exception = Error( e )
			exception.module = "gemini"
			exception.cause = "Chat"
			exception.method = "build_urls(self, urls: List[str], max_urls: int = 10) -> List[str]"
			Logger( ).write( exception )
			raise exception
		